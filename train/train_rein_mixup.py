import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn as nn

import argparse
import timm
import numpy as np
from utils import read_conf, validation_accuracy, calculate_flops

import random
import rein

import dino_variant
from sklearn.metrics import f1_score
from data import cifar10, cub, ham10000
from losses import RankMixup_MNDCG  

from fvcore.nn import FlopCountAnalysis



def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_requires_grad(model, layers_to_train):
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_train):
            param.requires_grad = True
        else:
            param.requires_grad = False


def mixup_data(self, x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam



def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='cub')
    parser.add_argument('--gpu', '-g', default='0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    args = parser.parse_args()

    # config = utils.read_conf('conf/'+args.data+'.json')
    config = read_conf('conf/data/' + args.data + '.yaml')
    device = 'cuda:' + args.gpu
    save_path = os.path.join(config['save_path'], args.save_path)
    data_path = config['data_root']
    batch_size = int(config['batch_size'])
    max_epoch = int(config['epoch'])

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    lr_decay = [int(0.5 * max_epoch), int(0.75 * max_epoch), int(0.9 * max_epoch)]

    if args.data == 'cifar10':
        train_loader, valid_loader = cifar10.get_train_valid_loader(batch_size, augment=True, random_seed=42, valid_size=0.1, shuffle=True, num_workers=4, pin_memory=True, get_val_temp=0, data_dir=data_path)
    elif args.data == 'cub':
        train_loader, valid_loader = cub.get_train_val_loader(data_path, batch_size=32, scale_size=256, crop_size=224, num_workers=8, pin_memory=True)
    elif args.data == 'ham10000':
        # Train DataLoader
        train_loader, _ = ham10000.create_dataloader(
            annotations_file=os.path.join(data_path, 'ISIC2018_Task3_Training_GroundTruth.csv'),
            img_dir=os.path.join(data_path, 'train/'),
            batch_size=batch_size,
            shuffle=True,
            transform_mode='augment'
        )

        # Validation DataLoader
        valid_loader, _ = ham10000.create_dataloader(
            annotations_file=os.path.join(data_path, 'ISIC2018_Task3_Validation_GroundTruth.csv'),
            img_dir=os.path.join(data_path, 'valid/'),
            batch_size=batch_size,
            shuffle=False,
            transform_mode='base'
        )

    if args.netsize == 's':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant
    elif args.netsize == 'b':
        model_load = dino_variant._base_dino
        variant = dino_variant._base_variant
    elif args.netsize == 'l':
        model_load = dino_variant._large_dino
        variant = dino_variant._large_variant

    model = torch.hub.load('facebookresearch/dinov2', model_load)
    dino_state_dict = model.state_dict()

    model = rein.ReinsDinoVisionTransformer(
        **variant
    )
    set_requires_grad(model, ["reins", "linear"])
    model.load_state_dict(dino_state_dict, strict=False)
    model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.to(device)
    
    print(model)
    
    num_params = count_trainable_params(model)
    print(f"Number of trainable parameters: {num_params}")

    criterion = RankMixup_MNDCG(num_classes=config['num_classes'], alpha=0.1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)
    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir=save_path, max_history=1)


    avg_accuracy = 0.0
    for epoch in range(max_epoch):
        ## training
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            mixed_inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=1.0)

            optimizer.zero_grad()

            features = model.forward_features(mixed_inputs)
            features = features[:, 0, :]
            outputs = model.linear(features)

            loss, loss_ce, loss_mixup = criterion(outputs, targets_a, outputs, targets_b, lam)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total += targets.size(0)
            _, predicted = outputs[:len(targets)].max(1)
            correct += predicted.eq(targets).sum().item()
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (total_loss / (batch_idx + 1), 100. * correct / total, correct, total), end='')

        train_avg_loss = total_loss / len(train_loader)
        print()

        ## validation
        model.eval()
        valid_accuracy = validation_accuracy(model, valid_loader, device)
        if epoch >= max_epoch - 10:
            avg_accuracy += valid_accuracy
        scheduler.step()

        saver.save_checkpoint(epoch, metric=valid_accuracy)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}]\n'.format(epoch, train_avg_loss, 100. * correct / total, valid_accuracy))

    with open(os.path.join(save_path, 'avgacc.txt'), 'w') as f:
        f.write(str(avg_accuracy / 10))

if __name__ == '__main__':
    train()