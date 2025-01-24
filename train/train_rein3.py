import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

import os
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


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_requires_grad(model, layers_to_train):
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_train):
            param.requires_grad = True
        else:
            param.requires_grad = False


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='cub')
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    # parser.add_argument('--save_path', '-s', type=str)
    # parser.add_argument('--noise_rate', '-n', type=float, default=0.2)
    args = parser.parse_args()

    # config = utils.read_conf('conf/'+args.data+'.json')
    config = read_conf('conf/data/'+args.data+'.yaml')
    device = 'cuda:'+args.gpu
    save_path = os.path.join(config['save_path'], args.save_path)
    data_path = config['data_root']
    batch_size = int(config['batch_size'])
    max_epoch = int(config['epoch'])
    # noise_rate = args.noise_rate

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    lr_decay = [int(0.5*max_epoch), int(0.75*max_epoch), int(0.9*max_epoch)]

    # if args.data == 'ham10000':
    #     train_loader, valid_loader = utils.get_dataset(data_path, batch_size = batch_size)
    # elif args.data == 'aptos':
    #     train_loader, valid_loader = utils.get_aptos_noise_dataset(data_path, noise_rate=noise_rate, batch_size = batch_size)
        
    if args.data == 'cifar10':
        train_loader, valid_loader = cifar10.get_train_valid_loader(batch_size, augment=True, random_seed=42, valid_size=0.1, shuffle=True, num_workers=4, pin_memory=True, get_val_temp=0, data_dir=data_path)
    elif args.data == 'cub':
        train_loader, valid_loader = cub.get_train_val_loader(data_path, batch_size=32, scale_size=256, crop_size=224, num_workers=4, pin_memory=True)
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
    
    # variant['embed_dim'] = int(variant['embed_dim'] * 0.33) # 각 head의 embed_dim을 1/3로 줄임

    model = rein.ReinsDinoVisionTransformer_3_head(
        **variant,
        # token_lengths = [33, 33, 33]
        token_lengths = [100, 100, 100]
        # token_lengths= [85, 85, 85]
        # token_lengths= [256, 256, 256]
    )
    set_requires_grad(model, ["reins1", "reins2", "reins3",  "linear"])
    model.load_state_dict(dino_state_dict, strict=False)
    model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.to(device)
    
    print(model)
    
    num_params = count_trainable_params(model)
    print(f"Number of trainable parameters: {num_params}")
    # calculate_flops(model, input_size=(1, 3, 224, 224))
    
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)
    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir= save_path, max_history = 1) 
    # print(train_loader.dataset[0][0].shape)

    # f = open(os.path.join(save_path, 'epoch_acc.txt'), 'w')
    avg_accuracy = 0.0
    for epoch in range(max_epoch):
        ## training
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if targets.ndim > 1 and targets.size(1) > 1:
                targets = torch.argmax(targets, dim=1)

            optimizer.zero_grad()            
            features = model.forward_features1(inputs)
            features = features[:, 0, :]
            outputs1 = model.linear(features)

            features = model.forward_features2(inputs)
            features = features[:, 0, :]
            outputs2 = model.linear(features)

            features = model.forward_features3(inputs)
            features = features[:, 0, :]
            outputs3 = model.linear(features)

            loss = criterion(outputs1, targets) + criterion(outputs2, targets) + criterion(outputs3, targets)
            loss.backward()            

            optimizer.step()

            outputs = outputs1 + outputs1 + outputs3
            total_loss += loss
            total += targets.size(0)
            _, predicted = outputs[:len(targets)].max(1)            
            correct += predicted.eq(targets).sum().item()            
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct/total, correct, total), end = '')
            train_accuracy = correct/total
                  
        train_avg_loss = total_loss/len(train_loader)
        print()

        ## validation
        model.eval()
        total_loss = 0
        total = 0
        correct = 0

        valid_accuracy = validation_accuracy(model, valid_loader, device, mode='rein3')
        if epoch >= max_epoch-10:
            avg_accuracy += valid_accuracy 
        scheduler.step()

        saver.save_checkpoint(epoch, metric = valid_accuracy)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy))
        print(scheduler.get_last_lr())

    with open(os.path.join(save_path, 'avgacc.txt'), 'w') as f:
        f.write(str(avg_accuracy/10))
    
if __name__ =='__main__':
    train()