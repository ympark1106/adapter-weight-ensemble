import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")
import sys
sys.path.append("/home/youmin/workspace/VFMs-Adapters-Ensemble/adapter_ensemble")
import time
from datetime import timedelta

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = '1'
import torch
import torch.nn as nn

import argparse
import timm
import numpy as np
from utils import read_conf, validation_accuracy

import random
import rein

import dino_variant
from sklearn.metrics import f1_score
from data import cifar10, cifar100, cub, ham10000, bloodmnist, pathmnist, retinamnist
from losses import RankMixup_MNDCG, RankMixup_MRL, focal_loss, focal_loss_adaptive_gamma

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
    args = parser.parse_args()

    config = read_conf('conf/data/'+args.data+'.yaml')
    device = 'cuda:'+args.gpu
    save_path = os.path.join(config['save_path'], args.save_path)
    data_path = config['data_root']
    batch_size = int(config['batch_size'])
    max_epoch = int(config['epoch'])

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if args.data == 'cifar10':
        train_loader, valid_loader = cifar10.get_train_valid_loader(batch_size, augment=True, random_seed=42, valid_size=0.1, shuffle=True, num_workers=4, pin_memory=True, get_val_temp=0, data_dir=data_path)
    elif args.data == 'cifar100':
        train_loader, valid_loader = cifar100.get_train_valid_loader(data_dir=data_path, augment=True, batch_size=32, valid_size=0.1, random_seed=42, shuffle=True, num_workers=4, pin_memory=True)
    elif args.data == 'cub':
        train_loader, valid_loader = cub.get_train_val_loader(data_path, batch_size=32, scale_size=256, crop_size=224, num_workers=8, pin_memory=True)
    elif args.data == 'ham10000':
        train_loader, valid_loader, test_loader = ham10000.get_dataloaders(data_path, batch_size=32, num_workers=4)
    elif args.data == 'bloodmnist':
        train_loader, valid_loader,_ = bloodmnist.get_dataloader(batch_size, download=True, num_workers=4)
    elif args.data == 'pathmnist':
        train_loader, valid_loader,_ = pathmnist.get_dataloader(batch_size, download=True, num_workers=4)
    elif args.data == 'retinamnist':
        train_loader, valid_loader,_ = retinamnist.get_dataloader(batch_size, download=True, num_workers=4)

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

    swa_model = rein.ReinsDinoVisionTransformer(**variant)
    swa_model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    swa_model.to(device)

    swa_n = 0  # Counter for SWA updates
    print(model)

    print("Max epoch: ", max_epoch)

    criterion = focal_loss.FocalLoss(gamma=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    lr_decay_epochs = 70
    lr_scheduler_decay = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 70], gamma=0.1)

    total_cyclic_epochs = max_epoch - lr_decay_epochs
    print("Total Cyclic Epochs: ", total_cyclic_epochs)
    cyclic_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=1e-5)

    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir=save_path, max_history=1)

    torch.save(model.state_dict(), os.path.join(save_path, f'cyclic_checkpoint_epoch{max_epoch}.pth'))

    avg_accuracy = 0.0
    start_time = time.time()

    for epoch in range(max_epoch):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            targets = targets.type(torch.LongTensor)
            inputs, targets = inputs.to(device), targets.to(device)           
            if targets.ndim > 1 and targets.size(1) > 1:
                targets = torch.argmax(targets, dim=1)
            if targets.ndim > 1:
                targets = targets.view(-1) 
            optimizer.zero_grad()
            features = model.forward_features(inputs)
            features = features[:, 0, :]
            outputs = model.linear(features)
            loss = criterion(outputs, targets)
            loss.backward()            
            optimizer.step()

            total_loss += loss
            total += targets.size(0)
            _, predicted = outputs[:len(targets)].max(1)        
            correct += predicted.eq(targets).sum().item()            
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct/total, correct, total), end='')
            train_accuracy = correct/total

        if epoch < lr_decay_epochs:
            lr_scheduler_decay.step()
        else:
            cyclic_scheduler.step()

        train_avg_loss = total_loss / len(train_loader)
        epoch_duration = time.time() - epoch_start_time
        epoch_time = str(timedelta(seconds=epoch_duration))
        remaining_time = (max_epoch - (epoch + 1)) * epoch_duration
        formatted_remaining_time = str(timedelta(seconds=remaining_time))
        print(f"\nEpoch {epoch} took {epoch_time}")
        print(f"Estimated remaining training time: {formatted_remaining_time}")

        # SWA Updates
        # if epoch >= lr_decay_epochs and (epoch - lr_decay_epochs+1) % 30 == 0:
        if optimizer.param_groups[0]['lr'] <= 0.00002:
            with torch.no_grad():
                for swa_param, param in zip(swa_model.parameters(), model.parameters()):
                    swa_param.data *= swa_n / (swa_n + 1)
                    swa_param.data += param.data / (swa_n + 1)
                swa_n += 1

        model.eval()
        valid_accuracy = validation_accuracy(model, valid_loader, device)
        if epoch >= max_epoch - 10:
            avg_accuracy += valid_accuracy

        saver.save_checkpoint(epoch, metric=valid_accuracy)

        print(f'Epoch {epoch + 1}/{max_epoch} | Loss: {train_avg_loss:.4f} | '
              f'Train Acc: {train_accuracy:.4f} | Valid Acc: {valid_accuracy:.4f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

    # Save final SWA model
    torch.save(swa_model.state_dict(), os.path.join(save_path, 'swa_model_final.pth'))

    total_duration = time.time() - start_time
    totoal_time = str(timedelta(seconds=total_duration))
    print(f"Total training time: {totoal_time}")

    with open(os.path.join(save_path, 'avgacc.txt'), 'w') as f:
        f.write(str(avg_accuracy / 10))

if __name__ == '__main__':
    train()
