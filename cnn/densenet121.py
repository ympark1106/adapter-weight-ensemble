import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")
import sys
sys.path.append("/home/youmin/workspace/VFMs-Adapters-Ensemble/adapter_ensemble")
import os
import time
from datetime import timedelta
import torch
import timm
import torch.nn as nn
import argparse
import numpy as np
from torchvision import models

from utils import read_conf, validation_accuracy, ModelWithTemperature, validate, evaluate
from data import cifar10, ham10000

device = 'cuda:0'

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='cifar10')
    parser.add_argument('--gpu', '-g', default = '0', type=str)
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
    elif args.data == 'ham10000':
        train_loader, valid_loader, test_loader = ham10000.get_dataloaders(data_path, batch_size=32, num_workers=4)
    
    
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, config['num_classes']) 
    model = model.to(device)    

    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir= save_path, max_history = 1) 
    
    avg_accuracy = 0.0
    start_time = time.time()
    patience = 10
    best_valid_accuracy = 0.0
    epochs_no_improve = 0
    
    for epoch in range(max_epoch):
        epoch_start_time = time.time()
        ## training
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()            
            optimizer.step()

            total_loss += loss
            total += targets.size(0)
            _, predicted = outputs[:len(targets)].max(1)            
            correct += predicted.eq(targets).sum().item()            
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct/total, correct, total), end = '')
            train_accuracy = correct/total
                  
        train_avg_loss = total_loss/len(train_loader)
        epoch_duration = time.time() - epoch_start_time
        epoch_time = str(timedelta(seconds=epoch_duration))
        remaining_time = (max_epoch - (epoch + 1)) * epoch_duration
        formatted_remaining_time = str(timedelta(seconds=remaining_time))
        print(f"\nEpoch {epoch} took {epoch_time}")
        print(f"Estimated remaining training time: {formatted_remaining_time}")
        print()
        print()

        
        model.eval()
        total_loss = 0
        total = 0
        correct = 0

        valid_accuracy = validation_accuracy(model, valid_loader, device, mode = 'resnet')
        scheduler.step()
        
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            epochs_no_improve = 0  # Reset counter if validation improves
            saver.save_checkpoint(epoch, metric=valid_accuracy)
        else:
            epochs_no_improve += 1
            
        # Early stopping condition
        # if epochs_no_improve >= patience:
        #     print(f'Early stopping triggered after {epoch + 1} epochs.')
        #     break

        saver.save_checkpoint(epoch, metric = valid_accuracy)
        
        print(f'Epoch {epoch + 1}/{max_epoch} | Loss: {train_avg_loss:.4f} | '
              f'Train Acc: {train_accuracy:.4f} | Valid Acc: {valid_accuracy:.4f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

    total_duration = time.time() - start_time
    totoal_time = str(timedelta(seconds=total_duration))
    print(f"Total training time: {totoal_time}")
    
if __name__ =='__main__':
    train() 
    