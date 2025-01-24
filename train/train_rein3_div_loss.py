import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import timm
import numpy as np
import utils

import random
import rein

import dino_variant
from sklearn.metrics import f1_score

def js_loss_compute(pred, soft_targets, reduce=True):    
    pred_softmax = F.softmax(pred, dim=1)
    targets_softmax = F.softmax(soft_targets, dim=1)
    mean = (pred_softmax + targets_softmax) / 2
    kl_1 = F.kl_div(F.log_softmax(pred, dim=1), mean, reduce=False)
    kl_2 = F.kl_div(F.log_softmax(soft_targets, dim=1), mean, reduce=False)
    js = (kl_1 + kl_2) / 2 
    
    if reduce:
        return torch.mean(torch.sum(js, dim=1))
    else:
        return torch.sum(js, 1)

def three_js_loss(pred1, pred2, pred3):
    js12 = js_loss_compute(pred1, pred2)
    js13 = js_loss_compute(pred1, pred3)
    js23 = js_loss_compute(pred2, pred3)

    return js12 + js13 + js23 

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--noise_rate', '-n', type=float, default=0.2)
    args = parser.parse_args()

    config = utils.read_conf('conf/'+args.data+'.json')
    device = 'cuda:'+args.gpu
    save_path = os.path.join(config['save_path'], args.save_path)
    data_path = config['id_dataset']
    batch_size = int(config['batch_size'])
    max_epoch = int(config['epoch'])
    noise_rate = args.noise_rate

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    lr_decay = [int(0.5*max_epoch), int(0.75*max_epoch), int(0.9*max_epoch)]

    if args.data == 'ham10000':
        train_loader, valid_loader = utils.get_dataset(data_path, batch_size = batch_size)
    elif args.data == 'aptos':
        train_loader, valid_loader = utils.get_aptos_noise_dataset(data_path, noise_rate=noise_rate, batch_size = batch_size)

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

    model = rein.ReinsDinoVisionTransformer_3_head(
        **variant,
        token_lengths = [33, 33, 33]
    )
    model.load_state_dict(dino_state_dict, strict=False)
    model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.to(device)
    
    print(model)
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)
    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir= save_path, max_history = 1) 
    print(train_loader.dataset[0][0].shape)

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
            js_loss = three_js_loss(outputs1, outputs2, outputs3)

            loss = loss - .25 * js_loss
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

        valid_accuracy = utils.validation_accuracy(model, valid_loader, device, mode='rein3')
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