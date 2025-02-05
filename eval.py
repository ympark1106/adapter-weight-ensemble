import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

import torch
import torch.nn as nn

import argparse
import timm
import numpy as np
from utils import read_conf, validation_accuracy, evaluate, validation_accuracy_lora    
from torch.cuda.amp.autocast_mode import autocast

import random
import rein

import dino_variant
from data import cifar10, cifar100, cub, ham10000, bloodmnist, pathmnist, retinamnist


def rein_forward(model, inputs):
    output = model.forward_features(inputs)[:, 0, :]
    output = model.linear(output)
    output = torch.softmax(output, dim=1)
    return output

def lora_forward(model, inputs):
    with autocast(enabled=True):
        features = model.forward_features(inputs)
        output = model.linear(features)
        output = torch.softmax(output, dim=1)
    return output


def rein_forward_mc_dropout(model, inputs, num_samples=10):
    outputs = []
    model.train()  # MC Dropout

    for i in range(num_samples):
        with torch.no_grad():
            output = model.forward_features(inputs)[:, 0, :]
            output = model.linear(output)
            output = torch.softmax(output, dim=1)
            outputs.append(output)
            # print(f"Sample {i+1} output mean: {output.mean().item()}")
    # print(torch.stack(outputs).shape)

    output = torch.mean(torch.stack(outputs), dim=0)
    # print(f"MC Dropout 평균화 후 output shape: {output.shape}")
    return output


def resnet_forward(model, inputs):
    output = model(inputs)
    output = torch.softmax(output, dim=1)
    return output



def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='cifar10')
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--type', '-t', default= 'rein', type=str)
    args = parser.parse_args()

    config = read_conf('conf/data/'+args.data+'.yaml')

    device = 'cuda:'+args.gpu
    save_path = os.path.join(config['save_path'], args.save_path)
    data_path = config['data_root']
    # batch_size = int(config['batch_size'])
    batch_size = 32


    if not os.path.exists(save_path):
        os.mkdir(save_path)


    if args.data == 'cifar10':
        test_loader = cifar10.get_test_loader(batch_size, shuffle=True, num_workers=4, pin_memory=True, get_val_temp=0, data_dir=data_path)
    elif args.data == 'cifar100':
        test_loader = cifar100.get_test_loader(data_dir=data_path, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    elif args.data == 'ham10000':
        train_loader, valid_loader, test_loader = ham10000.get_dataloaders(data_path, batch_size=batch_size, num_workers=4)
    # elif args.data == 'bloodmnist':
    #     train_loader, test_loader, valid_loader = bloodmnist.get_dataloader(batch_size, download=True, num_workers=4)
    # elif args.data == 'pathmnist':
    #     train_loader, test_loader, valid_loader = pathmnist.get_dataloader(batch_size, download=True, num_workers=4)
    # elif args.data == 'retinamnist':
    #     train_loader, test_loader, valid_loader = retinamnist.get_dataloader(batch_size, download=True, num_workers=4)
        
        
    if args.netsize == 's':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant


    dino = torch.hub.load('facebookresearch/dinov2', model_load)
    dino_state_dict = dino.state_dict()

    if args.type == 'rein':
        model = rein.ReinsDinoVisionTransformer(
            **variant
        )
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.load_state_dict(dino_state_dict, strict=False)
        model.to(device)
    elif args.type == 'rein_dropout':
        model = rein.ReinsDinoVisionTransformer_Dropout(
            **variant,
            dropout_rate=0.5
        )
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.load_state_dict(dino_state_dict, strict=False)
        model.to(device)
    elif args.type == 'lora':
        new_state_dict = dict()
        for k in dino_state_dict.keys():
            new_k = k.replace("attn.qkv", "attn.qkv.qkv")
            new_state_dict[new_k] = dino_state_dict[k]
        model = rein.LoRADinoVisionTransformer(dino)
        model.dino.load_state_dict(new_state_dict, strict=False)
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        model.to(device)


    # state_dict = torch.load(os.path.join(save_path, 'last.pth.tar'), map_location='cpu')['state_dict']
    state_dict = torch.load(os.path.join(save_path, 'cyclic_checkpoint_epoch129.pth'), map_location='cpu')
    # state_dict = torch.load(os.path.join(save_path, 'checkpoint_epoch_70.pth'), map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    
    if args.type == 'rein_dropout':
        model.train() # MC Dropout
    else:
        model.eval()        
            
    # print(model)

    ## validation 
    if args.type == 'lora':
        test_accuracy = validation_accuracy_lora(model, test_loader, device)
    else:
        test_accuracy = validation_accuracy(model, test_loader, device, mode=args.type)
    print('test acc:', test_accuracy)

    outputs = []
    targets = []
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(test_loader):
            # print(f"Batch {batch_idx} targets:", target)
            inputs, target = inputs.to(device), target.to(device)
            if args.type == 'rein':
                output = rein_forward(model, inputs)
                # print(output.shape)  
            elif args.type == 'rein_dropout':
                output = rein_forward_mc_dropout(model, inputs, num_samples=10)
                # print(output.shape)
            elif args.type == 'resnet':
                output = resnet_forward(model, inputs)
            elif args.type == 'lora':
                with autocast(enabled=True):
                    output = lora_forward(model, inputs)
                    # print(output.shape)
                
            outputs.append(output.cpu())
            targets.append(target.cpu())
    outputs = torch.cat(outputs).numpy()
    targets = torch.cat(targets).numpy()
    targets = targets.astype(int)
    evaluate(outputs, targets, verbose=True)



if __name__ =='__main__':
    train()