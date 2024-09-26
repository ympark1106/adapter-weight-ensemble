import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

import os
import torch
import torch.nn as nn

import argparse
import timm
import numpy as np
from utils import read_conf, validation_accuracy

import random
import rein

import dino_variant
import evaluation
from data import cifar10, cub, ham10000


def rein_forward(model, inputs):
    output = model.forward_features(inputs)[:, 0, :]
    output = model.linear(output)
    output = torch.softmax(output, dim=1)

    return output


def ensemble_forward(model1, model2, model3, inputs):
    ensemble_output = 0

    
    output1 = model1.forward_features(inputs)[:, 0, :]
    output1 = model1.linear(output1)
    output1 = torch.softmax(output1, dim=1)
    ensemble_output += output1

    output2 = model2.forward_features(inputs)[:, 0, :]
    output2 = model2.linear(output2)
    output2 = torch.softmax(output2, dim=1)
    ensemble_output += output2
    
    output3 = model3.forward_features(inputs)[:, 0, :]
    output3 = model3.linear(output3)
    output3 = torch.softmax(output3, dim=1)
    ensemble_output += output3

    # Average the outputs
    ensemble_output /= 3
    return ensemble_output


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='cub')
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    # parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--type', '-t', default= 'rein', type=str)
    args = parser.parse_args()

    save_path1 = 'adapter1'
    save_path2 = 'adapter2'
    save_path3 = 'adapter3'

    config = read_conf('conf/data/'+args.data+'.yaml')
    
    # device = 'cuda:'+args.gpu
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')  # Set device
    data_path = config['data_root']
    batch_size = int(config['batch_size'])
    
    save_path1 = os.path.join(config['save_path'], save_path1)
    save_path2 = os.path.join(config['save_path'], save_path2)
    save_path3 = os.path.join(config['save_path'], save_path3)

    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)


    if args.data == 'cifar10':
        test_loader = cifar10.get_train_valid_loader(batch_size, augment=True, random_seed=42, valid_size=0.1, shuffle=True, num_workers=4, pin_memory=True, get_val_temp=0, data_dir=data_path)
    elif args.data == 'cub':
        test_loader = cub.get_test_loader(data_path, batch_size=32, scale_size=256, crop_size=224, num_workers=4, pin_memory=True)
    elif args.data == 'ham10000':
        # Test DataLoader
        test_loader, _ = ham10000.create_dataloader(
            annotations_file=os.path.join(data_path, 'ISIC2018_Task3_Test_GroundTruth.csv'),
            img_dir=os.path.join(data_path, 'test/'),
            batch_size=batch_size,
            shuffle=True,
            transform_mode='base'
        )


        
    if args.netsize == 's':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant


    model1 = torch.hub.load('facebookresearch/dinov2', model_load)
    dino_state_dict = model1.state_dict()
    model2 = torch.hub.load('facebookresearch/dinov2', model_load)
    dino_state_dict = model2.state_dict()
    model3 = torch.hub.load('facebookresearch/dinov2', model_load) 
    dino_state_dict = model3.state_dict()
    
    models = [model1, model2, model3]
    
    model1 = rein.ReinsDinoVisionTransformer(
        **variant
    )
    model2 = rein.ReinsDinoVisionTransformer(
        **variant
    )
    model3 = rein.ReinsDinoVisionTransformer(
        **variant
    )
    
    model1.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model2.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model3.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    
    state_dict1 = torch.load(os.path.join(save_path1, 'last.pth.tar'), map_location='cpu')['state_dict']
    state_dict2 = torch.load(os.path.join(save_path2, 'last.pth.tar'), map_location='cpu')['state_dict']
    state_dict3 = torch.load(os.path.join(save_path3, 'last.pth.tar'), map_location='cpu')['state_dict']
    
    
    model1.load_state_dict(state_dict1, strict=False)
    model2.load_state_dict(state_dict2, strict=False)
    model3.load_state_dict(state_dict3, strict=False)
    
    model1.to(device)
    model2.to(device)
    model3.to(device)
    
    
    test_accuracy_list = []


    ## validation
    model1.eval()
    model2.eval()
    model3.eval()
    
    test_accuracy1 = validation_accuracy(model1, test_loader, device, mode=args.type)
    test_accuracy_list.append(test_accuracy1)
    test_accuracy2 = validation_accuracy(model2, test_loader, device, mode=args.type)
    test_accuracy_list.append(test_accuracy2)
    test_accuracy3 = validation_accuracy(model3, test_loader, device, mode=args.type)
    test_accuracy_list.append(test_accuracy3)

    print('test acc:', test_accuracy_list)

    outputs = []
    targets = []
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(test_loader):
            inputs, target = inputs.to(device), target.to(device)
            if args.type == 'rein':
                output = ensemble_forward(model1, model2, model3, inputs)
            outputs.append(output.cpu())
            targets.append(target.cpu())
    outputs = torch.cat(outputs).numpy()
    targets = torch.cat(targets).numpy()
    targets = targets.astype(int)
    evaluation.evaluate(outputs, targets, verbose=True)



if __name__ =='__main__':
    train()