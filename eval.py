import os
import torch
import torch.nn as nn

import argparse
import timm
import numpy as np
import utils

import random
import rein

import dino_variant
import evaluation

def rein_forward(model, inputs):
    output = model.forward_features(inputs)[:, 0, :]
    output = model.linear(output)
    output = torch.softmax(output, dim=1)

    return output

def rein3_forward(model, inputs):
    f = model.forward_features1(inputs)
    f = f[:, 0, :]
    outputs1 = model.linear(f)

    f = model.forward_features2(inputs)
    f = f[:, 0, :]
    outputs2 = model.linear(f)

    f = model.forward_features3(inputs)
    f = f[:, 0, :]
    outputs3 = model.linear(f)

    outputs1 = torch.softmax(outputs1, dim=1)
    outputs2 = torch.softmax(outputs2, dim=1)
    outputs3 = torch.softmax(outputs3, dim=1)

    return (outputs1 + outputs2 + outputs3)/3


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--type', '-t', default= 'rein', type=str)
    args = parser.parse_args()



    config = utils.read_conf('conf/'+args.data+'.json')
    device = 'cuda:'+args.gpu
    save_path = os.path.join(config['save_path'], args.save_path)
    data_path = config['id_dataset']

    if not os.path.exists(save_path):
        os.mkdir(save_path)


    train_loader, valid_loader = utils.get_dataset(data_path)
        
    if args.netsize == 's':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant


    model = torch.hub.load('facebookresearch/dinov2', model_load)
    dino_state_dict = model.state_dict()

    if args.type == 'rein':
        model = rein.ReinsDinoVisionTransformer(
            **variant
        )
    if args.type == 'rein3':
        model = rein.ReinsDinoVisionTransformer_3_head(
            **variant,
            token_lengths = [33, 33, 33]
        )
    model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.load_state_dict(dino_state_dict, strict=False)
    model.to(device)

    state_dict = torch.load(os.path.join(save_path, 'last.pth.tar'))['state_dict']
    model.load_state_dict(state_dict, strict=True)
            
    #print(model)
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    ## validation
    model.eval()
    valid_accuracy = utils.validation_accuracy(model, valid_loader, device, mode=args.type)
    print(valid_accuracy)

    outputs = []
    targets = []
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(valid_loader):
            inputs, target = inputs.to(device), target.to(device)
            if args.type == 'rein':
                output = rein_forward(model, inputs)
            elif args.type == 'rein3':
                output = rein3_forward(model, inputs)
            outputs.append(output)
            targets.append(target)
    outputs = torch.cat(outputs)
    targets = torch.cat(targets)
    evaluation.evaluate(outputs.cpu().numpy(), targets.cpu().numpy(), verbose=True)



if __name__ =='__main__':
    train()