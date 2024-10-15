import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

import matplotlib
matplotlib.use('Agg') 

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = '1'

import torch
import torch.nn as nn

import argparse
import timm
import numpy as np
from utils import read_conf, validation_accuracy, ModelWithTemperature
import random
import rein
import operator
import matplotlib.pyplot as plt

import dino_variant
import evaluation
from data import cifar10, cifar100, cub, ham10000
from loss_landscape import plot_loss_landscape_with_models


def rein_forward(model, inputs, temp_scaler=None):
    output = model.forward_features(inputs)[:, 0, :]
    output = model.linear(output)
    if temp_scaler:  # Apply temperature scaling if available
        output = temp_scaler.temperature_scale(output)
    output = torch.softmax(output, dim=1)
    return output

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='cub')
    parser.add_argument('--gpu', '-g', default='0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--type', '-t', default='rein', type=str)
    args = parser.parse_args()

    config = read_conf(os.path.join('conf', 'data', f'{args.data}.yaml'))

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    data_path = config['data_root']
    batch_size = int(config['batch_size'])
    
    save_path1 = os.path.join(config['save_path'], 'adapter2222')
    save_path2 = os.path.join(config['save_path'], 'adapter22')
    save_path3 = os.path.join(config['save_path'], 'adapter222')

    if args.data == 'cifar10':
        test_loader = cifar10.get_train_valid_loader(
            batch_size, augment=True, random_seed=42, valid_size=0.1, shuffle=True, num_workers=4, pin_memory=True, get_val_temp=0, data_dir=data_path)
    elif args.data == 'cifar100':
        _, valid_loader = cifar100.get_train_valid_loader(
            data_dir=data_path, augment=True, batch_size=32, valid_size=0.1, random_seed=42, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = cifar100.get_test_loader(
            data_dir=data_path, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    elif args.data == 'cub':
        _, valid_loader = cub.get_train_val_loader(data_path, batch_size=32, scale_size=256, crop_size=224, num_workers=4, pin_memory=True)
        test_loader = cub.get_test_loader(
            data_path, batch_size=32, scale_size=256, crop_size=224, num_workers=4, pin_memory=True)
    elif args.data == 'ham10000':
        _, valid_loader, test_loader = ham10000.get_dataloaders(data_path, batch_size=32, num_workers=4)

    if args.netsize == 's':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant

    model1 = rein.ReinsDinoVisionTransformer(**variant)
    model2 = rein.ReinsDinoVisionTransformer(**variant)
    model3 = rein.ReinsDinoVisionTransformer(**variant)
    
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
    
    model1.eval()
    model2.eval()
    model3.eval()
    
    # Wrap the models with temperature scaling
    model_with_temp1 = ModelWithTemperature(model1, device=device)
    model_with_temp2 = ModelWithTemperature(model2, device=device)
    model_with_temp3 = ModelWithTemperature(model3, device=device)

    model_with_temp1.set_temperature(valid_loader)
    model_with_temp2.set_temperature(valid_loader)
    model_with_temp3.set_temperature(valid_loader)
    
    models_temp = [model_with_temp1, model_with_temp2, model_with_temp3]
    
    model_dict = {
        "model1": model_with_temp1,
        "model2": model_with_temp2,
        "model3": model_with_temp3,
        "valid_accuracy1": None,
        "valid_accuracy2": None,
        "valid_accuracy3": None
    }

    # Calculate validation accuracy
    model_dict["valid_accuracy1"] = validation_accuracy(model_with_temp1, valid_loader, device, mode=args.type)
    model_dict["valid_accuracy2"] = validation_accuracy(model_with_temp2, valid_loader, device, mode=args.type)
    model_dict["valid_accuracy3"] = validation_accuracy(model_with_temp3, valid_loader, device, mode=args.type)
    
    print(f"Model 1 Valid Accuracy: {model_dict['valid_accuracy1']}")
    print(f"Model 2 Valid Accuracy: {model_dict['valid_accuracy2']}")
    print(f"Model 3 Valid Accuracy: {model_dict['valid_accuracy3']}")

    # Sort models by validation accuracy
    sorted_models = sorted(
        [(model_dict["model1"], model_dict["valid_accuracy1"], model_with_temp1.get_temperature()),
         (model_dict["model2"], model_dict["valid_accuracy2"], model_with_temp2.get_temperature()),
         (model_dict["model3"], model_dict["valid_accuracy3"], model_with_temp3.get_temperature())],
        key=lambda x: x[1],  
        reverse=True  
    )

    for idx, (model, accuracy, temp) in enumerate(sorted_models, 1):
        print(f"Model {idx} has valid accuracy {accuracy:.4f} and temperature {temp:.4f}")
        
    max_accuracy = sorted_models[0][1]
    greedy_soup_params = sorted_models[0][0].state_dict()
    greedy_soup_ingredients = [sorted_models[0][0]]
    
    for i in range(1, len(models_temp)):
        print(f'Testing model {i} of {len(models_temp)}')
        new_ingredient_params = sorted_models[i][0].state_dict()
        num_ingredients = len(greedy_soup_ingredients)
        print(i,sorted_models[i][2])
        temperature_weight = 1.0 / sorted_models[i][2]# Use inverse of temperature as weight
        print(temperature_weight)
        total_weight = sum(1.0 / model[2] for model in sorted_models[:i+1])  # Normalize weights
        print(total_weight)
        
        potential_greedy_soup_params = {
            k: greedy_soup_params[k].clone() * (1 - temperature_weight / total_weight) +
            new_ingredient_params[k].clone() * (temperature_weight / total_weight)
            for k in new_ingredient_params
    }

    
        model.load_state_dict(potential_greedy_soup_params)
        model.eval()
        
        held_out_val_accuracy = validation_accuracy(model, valid_loader, device, mode=args.type)
        
        print(f'Potential greedy soup test acc {held_out_val_accuracy}, best so far {max_accuracy}.')
        if held_out_val_accuracy > max_accuracy:
            greedy_soup_ingredients.append(sorted_models[i])
            max_accuracy = held_out_val_accuracy
            greedy_soup_params = potential_greedy_soup_params
    
    model_with_temp1.load_state_dict(greedy_soup_params)
    model_with_temp1.eval()
    
    test_accuracy = validation_accuracy(model_with_temp1, test_loader, device, mode=args.type)
    print('test acc:', test_accuracy)

    outputs = []
    targets = []
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(test_loader):
            inputs, target = inputs.to(device), target.to(device)
            output = rein_forward(model1, inputs, temp_scaler=model_with_temp3)
            outputs.append(output.cpu())
            targets.append(target.cpu())
    outputs = torch.cat(outputs).numpy()
    targets = torch.cat(targets).numpy()
    targets = targets.astype(int)
    evaluation.evaluate(outputs, targets, verbose=True)
    
if __name__ == '__main__':
    train()
