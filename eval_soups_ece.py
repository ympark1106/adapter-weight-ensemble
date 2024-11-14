import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

import os
import torch
import torch.nn as nn
import argparse
import numpy as np

from utils import read_conf, validation_accuracy, ModelWithTemperature, validate, evaluate, calculate_ece, calculate_nll
import dino_variant
from data import cifar10, cifar100, cub, ham10000, bloodmnist, pathmnist
import rein

# Model forward function
def rein_forward(model, inputs):
    output = model.forward_features(inputs)[:, 0, :]
    output = model.linear(output)
    output = torch.softmax(output, dim=1)
    return output

# Model initialization
def initialize_models(save_paths, variant, config, device):
    models = []
    for save_path in save_paths:
        model = rein.ReinsDinoVisionTransformer(**variant)
        model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        state_dict = torch.load(os.path.join(save_path, 'last.pth.tar'), map_location='cpu')['state_dict']
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        models.append(model)
    return models

def get_model_from_sd(state_dict, variant, config, device):
    model = rein.ReinsDinoVisionTransformer(**variant)
    model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    
    return model
            
# Data loader setup
def setup_data_loaders(args, data_path, batch_size):
    if args.data == 'cifar10':
        test_loader = cifar10.get_test_loader(batch_size, shuffle=True, num_workers=4, pin_memory=True, get_val_temp=0, data_dir=data_path)
        valid_loader = None
    elif args.data == 'cifar100':
        test_loader = cifar100.get_test_loader(data_dir=data_path, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        _, valid_loader = cifar100.get_train_valid_loader(data_dir=data_path, augment=True, batch_size=32, valid_size=0.1, random_seed=42, shuffle=True, num_workers=4, pin_memory=True)
    elif args.data == 'cub':
        test_loader = cub.get_test_loader(data_path, batch_size=32, scale_size=256, crop_size=224, num_workers=4, pin_memory=True)
        _, valid_loader = cub.get_train_val_loader(data_path, batch_size=32, scale_size=256, crop_size=224, num_workers=4, pin_memory=True)
    elif args.data == 'ham10000':
        _, valid_loader, test_loader = ham10000.get_dataloaders(data_path, batch_size=32, num_workers=4)
    elif args.data == 'bloodmnist':
        _, valid_loader, test_loader = bloodmnist.get_dataloader(batch_size=32, download=True, num_workers=4)
    elif args.data == 'pathmnist':
        _, valid_loader, test_loader = pathmnist.get_dataloader(batch_size=32, download=True, num_workers=4)
    else:
        raise ValueError(f"Unsupported data type: {args.data}")
    
    return test_loader, valid_loader

# Greedy soup model ensembling
def greedy_soup_ensemble(models, model_names, valid_loader, device, variant, config):
    # Calculate ECE for each model and sort them by ECE in ascending order (lower ECE is better)
    
    ece_list = [validate(model, valid_loader, device) for model in models]
    n = len(ece_list)
    model_ece_pairs = [(model, ece, name) for model, ece, name in zip(models, ece_list, model_names)]
    
    # Sort models based on ECE
    sorted_models = sorted(model_ece_pairs, key=lambda x: x[1])
    
    # Print each model's name and ECE
    print("Sorted models with ECE performance:")
    for model, ece, name in sorted_models:  
        print(f'Model: {name}, ECE: {ece}')

    best_ece = sorted_models[0][1]
    greedy_soup_params = sorted_models[0][0].state_dict()
    greedy_soup_ingredients = [sorted_models[0][0]]
    
    TOLERANCE = (sorted_models[n-1][1] - sorted_models[0][1])/2 # Acceptable tolerance for ECE
    TOLERANCE = -0.01
    print(f'Tolerance: {TOLERANCE}')

    for i in range(1, len(models)):
        new_ingredient_params = sorted_models[i][0].state_dict()
        num_ingredients = len(greedy_soup_ingredients)
        print(f'Adding ingredient {i+1} ({sorted_models[i][2]}) to the greedy soup. Num ingredients: {num_ingredients}')
    
        
        # Calculate potential new parameters with the new ingredient
        potential_greedy_soup_params = {
                k : greedy_soup_params[k].clone() * (num_ingredients / (num_ingredients + 1.)) + new_ingredient_params[k].clone() * (1. / (num_ingredients + 1))
                for k in new_ingredient_params
        }

        
        # Temporarily load potential parameters to test validation ECE
        temp_model = get_model_from_sd(potential_greedy_soup_params, variant, config, device)
        temp_model.eval()
        
        outputs, targets = [], []
        with torch.no_grad():
            for inputs, target in valid_loader:
                inputs, target = inputs.to(device), target.to(device)
                output = rein_forward(temp_model, inputs)
                outputs.append(output.cpu())
                targets.append(target.cpu())
        
        outputs = torch.cat(outputs).numpy()
        targets = torch.cat(targets).numpy().astype(int)
        held_out_val_ece = calculate_ece(outputs, targets)
        
        print(f'Potential greedy soup ECE: {held_out_val_ece}, best ECE so far: {best_ece}.')
        print(f'Current greedy_soup_params checksum: {torch.sum(torch.stack([v.sum() for v in greedy_soup_params.values()]))}')
        print(f'Potential greedy_soup_params checksum: {torch.sum(torch.stack([v.sum() for v in potential_greedy_soup_params.values()]))}')

        # Add new ingredient to the greedy soup if it improves ECE or is within tolerance
        if held_out_val_ece < best_ece + TOLERANCE:
            best_ece = held_out_val_ece
            greedy_soup_ingredients.append(sorted_models[i][0])
            greedy_soup_params = potential_greedy_soup_params
            print(f'<Added new ingredient to soup. Total ingredients: {len(greedy_soup_ingredients)}>\n')
        else:
            print(f'<No improvement. Reverting to best-known parameters.>\n')
            
    final_model = get_model_from_sd(greedy_soup_params, variant, config, device)
        
    return greedy_soup_params, final_model

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
    
    save_paths = [
        # os.path.join(config['save_path'], 'reins_ce1'),
        # os.path.join(config['save_path'], 'reins_ce2'),
        # os.path.join(config['save_path'], 'reins_ce3'),
        # os.path.join(config['save_path'], 'reins_ce4'),
        
        os.path.join(config['save_path'], 'reins_focal_1'),
        os.path.join(config['save_path'], 'reins_focal_2'),
        os.path.join(config['save_path'], 'reins_focal_3'),
        os.path.join(config['save_path'], 'reins_focal_4'),
        os.path.join(config['save_path'], 'reins_focal_5'),
        # os.path.join(config['save_path'], 'reins_focal_lr_1'),
        # os.path.join(config['save_path'], 'reins_focal_lr_2'),
        # os.path.join(config['save_path'], 'reins_focal_lr_3'),
        # os.path.join(config['save_path'], 'reins_focal_lr_4'),
        # os.path.join(config['save_path'], 'reins_focal_lr_5'),
        
        # os.path.join(config['save_path'], 'reins_adafocal1'),
        # os.path.join(config['save_path'], 'reins_adafocal2'),
        # os.path.join(config['save_path'], 'reins_adafocal3'),
        # os.path.join(config['save_path'], 'reins_adafocal4'),
    ]
    
    model_names = [os.path.basename(path) for path in save_paths]

    variant = dino_variant._small_variant
    models = initialize_models(save_paths, variant, config, device)
    test_loader, valid_loader = setup_data_loaders(args, data_path, batch_size)
    
    # Step 1: Compute greedy soup parameters
    greedy_soup_params, model = greedy_soup_ensemble(models, model_names, valid_loader, device, variant, config)

    # Evaluate the final model on the test set
    # model1.load_state_dict(greedy_soup_params)
    model = get_model_from_sd(greedy_soup_params, variant, config, device)
    model.eval()
    test_accuracy = validation_accuracy(model, test_loader, device, mode=args.type)
    print('Test accuracy:', test_accuracy)

    outputs, targets = [], []
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs, target = inputs.to(device), target.to(device)
            output = rein_forward(model, inputs)
            outputs.append(output.cpu())
            targets.append(target.cpu())
    
    outputs = torch.cat(outputs).numpy()
    targets = torch.cat(targets).numpy().astype(int)
    evaluate(outputs, targets, verbose=True)

if __name__ == '__main__':
    train()
