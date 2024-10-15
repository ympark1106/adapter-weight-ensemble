import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

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
from valid import validate

import dino_variant
import evaluation
from data import cifar10, cifar100, cub, ham10000


def rein_forward(model, inputs):
    output = model.forward_features(inputs)[:, 0, :]
    output = model.linear(output)
    output = torch.softmax(output, dim=1)

    return output


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='cub')
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--type', '-t', default= 'rein', type=str)
    args = parser.parse_args()


    config = read_conf(os.path.join('conf', 'data', f'{args.data}.yaml'))

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    data_path = config['data_root']
    batch_size = int(config['batch_size'])
    
    save_path1 = os.path.join(config['save_path'], 'adapter2222')
    save_path2 = os.path.join(config['save_path'], 'adapter22')
    save_path3 = os.path.join(config['save_path'], 'adapter222')
    # save_path4 = os.path.join(config['save_path'], 'adapter2222')

    if args.data == 'cifar10':
        test_loader = cifar10.get_test_loader(batch_size, shuffle=True, num_workers=4, pin_memory=True, get_val_temp=0, data_dir=data_path)
    elif args.data == 'cifar100':
        test_loader = cifar100.get_test_loader(data_dir=data_path, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        _, valid_loader = cifar100.get_train_valid_loader(data_dir=data_path, augment=True, batch_size=32, valid_size=0.1, random_seed=42, shuffle=True, num_workers=4, pin_memory=True)
    elif args.data == 'cub':
        test_loader = cub.get_test_loader(data_path, batch_size=32, scale_size=256, crop_size=224, num_workers=4, pin_memory=True)
        _,valid_loader = cub.get_train_val_loader(data_path, batch_size=32, scale_size=256, crop_size=224, num_workers=4, pin_memory=True)
    elif args.data == 'ham10000':
        train_loader, valid_loader, test_loader = ham10000.get_dataloaders(data_path, batch_size=32, num_workers=4)
        
        
    if args.netsize == 's':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant

    model1 = torch.hub.load('facebookresearch/dinov2', model_load)
    model2 = torch.hub.load('facebookresearch/dinov2', model_load)
    model3 = torch.hub.load('facebookresearch/dinov2', model_load)
    # model4 = torch.hub.load('facebookresearch/dinov2', model_load)
    
    models = [model1, model2, model3, 
            #   model4
              ]
    
    model1 = rein.ReinsDinoVisionTransformer(**variant)
    model2 = rein.ReinsDinoVisionTransformer(**variant)
    model3 = rein.ReinsDinoVisionTransformer(**variant)
    # model4 = rein.ReinsDinoVisionTransformer(**variant)
    
    model1.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model2.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model3.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    # model4.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    
    state_dict1 = torch.load(os.path.join(save_path1, 'last.pth.tar'), map_location='cpu')['state_dict']
    state_dict2 = torch.load(os.path.join(save_path2, 'last.pth.tar'), map_location='cpu')['state_dict']
    state_dict3 = torch.load(os.path.join(save_path3, 'last.pth.tar'), map_location='cpu')['state_dict']
    # state_dict4 = torch.load(os.path.join(save_path4, 'last.pth.tar'), map_location='cpu')['state_dict']
    
    model_path1 = os.path.join(save_path1, 'last.pth.tar')
    model_path2 = os.path.join(save_path2, 'last.pth.tar')
    model_path3 = os.path.join(save_path3, 'last.pth.tar')
    # model_path4 = os.path.join(save_path4, 'last.pth.tar')
    
    model_paths = [model_path1, model_path2, model_path3, 
                #    model_path4
                   ]
        
    model1.load_state_dict(state_dict1, strict=False)
    model2.load_state_dict(state_dict2, strict=False)
    model3.load_state_dict(state_dict3, strict=False)
    # model4.load_state_dict(state_dict4, strict=False)
    
    
    model1.to(device)
    model2.to(device)
    model3.to(device)
    # model4.to(device)
    

    ## validation
    model1.eval()
    model2.eval()
    model3.eval()
    # model4.eval()


    model_dict = {
        "model1": model1,
        "model2": model2,
        "model3": model3,
        # "model4": model4,
        'ece1': None,
        'ece2': None,
        'ece3': None,
        # 'ece4': 0.062
    }
    
    


    model_dict["ece1"] = validate(model_dict["model1"], valid_loader, device, evaluation)
    model_dict["ece2"] = validate(model_dict["model2"], valid_loader, device, evaluation)
    model_dict["ece3"] = validate(model_dict["model3"], valid_loader, device, evaluation)
    
    

            
    # Greedy Soup     
    sorted_models = sorted(
        [(model_dict["model1"], model_dict["ece1"]),
        (model_dict["model2"], model_dict["ece2"]),
        (model_dict["model3"], model_dict["ece3"]),
        # (model_dict["model4"], model_dict["ece4"])
        ],
        
        key=lambda x: x[1],  
        reverse=False 
    )
    
    best_ece = sorted_models[0][1]

    for idx, (model, ece) in enumerate(sorted_models, 1):
        print(f"Model {idx} has ece {ece:.4f}")

    greedy_soup_params = sorted_models[0][0].state_dict() 
    greedy_soup_ingredients = [sorted_models[0][0]]

    for i in range(1, len(models)):
        print(f'Testing model {i+1} of {len(models)}')

        new_ingredient_params = sorted_models[i][0].state_dict()
        num_ingredients = len(greedy_soup_ingredients)
        print(f'Num ingredients: {num_ingredients}')
        
        potential_greedy_soup_params = {
            k : greedy_soup_params[k].clone() * (num_ingredients / (num_ingredients + 1.)) + 
                new_ingredient_params[k].clone() * (1. / (num_ingredients + 1))
            for k in new_ingredient_params
        }
    
        sorted_models[0][0].load_state_dict(potential_greedy_soup_params)
        sorted_models[0][0].eval()
        
        outputs = []
        targets = []
        with torch.no_grad():
            for batch_idx, (inputs, target) in enumerate(valid_loader):
                inputs, target = inputs.to(device), target.to(device)
                output = rein_forward(model1, inputs)
                outputs.append(output.cpu())
                targets.append(target.cpu())
        outputs = torch.cat(outputs).numpy()
        targets = torch.cat(targets).numpy()
        targets = targets.astype(int)
        held_out_val_ece = evaluation.calculate_ece(outputs, targets)
    
        # If accuracy on the held-out val set increases, add the new model to the greedy soup.
        print(f'Potential greedy soup val ece {held_out_val_ece}, best so far {best_ece}.')
        if held_out_val_ece < best_ece:
            greedy_soup_ingredients.append(sorted_models[i])
            best_ece = held_out_val_ece
            greedy_soup_params = potential_greedy_soup_params
            # print(f'Adding to soup. New soup is {greedy_soup_ingredients}')
            
    
    model1.load_state_dict(greedy_soup_params)
    
    model_with_temp1 = ModelWithTemperature(model1, device=device)
    model_with_temp1.set_temperature(valid_loader)  # Apply temperature scaling
    print(f"Temperature for model: {model_with_temp1.get_temperature():.4f}")
    
    
    test_accuracy = validation_accuracy(model1, test_loader, device, mode=args.type)
    print('test acc:', test_accuracy)

    outputs = []
    targets = []
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(test_loader):
            inputs, target = inputs.to(device), target.to(device)
            output = rein_forward(model1, inputs)
            outputs.append(output.cpu())
            targets.append(target.cpu())
    outputs = torch.cat(outputs).numpy()
    targets = torch.cat(targets).numpy()
    targets = targets.astype(int)
    evaluation.evaluate(outputs, targets, verbose=True)


if __name__ =='__main__':
    train()
