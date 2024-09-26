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
from utils import read_conf, validation_accuracy

import random
import rein
import operator

import dino_variant
import evaluation
from data import cifar10, cub, ham10000


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

    save_path1 = 'adapter1'
    save_path2 = 'adapter2'
    save_path3 = 'adapter3'

    config = read_conf(os.path.join('conf', 'data', f'{args.data}.yaml'))

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    data_path = config['data_root']
    batch_size = int(config['batch_size'])
    
    save_path1 = os.path.join(config['save_path'], 'adapter1')
    save_path2 = os.path.join(config['save_path'], 'adapter11')
    save_path3 = os.path.join(config['save_path'], 'adapter111')

    if args.data == 'cifar10':
        test_loader = cifar10.get_train_valid_loader(
            batch_size, augment=True, random_seed=42, valid_size=0.1, shuffle=True, num_workers=4, pin_memory=True, get_val_temp=0, data_dir=data_path)
    elif args.data == 'cub':
        test_loader = cub.get_test_loader(
            data_path, batch_size=32, scale_size=256, crop_size=224, num_workers=4, pin_memory=True)
    elif args.data == 'ham10000':
        test_loader, _ = ham10000.create_dataloader(
            annotations_file=os.path.join(data_path, 'ISIC2018_Task3_Test_GroundTruth.csv'),
            img_dir=os.path.join(data_path, 'test'),
            batch_size=batch_size,
            shuffle=True,
            transform_mode='base'
        )

    if args.netsize == 's':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant

    model1 = torch.hub.load('facebookresearch/dinov2', model_load)
    model2 = torch.hub.load('facebookresearch/dinov2', model_load)
    model3 = torch.hub.load('facebookresearch/dinov2', model_load)
    
    models = [model1, model2, model3]
    
    model1 = rein.ReinsDinoVisionTransformer(**variant)
    model2 = rein.ReinsDinoVisionTransformer(**variant)
    model3 = rein.ReinsDinoVisionTransformer(**variant)
    
    model1.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model2.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model3.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    
    state_dict1 = torch.load(os.path.join(save_path1, 'last.pth.tar'), map_location='cpu')['state_dict']
    state_dict2 = torch.load(os.path.join(save_path2, 'last.pth.tar'), map_location='cpu')['state_dict']
    state_dict3 = torch.load(os.path.join(save_path3, 'last.pth.tar'), map_location='cpu')['state_dict']
    
    model_path1 = os.path.join(save_path1, 'last.pth.tar')
    model_path2 = os.path.join(save_path2, 'last.pth.tar')
    model_path3 = os.path.join(save_path3, 'last.pth.tar')
    
    model_paths = [model_path1, model_path2, model_path3]
        
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


    model_dict = {
        "model1": model1,
        "model2": model2,
        "model3": model3,
        'ece1': 0.078,
        'ece2': 0.079,
        'ece3': 0.079,
    }

    # ## validation
    model1.eval()
    model2.eval()
    model3.eval()

    model_dict["test_accuracy1"] = validation_accuracy(model1, test_loader, device, mode=args.type)
    model_dict["test_accuracy2"] = validation_accuracy(model2, test_loader, device, mode=args.type)
    model_dict["test_accuracy3"] = validation_accuracy(model3, test_loader, device, mode=args.type)
    



    #Uniform Soup
    for j, model_path in enumerate(model_paths):

        print(f'Adding model {j+1} of {len(models)} to uniform soup.')
        assert os.path.exists(model_path)
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['state_dict']
        # for k, v in state_dict.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"Averaging layer: {k}")
        if j == 0:
            uniform_soup = {k: v * (1./len(models)) for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
        else:
            uniform_soup = {k: v * (1./len(models)) + uniform_soup[k] for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
            
            
    model1.load_state_dict(uniform_soup)
    model1.eval()
    
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
            
            
            
    # Greedy Soup     
    sorted_models = sorted(
        [(model_dict["model1"], model_dict["ece1"]),
        (model_dict["model2"], model_dict["ece2"]),
        (model_dict["model3"], model_dict["ece3"])],
        key=lambda x: x[1],  
        reverse=False 
    )
    
    best_ece = sorted_models[0][1]

    for idx, (model, ece) in enumerate(sorted_models, 1):
        print(f"Model {idx} has accuracy {ece:.4f}")

    greedy_soup_params = sorted_models[0][0].state_dict() 
    greedy_soup_ingredients = [sorted_models[0][0]]

    for i in range(1, len(models)):
        print(f'Testing model {i} of {len(models)}')

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
            for batch_idx, (inputs, target) in enumerate(test_loader):
                inputs, target = inputs.to(device), target.to(device)
                output = rein_forward(model1, inputs)
                outputs.append(output.cpu())
                targets.append(target.cpu())
        outputs = torch.cat(outputs).numpy()
        targets = torch.cat(targets).numpy()
        targets = targets.astype(int)
        held_out_val_ece = evaluation.calculate_ece(outputs, targets, verbose=False)
    
        # If accuracy on the held-out val set increases, add the new model to the greedy soup.
        print(f'Potential greedy soup val ece {held_out_val_ece}, best so far {best_ece}.')
        if held_out_val_ece < best_ece:
            greedy_soup_ingredients.append(sorted_models[i])
            best_ece = held_out_val_ece
            greedy_soup_params = potential_greedy_soup_params
            # print(f'Adding to soup. New soup is {greedy_soup_ingredients}')
            
    
    model1.load_state_dict(greedy_soup_params)
    model1.eval()
    
    
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
