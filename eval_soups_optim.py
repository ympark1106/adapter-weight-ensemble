import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

import os
import torch
import torch.nn as nn
import numpy as np
import argparse

from utils import read_conf, validation_accuracy, evaluate, calculate_ece
import dino_variant
from data import cifar10, cifar100, cub, ham10000, bloodmnist, pathmnist
import rein
from losses import DECE

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
        model.eval()  # 모델은 기본적으로 CPU에 유지
        models.append(model)
    return models


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
    else:
        raise ValueError(f"Unsupported data type: {args.data}")
    
    return test_loader, valid_loader


# DECE-based optimization for model weights
def dece_soup_ensemble(models, model_names, valid_loader, device, variant, config):
    num_models = len(models)
    weights = torch.ones(num_models, device=device) / num_models  # Uniform initialization
    weights.requires_grad = True  # Enable gradient computation for weights
    
    dece_loss = DECE(device=device, num_bins=10, t_a=1.0, t_b=1.0)  # DECE loss initialization
    optimizer = torch.optim.Adam([weights], lr=0.01)  # Optimizer for weights
    
    print("Starting DECE-based weight optimization...")
    
    for epoch in range(5):  # Set the number of epochs as needed
        print(f"Epoch {epoch+1}")
        outputs_list = []
        targets_list = []
        
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Compute weighted ensemble outputs
            weighted_outputs = 0
            for i, model in enumerate(models):
                model.to(device)  # GPU로 이동
                model_output = rein_forward(model, inputs)
                model.to("cpu")  # 계산 후 다시 CPU로 이동
                weighted_outputs += weights[i] * model_output
            
            outputs_list.append(weighted_outputs.cpu())
            targets_list.append(targets.cpu())
        
        # Concatenate outputs and targets
        outputs = torch.cat(outputs_list).to(device)
        targets = torch.cat(targets_list).to(device)
        
        # Compute DECE loss
        dece_value = dece_loss(outputs, targets)
        print(f"Epoch {epoch+1}, DECE Loss: {dece_value.item()}")
        
        # Optimize weights
        optimizer.zero_grad()
        dece_value.backward()
        optimizer.step()
        
        # Normalize weights to ensure they sum to 1
        with torch.no_grad():
            weights /= weights.sum()
    
    # Print final weights
    print("Final model weights:", weights)
    
    # Create final weighted ensemble parameters
    final_params = {k: torch.zeros_like(v) for k, v in models[0].state_dict().items()}
    for i, model in enumerate(models):
        for k in final_params.keys():
            final_params[k] += weights[i] * model.state_dict()[k]
    
    # Load final parameters into a model
    final_model = rein.ReinsDinoVisionTransformer(**variant)
    final_model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    final_model.load_state_dict(final_params)
    final_model.to(device)
    final_model.eval()
    
    return final_model

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='cub')
    parser.add_argument('--gpu', '-g', default='0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    args = parser.parse_args()

    config = read_conf(os.path.join('conf', 'data', f'{args.data}.yaml'))
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    data_path = config['data_root']
    batch_size = int(config['batch_size'])
    
    save_paths = [
        os.path.join(config['save_path'], 'reins_focal_1'),
        os.path.join(config['save_path'], 'reins_focal_2'),
        os.path.join(config['save_path'], 'reins_focal_3'),
        os.path.join(config['save_path'], 'reins_focal_4'),
        os.path.join(config['save_path'], 'reins_focal_5')
    ]
    
    model_names = [os.path.basename(path) for path in save_paths]
    variant = dino_variant._small_variant
    models = initialize_models(save_paths, variant, config, device)
    
    # Load data loaders
    test_loader, valid_loader = setup_data_loaders(args, data_path, batch_size)
    
    # Perform DECE-based soup ensemble
    final_model = dece_soup_ensemble(models, model_names, valid_loader, device, variant, config)
    
    # Evaluate the final model on the test set
    test_accuracy = validation_accuracy(final_model, test_loader, device, mode="rein")
    print("Test accuracy:", test_accuracy)
    
    outputs, targets = [], []
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs, target = inputs.to(device), target.to(device)
            output = rein_forward(final_model, inputs)
            outputs.append(output.cpu())
            targets.append(target.cpu())
    
    outputs = torch.cat(outputs).numpy()
    targets = torch.cat(targets).numpy().astype(int)
    evaluate(outputs, targets, verbose=True)

if __name__ == '__main__':
    train()
