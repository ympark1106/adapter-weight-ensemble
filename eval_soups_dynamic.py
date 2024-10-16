import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

import os
import torch
import torch.nn as nn
import argparse
import numpy as np

from utils import read_conf, validation_accuracy, ModelWithTemperature, validate, evaluate, calculate_ece, calculate_nll

import dino_variant
from data import cifar10, cifar100, cub, ham10000
import rein


def rein_forward(model, inputs):
    output = model.forward_features(inputs)[:, 0, :]
    output = model.linear(output)
    output = torch.softmax(output, dim=1)
    return output


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
    else:
        raise ValueError(f"Unsupported data type: {args.data}")
    
    return test_loader, valid_loader


def calculate_ensemble_params(models, weights):
    ensemble_params = {name: torch.zeros_like(param) for name, param in models[0].state_dict().items()}
    for model, weight in zip(models, weights):
        model_params = model.state_dict()
        for name, param in model_params.items():
            ensemble_params[name] += param * weight
    return ensemble_params


def optimize_weights(models, valid_loader, device, num_iterations, alpha, learning_rate):
    # Initial ECE-based weights
    ece_list = [validate(model, valid_loader, device) for model in models]
    weights = [1 / ece for ece in ece_list]
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Convert weights to torch tensor with gradients enabled
    weights_tensor = torch.tensor(weights, requires_grad=True, device=device)

    for iteration in range(num_iterations):
        # Create ensemble output based on current weights
        ensemble_params = calculate_ensemble_params(models, weights_tensor)
        models[0].load_state_dict(ensemble_params)
        models[0].eval()

        # Calculate ECE and NLL of the ensemble
        outputs, targets = [], []
        with torch.no_grad():
            for inputs, target in valid_loader:
                inputs, target = inputs.to(device), target.to(device)
                output = rein_forward(models[0], inputs)
                outputs.append(output.cpu())
                targets.append(target.cpu())

        outputs = torch.cat(outputs).numpy()
        targets = torch.cat(targets).numpy().astype(int)

        # Calculate ECE and NLL as tensors
        ensemble_ece = torch.tensor(calculate_ece(outputs, targets), requires_grad=True, device=device)
        ensemble_nll = torch.tensor(calculate_nll(outputs, targets), requires_grad=True, device=device)

        # Meta loss as a tensor
        meta_loss = ensemble_ece + alpha * ensemble_nll

        # Compute gradients for weights
        meta_loss.backward()

        # Update weights using gradient descent
        with torch.no_grad():
            weights_tensor -= learning_rate * weights_tensor.grad

            # Apply threshold to weights
            threshold = 0.0
            weights_tensor = torch.clamp(weights_tensor, min=threshold)

            # Normalize weights
            total_weight = torch.sum(weights_tensor)
            if total_weight > 0:
                weights_tensor /= total_weight
            else:
                print("All weights are below the threshold. No models included in the soup.")
                return None

            # Reset gradients
            weights_tensor.grad.zero_()

        print(f"Iteration {iteration + 1}/{num_iterations}, Meta Loss: {meta_loss.item()}")

    # Return the final optimized weights
    return weights_tensor.tolist()



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
        os.path.join(config['save_path'], 'reins_ce1'),
        os.path.join(config['save_path'], 'reins_ce2'),
        os.path.join(config['save_path'], 'reins_ce3'),
        os.path.join(config['save_path'], 'reins_ce4'),
    ]

    # Initialize models and data loaders
    variant = dino_variant._small_variant
    models = initialize_models(save_paths, variant, config, device)
    test_loader, valid_loader = setup_data_loaders(args, data_path, batch_size)

    # Optimize weights
    num_iterations = 5
    alpha = 0.5
    learning_rate = 0.01
    optimized_weights = optimize_weights(models, valid_loader, device, num_iterations, alpha, learning_rate)

    if optimized_weights is None:
        return

    # Finalize ensemble with optimized weights
    ensemble_params = calculate_ensemble_params(models, optimized_weights)
    models[0].load_state_dict(ensemble_params)
    models[0].eval()

    # Final evaluation on test set
    test_accuracy = validation_accuracy(models[0], test_loader, device, mode=args.type)
    print('Final Test Accuracy:', test_accuracy)

    outputs, targets = [], []
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs, target = inputs.to(device), target.to(device)
            output = rein_forward(models[0], inputs)
            outputs.append(output.cpu())
            targets.append(target.cpu())

    outputs = torch.cat(outputs).numpy()
    targets = torch.cat(targets).numpy().astype(int)
    evaluate(outputs, targets, verbose=True)


if __name__ == '__main__':
    train()
