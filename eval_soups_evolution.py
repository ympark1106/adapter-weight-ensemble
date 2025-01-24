import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

import os
import torch
import torch.nn as nn
import argparse
import numpy as np
import random

from utils import read_conf, validation_accuracy, validate, evaluate, calculate_ece, calculate_nll

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


def evaluate_ensemble(models, weights, valid_loader, device):
    ece_list = [validate(model, valid_loader, device) for model in models]
    sorted_models = sorted([(model, ece) for model, ece in zip(models, ece_list)], key=lambda x: x[1])
    print(f'Sorted models ECE: {sorted_models[0][1]}, {sorted_models[1][1]}, {sorted_models[2][1]}, {sorted_models[3][1]}\n')
    
    ensemble_params = calculate_ensemble_params(sorted_models, weights)
    sorted_models[0].load_state_dict(ensemble_params) 
    sorted_models[0].eval()

    outputs, targets = [], []
    with torch.no_grad():
        for inputs, target in valid_loader:
            inputs, target = inputs.to(device), target.to(device)
            output = rein_forward(models[0], inputs)
            outputs.append(output.cpu())
            targets.append(target.cpu())

    outputs = torch.cat(outputs).numpy()
    targets = torch.cat(targets).numpy().astype(int)

    ece = calculate_ece(outputs, targets)
    nll = calculate_nll(outputs, targets)
    return ece + 0.5 * nll  # Meta loss


import numpy as np

def optimize_weights_evolution_strategy(models, valid_loader, device, pruning_threshold=0.01, population_size=10, generations=20, mutation_rate=0.1):
    # 초기 가중치 조합 무작위 생성 (균등 분포 사용)
    population = [np.random.dirichlet(np.ones(len(models))).tolist() for _ in range(population_size)]
    best_weights, best_loss = None, float('inf')

    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations}")

        fitness_scores = []
        for weights in population:
            # 현재 가중치로 앙상블 파라미터 계산
            ensemble_params = calculate_ensemble_params(models, weights)
            models[0].load_state_dict(ensemble_params)
            models[0].eval()

            # Meta loss 계산
            outputs, targets = [], []
            with torch.no_grad():
                for inputs, target in valid_loader:
                    inputs, target = inputs.to(device), target.to(device)
                    output = rein_forward(models[0], inputs)
                    outputs.append(output.cpu())
                    targets.append(target.cpu())

            outputs = torch.cat(outputs).numpy()
            targets = torch.cat(targets).numpy().astype(int)
            ece = calculate_ece(outputs, targets)
            nll = calculate_nll(outputs, targets)
            meta_loss = ece + 0.5 * nll

            fitness_scores.append((meta_loss, weights))

            # 최적 가중치 업데이트
            if meta_loss < best_loss:
                best_loss, best_weights = meta_loss, weights

        # 상위 절반 선택 및 변이
        fitness_scores.sort(key=lambda x: x[0])
        top_individuals = [weights for _, weights in fitness_scores[:population_size // 2]]

        new_population = []
        for weights in top_individuals:
            mutated_weights = [max(0, w + np.random.normal(0, mutation_rate)) for w in weights]
            total_weight = sum(mutated_weights)
            new_population.append([w / total_weight for w in mutated_weights])

        # 새로운 개체군 생성
        population = top_individuals + new_population
        print(f"Best Meta Loss in Generation {generation + 1}: {best_loss}")

    # pruning 적용 후 최적 가중치 반환
    best_weights = [w if w >= pruning_threshold else 0 for w in best_weights]
    total_weight = sum(best_weights)
    if total_weight > 0:
        best_weights = [w / total_weight for w in best_weights]
    else:
        print("All weights were pruned. Stopping.")
        return None
    
    print("Best Weights:", best_weights)

    return best_weights



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
        
        # os.path.join(config['save_path'], 'reins_focal1'),
        # os.path.join(config['save_path'], 'reins_focal2'),
        # os.path.join(config['save_path'], 'reins_focal3'),
        # os.path.join(config['save_path'], 'reins_focal4'),
        # os.path.join(config['save_path'], 'reins_focal5'),
        
        os.path.join(config['save_path'], 'reins_adafocal1'),
        os.path.join(config['save_path'], 'reins_adafocal2'),
        os.path.join(config['save_path'], 'reins_adafocal3'),
        os.path.join(config['save_path'], 'reins_adafocal4'),
    ]

    variant = dino_variant._small_variant
    models = initialize_models(save_paths, variant, config, device)
    test_loader, valid_loader = setup_data_loaders(args, data_path, batch_size)

    optimized_weights = optimize_weights_evolution_strategy(models, valid_loader, device, pruning_threshold=0.05, population_size=10, generations=3, mutation_rate=0.1)

    ensemble_params = calculate_ensemble_params(models, optimized_weights)
    models[0].load_state_dict(ensemble_params)
    models[0].eval()

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
