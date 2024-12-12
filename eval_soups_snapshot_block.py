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
        state_dict = torch.load(save_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        models.append(model)
    return models

def get_model_from_sd(state_dict, variant, config, device):
    model = rein.ReinsDinoVisionTransformer(**variant)
    model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.load_state_dict(state_dict, strict=True)
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


def extract_block_weights(models, block_prefixes):
    """
    모델들의 특정 블록의 가중치를 추출.

    Args:
        models: 학습된 모델 리스트.
        block_prefixes: 블록 이름의 접두어 리스트.

    Returns:
        block_weights: {block_name: [model1_weight, model2_weight, ...]} 구조의 딕셔너리.
    """
    block_weights = {prefix: [] for prefix in block_prefixes}
    for model in models:
        state_dict = model.state_dict()
        for prefix in block_prefixes:
            block_weights[prefix].append({k: v for k, v in state_dict.items() if k.startswith(prefix)})
    return block_weights


def average_block_weights(block_weights):
    """
    블록별 가중치를 평균화.

    Args:
        block_weights: {block_name: [model1_weight, model2_weight, ...]}.

    Returns:
        averaged_weights: {block_name: 평균화된 가중치}.
    """
    averaged_weights = {}
    for block_name, weight_list in block_weights.items():
        # 블록의 각 파라미터별 평균 계산
        averaged_block = {}
        keys = weight_list[0].keys()
        for key in keys:
            averaged_block[key] = torch.mean(torch.stack([weights[key] for weights in weight_list]), dim=0)
        averaged_weights[block_name] = averaged_block
    return averaged_weights


def apply_averaged_weights_to_model(base_model, averaged_weights):
    """
    평균화된 가중치를 모델에 적용.

    Args:
        base_model: 초기화된 모델 (가중치를 업데이트할 모델).
        averaged_weights: {block_name: 평균화된 가중치}.

    Returns:
        base_model: 업데이트된 모델.
    """
    model_dict = base_model.state_dict()  # 기본 모델의 전체 가중치 가져오기
    for block_name, weights in averaged_weights.items():
        for k, v in weights.items():
            if k in model_dict:
                model_dict[k] = v  # 평균화된 가중치 덮어쓰기
            else:
                print(f"Warning: {k} not found in model state_dict.")
    base_model.load_state_dict(model_dict)  # 업데이트된 가중치 로드
    return base_model

def greedy_soup_ensemble(models, model_names, valid_loader, device, variant, config):
    block_prefixes = ['reins', 'linear']  # Rein Adapter와 Linear Layer의 접두어
    print("Performing Sub-Block Weight Averaging...")

    # 각 블록별 가중치 추출
    block_weights = extract_block_weights(models, block_prefixes)

    # 블록별 평균화
    averaged_weights = average_block_weights(block_weights)

    # 최종 모델 생성 및 적용
    final_model = rein.ReinsDinoVisionTransformer(**variant)
    final_model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    final_model = apply_averaged_weights_to_model(final_model, averaged_weights)
    final_model.to(device)
    final_model.eval()

    return averaged_weights, final_model



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

        os.path.join(config['save_path'], 'reins_focal_hydra/cyclic_checkpoint_epoch99.pth'),
        os.path.join(config['save_path'], 'reins_focal_hydra/cyclic_checkpoint_epoch129.pth'),
        os.path.join(config['save_path'], 'reins_focal_hydra/cyclic_checkpoint_epoch159.pth'),
        os.path.join(config['save_path'], 'reins_focal_hydra/cyclic_checkpoint_epoch189.pth'),
        os.path.join(config['save_path'], 'reins_focal_hydra/cyclic_checkpoint_epoch219.pth'),
        
        # os.path.join(config['save_path'], 'reins_focal_hydra_1/cyclic_checkpoint_epoch99.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra_1/cyclic_checkpoint_epoch129.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra_1/cyclic_checkpoint_epoch159.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra_1/cyclic_checkpoint_epoch189.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra_1/cyclic_checkpoint_epoch219.pth'),
    
        # os.path.join(config['save_path'], 'reins_focal_hydra_2/cyclic_checkpoint_epoch89.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra_2/cyclic_checkpoint_epoch129.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra_2/cyclic_checkpoint_epoch169.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra_2/cyclic_checkpoint_epoch209.pth'),
    ]
    
    model_names = [os.path.basename(path) for path in save_paths]

    variant = dino_variant._small_variant
    models = initialize_models(save_paths, variant, config, device)
    test_loader, valid_loader = setup_data_loaders(args, data_path, batch_size)
    
    # greedy_soup_params, model = greedy_soup_ensemble(models, model_names, valid_loader, device, variant, config)
    averaged_weights, model = greedy_soup_ensemble(models, model_names, valid_loader, device, variant, config)

    final_model = rein.ReinsDinoVisionTransformer(**variant)
    final_model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    final_model = apply_averaged_weights_to_model(final_model, averaged_weights)
    final_model.to(device)
    final_model.eval()
    

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
