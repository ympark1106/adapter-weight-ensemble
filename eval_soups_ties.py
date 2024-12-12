import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")
import sys
sys.path.append("/home/youmin/workspace/VFMs-Adapters-Ensemble/adapter_ensemble")
import os
import torch
import torch.nn as nn
import argparse
import numpy as np

from utils import read_conf, validation_accuracy, ModelWithTemperature, validate, evaluate, calculate_ece, calculate_nll, state_dict_to_vector, vector_to_state_dict, add_ptm_to_tv, check_parameterNamesMatch, check_state_dicts_equal, ties_merging
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


def ties_merge_ensemble(models, variant, config, device, K=20, merge_func="dis-mean", lamda=1):
    # 1. 모든 모델의 state_dict를 벡터화
    state_dicts = [model.state_dict() for model in models]
    flat_state_dicts = [state_dict_to_vector(sd) for sd in state_dicts]
    
    # 2. PTM(기본 모델) 생성 및 벡터화
    ptm_model = rein.ReinsDinoVisionTransformer(**variant)
    ptm_model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    ptm_model.to(device)
    ptm_state_dict = ptm_model.state_dict()
    flat_ptm = state_dict_to_vector(ptm_state_dict)
    
    # 3. 작업 벡터 계산
    task_vectors = [flat - flat_ptm for flat in flat_state_dicts]
    task_vectors = torch.stack(task_vectors)
    
    # 4. TIES-MERGING 병합
    merged_vector = ties_merging(
        flat_task_checks=task_vectors,
        reset_thresh=K,
        merge_func=merge_func
    )
    
    # 5. 병합된 모델 생성
    merged_state_dict = vector_to_state_dict(flat_ptm + lamda * merged_vector, ptm_state_dict)
    merged_model = rein.ReinsDinoVisionTransformer(**variant)
    merged_model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    merged_model.load_state_dict(merged_state_dict)
    merged_model.to(device)
    
    return merged_model

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
        # os.path.join(config['save_path'], 'reins_focal_swa/cyclic_checkpoint_epoch99.pth'),
        # os.path.join(config['save_path'], 'reins_focal_swa/cyclic_checkpoint_epoch129.pth'),
        # os.path.join(config['save_path'], 'reins_focal_swa/cyclic_checkpoint_epoch159.pth'),
        # os.path.join(config['save_path'], 'reins_focal_swa/cyclic_checkpoint_epoch189.pth'),
        # os.path.join(config['save_path'], 'reins_focal_swa/cyclic_checkpoint_epoch219.pth'),
        

        # os.path.join(config['save_path'], 'reins_focal_hydra/cyclic_checkpoint_epoch99.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra/cyclic_checkpoint_epoch129.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra/cyclic_checkpoint_epoch159.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra/cyclic_checkpoint_epoch189.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra/cyclic_checkpoint_epoch219.pth'),
        
        # os.path.join(config['save_path'], 'reins_focal_hydra_1/cyclic_checkpoint_epoch99.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra_1/cyclic_checkpoint_epoch129.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra_1/cyclic_checkpoint_epoch159.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra_1/cyclic_checkpoint_epoch189.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra_1/cyclic_checkpoint_epoch219.pth'),
        
        os.path.join(config['save_path'], 'reins_focal_hydra_1/cyclic_checkpoint_epoch89.pth'),
        os.path.join(config['save_path'], 'reins_focal_hydra_1/cyclic_checkpoint_epoch129.pth'),
        os.path.join(config['save_path'], 'reins_focal_hydra_1/cyclic_checkpoint_epoch169.pth'),
        os.path.join(config['save_path'], 'reins_focal_hydra_1/cyclic_checkpoint_epoch209.pth'),
        os.path.join(config['save_path'], 'reins_focal_hydra_1/cyclic_checkpoint_epoch249.pth'),
    
        # os.path.join(config['save_path'], 'reins_focal_hydra_2/cyclic_checkpoint_epoch89.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra_2/cyclic_checkpoint_epoch129.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra_2/cyclic_checkpoint_epoch169.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra_2/cyclic_checkpoint_epoch209.pth'),
        
        
        # os.path.join(config['save_path'], 'reins_focal_hydra_3/cyclic_checkpoint_epoch99.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra_3/cyclic_checkpoint_epoch129.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra_3/cyclic_checkpoint_epoch159.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra_3/cyclic_checkpoint_epoch189.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra_3/cyclic_checkpoint_epoch219.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra_3/cyclic_checkpoint_epoch249.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra_3/cyclic_checkpoint_epoch279.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra_3/cyclic_checkpoint_epoch309.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra_3/cyclic_checkpoint_epoch369.pth'),
        # os.path.join(config['save_path'], 'reins_focal_hydra_3/cyclic_checkpoint_epoch399.pth'),
    ]
    
    model_names = [os.path.basename(path) for path in save_paths]

    variant = dino_variant._small_variant
    models = initialize_models(save_paths, variant, config, device)
    test_loader, valid_loader = setup_data_loaders(args, data_path, batch_size)
    
    merged_model = ties_merge_ensemble(
                                        models=models,
                                        variant=variant,
                                        config=config,
                                        device=device,
                                        K=99,  # 상위 20% 매개변수만 유지
                                        merge_func="dis-weighted-mean",  # 평균 병합
                                        lamda=1  # 스케일링 파라미터
                                    )

    merged_model.eval()
    

    test_accuracy = validation_accuracy(merged_model, test_loader, device, mode=args.type)
    print('Test accuracy:', test_accuracy)

    outputs, targets = [], []
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs, target = inputs.to(device), target.to(device)
            output = rein_forward(merged_model, inputs)
            outputs.append(output.cpu())
            targets.append(target.cpu())
    
    outputs = torch.cat(outputs).numpy()
    targets = torch.cat(targets).numpy().astype(int)
    evaluate(outputs, targets, verbose=True)

if __name__ == '__main__':
    train()
