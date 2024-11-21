import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

import os
import torch
import torch.nn as nn
import numpy as np
from utils import read_conf, evaluate

import argparse
import rein
import dino_variant
from data import cifar10, cifar100, cub, ham10000, bloodmnist, pathmnist

def evaluate_with_mc_dropout(model, loader, device, num_samples=10):
    model.train()  # Dropout 활성화
    total = 0
    correct = 0
    outputs_all = []
    targets_all = []

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        batch_outputs = []

        # 여러 번 샘플링하여 Dropout 효과를 적용
        for _ in range(num_samples):
            with torch.no_grad():
                output = model.forward_features(inputs)[:, 0, :]
                output = model.linear(output)
                output = torch.softmax(output, dim=1)
                batch_outputs.append(output)

        # 샘플링한 결과의 평균을 최종 예측으로 사용
        averaged_output = torch.mean(torch.stack(batch_outputs), dim=0)
        _, predicted = averaged_output.max(1)
        
        # 정확도 계산
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

        # 평가 메트릭 계산을 위해 모든 배치 출력과 타겟을 저장
        outputs_all.append(averaged_output.cpu())
        targets_all.append(targets.cpu())

    # 정확도 출력
    accuracy = correct / total
    print(f'MC Dropout 평가 정확도: {accuracy:.4f}')
    
    # 전체 평가 메트릭 계산
    outputs_all = torch.cat(outputs_all).numpy()
    targets_all = torch.cat(targets_all).numpy()
    evaluate(outputs_all, targets_all, verbose=True)

    return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='cub')
    parser.add_argument('--gpu', '-g', default='0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--num_samples', type=int, default=10, help='MC Dropout 샘플링 횟수')
    args = parser.parse_args()

    config = read_conf('conf/data/' + args.data + '.yaml')
    device = 'cuda:' + args.gpu
    save_path = os.path.join(config['save_path'], args.save_path)
    data_path = config['data_root']
    batch_size = int(config['batch_size'])

    if args.data == 'cifar10':
        test_loader = cifar10.get_test_loader(batch_size, shuffle=True, num_workers=4, pin_memory=True, get_val_temp=0, data_dir=data_path)
    elif args.data == 'cifar100':
        test_loader = cifar100.get_test_loader(data_dir=data_path, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    elif args.data == 'cub':
        test_loader = cub.get_test_loader(data_path, batch_size=32, scale_size=256, crop_size=224, num_workers=4, pin_memory=True)
    elif args.data == 'ham10000':
        train_loader, valid_loader, test_loader = ham10000.get_dataloaders(data_path, batch_size=32, num_workers=4)
    elif args.data == 'bloodmnist':
        train_loader, test_loader, valid_loader = bloodmnist.get_dataloader(batch_size, download=True, num_workers=4)
    elif args.data == 'pathmnist':
        train_loader, test_loader, valid_loader = pathmnist.get_dataloader(batch_size, download=True, num_workers=4)
        
    # 모델 초기화
    if args.netsize == 's':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant
    model = torch.hub.load('facebookresearch/dinov2', model_load)
    dino_state_dict = model.state_dict()

    # MC Dropout을 위한 모델 로드
    model = rein.ReinsDinoVisionTransformer_Dropout(
        **variant,
        dropout_rate=0.5
    )
    model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.load_state_dict(dino_state_dict, strict=False)
    model.to(device)
    
    print(model)

    # 체크포인트에서 모델 파라미터 불러오기
    state_dict = torch.load(os.path.join(save_path, 'last.pth.tar'), map_location='cpu')['state_dict']
    model.load_state_dict(state_dict, strict=True)
    
    # MC Dropout 평가
    accuracy = evaluate_with_mc_dropout(model, test_loader, device, args.num_samples)
    print(f'MC Dropout 평가 최종 정확도: {accuracy:.4f}')

if __name__ == '__main__':
    main()