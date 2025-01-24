import os

import sys
sys.path.append("/home/youmin/workspace/VFMs-Adapters-Ensemble/adapter_ensemble")
import torch
import torch.nn as nn
from torchvision import models
from utils import read_conf, validation_accuracy, evaluate
from data import cifar10, ham10000
import argparse

import timm




device = 'cuda:0'

def load_model(model_type, config, save_path):
    """
    모델을 로드하고 체크포인트를 적용하는 함수
    """
    if model_type == 'resnet':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, config['num_classes'])
    elif model_type == 'vgg':
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, config['num_classes'])
    elif model_type == 'densenet':
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, config['num_classes'])
    else:
        raise ValueError("Invalid model type")
    
    model = model.to(device)
    state_dict = torch.load(os.path.join(save_path, 'last.pth.tar'), map_location='cpu')['state_dict']
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

def ensemble_evaluate(models, test_loader, device):
    outputs = []  
    targets = []  
    correct = 0    
    total = 0       

    # Disable gradient computation for evaluation
    with torch.no_grad():
        
        # Iterate through the test dataset loader
        for inputs, target in test_loader:
            inputs, target = inputs.to(device), target.to(device)
            
            
            batch_outputs = []                                              # List to store outputs of individual models for the current batch   
            
            # Iterate through all models in the ensemble
            for model in models:
                output = model(inputs)                                      # Perform forward pass through the model
                output = torch.softmax(output, dim=1)                       # Apply softmax to get class probabilities
                batch_outputs.append(output)                                # Store the model's output
            
            
            # Average the outputs of all models in the ensemble
            ensemble_output = torch.stack(batch_outputs).mean(dim=0)
            outputs.append(ensemble_output.cpu())                           # Store the ensemble output for the current batch
            targets.append(target.cpu())                                    # Store the true labels for the current batch
            
            
            # Get the predicted class by finding the index of the maximum probability
            _, predicted = torch.max(ensemble_output, dim=1)
            correct += (predicted == target).sum().item()                   # Count the correct predictions
            total += target.size(0)                                         # Update the total number of samples
    
    # Calculate accuracy as a percentage
    accuracy = correct / total * 100
    print(f"Ensemble Accuracy: {accuracy:.2f}%")

    
    # 평가 결과 출력
    outputs = torch.cat(outputs).numpy()
    targets = torch.cat(targets).numpy()
    evaluate(outputs, targets, verbose=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='ham10000')
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--type', '-t', default= 'rein', type=str)
    args = parser.parse_args()

    config = read_conf('conf/data/' + args.data + '.yaml')
    data_path = config['data_root']
    batch_size = int(config['batch_size'])
    
    save_path_resent = '/home/youmin/workspace/VFMs-Adapters-Ensemble/adapter_ensemble/checkpoints/checkpoints_ham10000/resnet50'
    save_path_vgg = '/home/youmin/workspace/VFMs-Adapters-Ensemble/adapter_ensemble/checkpoints/checkpoints_ham10000/vgg16'
    save_path_densenet = '/home/youmin/workspace/VFMs-Adapters-Ensemble/adapter_ensemble/checkpoints/checkpoints_ham10000/densenet121'

    # Test data 로드
    if args.data == 'cifar10':
        test_loader = cifar10.get_test_loader(batch_size, shuffle=False, num_workers=4, pin_memory=True, data_dir=data_path)
    elif args.data == 'ham10000':
        train_loader, valid_loader, test_loader = ham10000.get_dataloaders(data_path, batch_size=32, num_workers=4)

    # state_dict_resnet = torch.load(os.path.join(save_path_resent, 'last.pth.tar'), map_location='cpu')['state_dict']  
    # state_dict_vgg = torch.load(os.path.join(save_path_vgg, 'last.pth.tar'), map_location='cpu')['state_dict']
    # state_dict_densenet = torch.load(os.path.join(save_path_densenet, 'last.pth.tar'), map_location='cpu')['state_dict']

    # 모델 로드
    resnet = load_model('resnet', config, save_path_resent)
    vgg = load_model('vgg', config, save_path_vgg)
    densenet = load_model('densenet', config, save_path_densenet)

    # 앙상블 평가
    ensemble_evaluate([resnet, vgg, densenet], test_loader, device)

if __name__ == '__main__':
    main()
