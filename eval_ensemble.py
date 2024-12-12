import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

import os
import torch
import torch.nn as nn
import argparse
import numpy as np

import torch.nn.functional as F
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

def compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true):
    true = np.array(true).reshape(-1)
    pred = np.array(pred).reshape(-1)
    conf = np.array(conf).reshape(-1)

    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0, 0, 0
    else:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])
        len_bin = len(filtered_tuples)
        avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin
        accuracy = float(correct) / len_bin
        return accuracy, avg_conf, len_bin

def ECE(conf, pred, true, bin_size=0.1):
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)
    n = len(conf)
    ece = 0
    for conf_thresh in upper_bounds:
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)
        ece += np.abs(acc - avg_conf) * len_bin / n
    return ece

def OE(conf, pred, true, bin_size = 0.1):
    
    """
    Expected Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
        
    Returns:
        ece: expected calibration error
    """
    true = np.array(true).reshape(-1)  
    pred = np.array(pred).reshape(-1)
    conf = np.array(conf).reshape(-1)

    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)  # Get bounds of bins
    
    n = len(conf)
    ece = 0  # Starting error
    
    for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)  
        # print(acc, avg_conf, len_bin)
        if avg_conf > acc:
            ece += avg_conf * (avg_conf - acc) * len_bin / n 
        
    return ece


def evaluate_ensemble(outputs, targets, bins=15):
    probs = np.array(outputs)
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    
    ece = ECE(confs, preds, targets, bin_size=1/bins)
    oe = OE(confs, preds, targets, bin_size = 1/bins)
    
    
    print("=== Evaluation Results ===")
    print(f"ECE: {ece:.4f}")
    print(f"OE: {oe:.4f}")
    print("==========================")

def ensemble_evaluate(models, test_loader, device):
    outputs = []
    targets = []
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, target in test_loader:
            inputs, target = inputs.to(device), target.to(device)
            # if targets.ndim > 1 and targets.size(1) > 1:
            #     targets = torch.argmax(targets, dim=1)
            # if targets.ndim > 1:
            #     targets = targets.view(-1)
                
            batch_outputs = []
            for model in models:
                output = rein_forward(model, inputs)
                batch_outputs.append(output)

            ensemble_output = torch.stack(batch_outputs).mean(dim=0)
            outputs.append(ensemble_output.cpu().numpy())
            targets.append(target.cpu().numpy())

            _, predicted = torch.max(ensemble_output, dim=1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    accuracy = correct / total * 100
    print(f"Ensemble Accuracy: {accuracy:.2f}%")

    outputs = np.vstack(outputs)
    targets = np.concatenate(targets)
    evaluate_ensemble(outputs, targets)

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
    
    ensemble_evaluate(models, test_loader, device)
    

if __name__ == '__main__':
    train()
