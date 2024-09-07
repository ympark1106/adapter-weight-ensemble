import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score, balanced_accuracy_score
import torch.nn.functional as F

recall_level_default = 0.95

def validation_accuracy(model, loader, device, mode = 'rein'):
    total = 0
    correct = 0
    
    def linear(model, inputs):
        f = model(inputs)
        outputs = model.linear(f)
        return outputs
    
    def rein(model, inputs):
        f = model.forward_features(inputs)
        f = f[:, 0, :]
        outputs = model.linear(f)
        return outputs
    
    def rein3(model, inputs):
        f = model.forward_features1(inputs)
        f = f[:, 0, :]
        outputs1 = model.linear(f)

        f = model.forward_features2(inputs)
        f = f[:, 0, :]
        outputs2 = model.linear(f)

        f = model.forward_features3(inputs)
        f = f[:, 0, :]
        outputs3 = model.linear(f)
        return outputs1 + outputs2 + outputs3


    def no_rein(model, inputs):
        f = model.forward_features_no_rein(inputs)
        f = f[:, 0, :]
        outputs = model.linear(f)
        return outputs
    
    def resnet(model, inputs):
        outputs = model(inputs)
        return outputs
    if mode == 'rein':
        out = rein
    elif mode == 'no_rein':
        out = no_rein
    elif mode == 'rein3':
        out = rein3
    elif mode == 'resnet':
        out = resnet
    else:
        out = linear

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if targets.ndim > 1 and targets.size(1) > 1:
                targets = torch.argmax(targets, dim=1)
            outputs = out(model, inputs)
            _, predicted = outputs.max(1)  
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    valid_accuracy = correct/total
    return valid_accuracy