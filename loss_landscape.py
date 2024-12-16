import rein
import os
import torch
import torch.nn as nn
import dino_variant
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import argparse
from utils import read_conf
from torch.cuda.amp.autocast_mode import autocast
from data import cifar10, cifar100, cub, ham10000, bloodmnist, pathmnist
from losses import RankMixup_MNDCG, RankMixup_MRL, focal_loss, focal_loss_adaptive_gamma


def lora_forward(model, inputs):
    with autocast(enabled=True):
        features = model.forward_features(inputs)
        output = model.linear(features)
        output = torch.softmax(output, dim=1)
    return output

parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', type=str, default='cub')
parser.add_argument('--gpu', '-g', default = '0', type=str)
parser.add_argument('--netsize', default='s', type=str)
parser.add_argument('--save_path', '-s', type=str)
args = parser.parse_args()

config = read_conf('conf/data/'+args.data+'.yaml')
device = 'cuda:'+args.gpu
# save_path = os.path.join(config['save_path'], args.save_path)
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

model_load = dino_variant._small_dino
variant = dino_variant._small_variant


dino = torch.hub.load('facebookresearch/dinov2', model_load)
dino_state_dict = dino.state_dict()
new_state_dict = dict()

for k in dino_state_dict.keys():
    new_k = k.replace("attn.qkv", "attn.qkv.qkv")
    new_state_dict[new_k] = dino_state_dict[k]
    
model = rein.LoRADinoVisionTransformer(dino)
model.dino.load_state_dict(new_state_dict, strict=False)
model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])

model.eval()

criterion = focal_loss.FocalLoss(gamma=3)

weights = model.state_dict()
trainable_keys = [k for k, v in weights.items() if v.requires_grad]
trainable_weights = {k: v.clone() for k, v in weights.items() if k in trainable_keys}
# state_dict = torch.load(os.path.join(save_path, 'last.pth.tar'), map_location='cpu')['state_dict']

grid_size = 10
delta = 0.1
loss_grid = np.zeros((grid_size, grid_size))
for i in range(grid_size):
    for j in range(grid_size):
        start = time.time()
        for k, v in weights.items():
            weights[k] = weights[k] + delta * (i - grid_size // 2) * v + delta * (
                j - grid_size // 2) * torch.randn_like(v)
        model.load_state_dict(weights)
        print('change weight takes',time.time()-start)
        running_loss = 0.0
        for _, data in tqdm(enumerate(train_loader, 0)):
            inputs, labels = data

            outputs = lora_forward(model, inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

        loss_grid[i, j] = running_loss
        print('all_process',time.time()-start)
        
        
plt.imshow(loss_grid, cmap='jet', interpolation='nearest')
plt.colorbar()
plt.title('Loss Landscape')
plt.xlabel('Delta X')
plt.ylabel('Delta Y')
plt.savefig('loss_landscape.png', dpi=300)  
plt.show()