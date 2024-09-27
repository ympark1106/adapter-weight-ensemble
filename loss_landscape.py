import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import rein
import dino_variant
from eval import rein_forward

model_load = dino_variant._small_dino
variant = dino_variant._small_variant

# 가중치 보간 함수 정의
def interpolate_weights(state_dict1, model_or_state_dict2, alpha):
    interpolated_state_dict = {}

    # 두 번째 인자가 모델인 경우 state_dict 가져오기
    if isinstance(model_or_state_dict2, torch.nn.Module):
        state_dict2 = model_or_state_dict2.state_dict()
    else:
        state_dict2 = model_or_state_dict2  # 이미 딕셔너리인 경우 그대로 사용

    # 딕셔너리 형태로 보간, 특정 키 필터링 (linear 계층 무시)
    for key in state_dict1:
        if key in state_dict2 and not key.startswith('linear'):
            interpolated_state_dict[key] = (1 - alpha) * state_dict1[key] + alpha * state_dict2[key]
    return interpolated_state_dict


# 손실 계산 함수
def compute_loss(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    outputs = []
    targets = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = rein_forward(model, inputs)
            outputs.append(output.cpu())
            targets.append(targets.cpu())
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(data_loader)

# 손실 경관에 모델 위치 표시
def plot_loss_landscape_with_models(model1, model2, model3, data_loader, criterion, device, save_path='loss_landscape_with_models.png', num_classes=200):
    alphas = np.linspace(0, 1, 50)
    betas = np.linspace(0, 1, 50)
    loss_landscape = np.zeros((len(alphas), len(betas)))

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # Model 1과 Model 2를 alpha로 보간한 후 Model 3과 beta로 보간
            temp_state_dict = interpolate_weights(model1.state_dict(), model2.state_dict(), alpha)
            interpolated_state_dict = interpolate_weights(temp_state_dict, model3.state_dict(), beta)
            
            # 임시 모델 생성
            temp_model = rein.ReinsDinoVisionTransformer(**variant)
            temp_model.load_state_dict(interpolated_state_dict)
            temp_model.linear = nn.Linear(variant['embed_dim'], num_classes)
            temp_model.to(device)

            # 손실 계산
            loss = compute_loss(temp_model, data_loader, criterion, device)
            loss_landscape[i, j] = loss

    # 손실 경관 그리기
    plt.figure(figsize=(10, 8))
    X, Y = np.meshgrid(alphas, betas)
    contour = plt.contourf(X, Y, loss_landscape, 20, cmap='viridis')
    plt.colorbar(contour)

    # 각 모델의 위치 계산 및 표시
    loss1 = compute_loss(model1, data_loader, criterion, device)
    loss2 = compute_loss(model2, data_loader, criterion, device)
    loss3 = compute_loss(model3, data_loader, criterion, device)
    
    plt.scatter(0, 0, color='red', label=f'Model 1: Loss = {loss1:.4f}')
    plt.scatter(1, 0, color='blue', label=f'Model 2: Loss = {loss2:.4f}')
    plt.scatter(0, 1, color='green', label=f'Model 3: Loss = {loss3:.4f}')
    
    plt.title('Loss Landscape with Model Positions')
    plt.xlabel('Alpha (Model 1 to Model 2)')
    plt.ylabel('Beta (Blended with Model 3)')
    plt.legend()
    
    # 이미지 저장
    plt.savefig(save_path)
    plt.close()


