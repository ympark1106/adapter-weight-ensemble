import matplotlib.pyplot as plt
import numpy as np

# Initial settings for the original LR schedule
epochs = 220  # Total epochs for the original schedule
initial_lr = 1e-3
lr_decay_50 = 1e-4
cycle_length = 30  # Cycle length for CosineAnnealingWarmRestarts
eta_min = 1e-5  # Minimum learning rate in cyclic schedule

# Initialize learning rate array for the original schedule
lr_values = []
current_lr = initial_lr

# Original LR schedule
for epoch in range(epochs):
    if epoch < 50:
        current_lr = initial_lr
    elif epoch < 70:
        current_lr = lr_decay_50
    else:
        # CosineAnnealingWarmRestarts starts at epoch 70
        cycle_epoch = (epoch - 70) % cycle_length
        cos_lr = eta_min + 0.5 * (initial_lr - eta_min) * (1 + np.cos(np.pi * cycle_epoch / cycle_length))
        current_lr = cos_lr
    lr_values.append(current_lr)

# Settings for the MultiStepLR schedule
max_epoch = 100  # Maximum epochs for MultiStepLR
lr_decay_milestones = [int(0.5 * max_epoch), int(0.75 * max_epoch), int(0.9 * max_epoch)]  # [50, 75, 90]
gamma = 0.1  # Decay factor

# Initialize learning rate array for the MultiStepLR schedule
lr_multistep = []
for epoch in range(max_epoch):
    # Count how many milestones have been passed
    milestones_passed = sum(epoch >= milestone for milestone in lr_decay_milestones)
    # Calculate the current LR based on the number of milestones passed
    current_lr_ms = initial_lr * (gamma ** milestones_passed)
    lr_multistep.append(current_lr_ms)

# Plotting the original LR schedule
plt.figure(figsize=(12, 7))
plt.plot(range(epochs), lr_values, label="Snapshot Soup LR Schedule", color='blue')

# Plotting the MultiStepLR schedule up to max_epoch
plt.plot(range(max_epoch), lr_multistep, alpha=0.7, label="Single Model LR Schedule", color='red')

# Adding vertical lines to indicate key points
plt.axvline(x=70, color='purple', alpha=0.5, label='Branching Point in Snapshot Soup', linestyle='--')

# Labels and title
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedules")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
