import matplotlib.pyplot as plt
import numpy as np

# Initial settings
epochs = 220  # Total epochs
initial_lr = 1e-3
lr_decay_50 = 1e-4
cycle_length = 30  # Cycle length for CosineAnnealingWarmRestarts
eta_min = 1e-5  # Minimum learning rate in cyclic schedule

# Initialize learning rate array
lr_values = []
current_lr = initial_lr

# LR schedule
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

# Plotting the LR schedule
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), lr_values, label="Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
# plt.axvline(x=50, color='r', linestyle='--', label='Epoch 50 (1e-3 -> 1e-4)')
plt.axvline(x=70, color='r', alpha = 0.5, linestyle='--', label='Braching Point (Cyclic LR Start)')
plt.legend()
plt.grid()
plt.show()



