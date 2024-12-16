import matplotlib.pyplot as plt
import numpy as np

# Define the loss landscape function with multiple local minima
def loss_landscape(x, y):
    return (
        np.sin(0.2 * np.sqrt(x**2 + y**2)) + 0.5 * np.cos(0.3 * x) +
        0.5 * np.sin(0.3 * y) + 0.2 * np.sin(0.7 * x) * np.cos(0.7 * y)
    )

# Create a meshgrid for the x and y axes
x = np.linspace(-10, 10, 500)
y = np.linspace(-10, 10, 500)
X, Y = np.meshgrid(x, y)
Z = loss_landscape(X, Y)

# Plot the loss landscape
fig, ax = plt.subplots(figsize=(8, 6))
cp = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
cbar = plt.colorbar(cp)

# Add local minima positions (manually placed for visualization)
# local_minima_x = [-5, 0, 5, -4, 3, 7]
# local_minima_y = [-5, 4, -3, 2, -2, 5]
# ax.scatter(local_minima_x, local_minima_y, color='red', s=50, label="Local Minima")

# Customize the plot
# ax.set_title("Loss Landscape with Local Minima")
# ax.set_xlabel("Parameter 1")
# ax.set_ylabel("Parameter 2")
ax.legend()

plt.tight_layout()
plt.show()
