import matplotlib.pyplot as plt
import numpy as np

# Generate 2D grid for plotting
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)

# Define the loss landscape with multiple local minima
Z = (
    np.sin(0.3 * X) * np.cos(0.3 * Y) + 
    0.2 * np.sin(0.6 * X) * np.cos(0.6 * Y) +
    0.1 * np.sin(1.2 * X) * np.cos(1.2 * Y)
)

# Adjust Z axis range to better match the graph boundaries
Z_min, Z_max = Z.min(), Z.max()

# Create the plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9, edgecolor='none')

# Contour plot at the bottom
contour_offset = Z_min - 0.2  # Slight offset for visibility
ax.contour(X, Y, Z, levels=15, zdir='z', offset=contour_offset, cmap='viridis', linestyles="solid")

# Labels and formatting
# ax.set_title("Illustration of Loss Landscape with Multiple Local Minima")
# ax.set_xlabel("Parameter 1")
# ax.set_ylabel("Parameter 2")
# ax.set_zlabel("Loss")

# Set axis limits to match the graph area perfectly
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(contour_offset, Z_max + 0.1)
ax.view_init(elev=30, azim=-60)  # Adjust the viewing angle

# Display the plot
plt.tight_layout()
plt.show()
