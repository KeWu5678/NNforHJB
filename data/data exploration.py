import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data_path = 'data/VDP_beta_3_patch1.npy'
data = np.load(data_path)

print(data.shape)

# Filter out NaN values if any
valid_data = data[~np.isnan(data['v'])]
print(f"Valid data points: {valid_data.shape[0]}")

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Extract x0 (which has 2 components) and V
x0_comp1 = valid_data['x'][:, 0]  # First component of x0
x0_comp2 = valid_data['x'][:, 1]  # Second component of x0
v_values = valid_data['v']

# Create scatter plot
scatter = ax.scatter(x0_comp1, x0_comp2, v_values, c=v_values, cmap='viridis', 
                    s=50, alpha=0.8)

# Add a color bar
cbar = fig.colorbar(scatter, ax=ax, shrink=0.7)
cbar.set_label('Value Function (V)')

# Set labels and title
ax.set_xlabel('x₀[0]')
ax.set_ylabel('x₀[1]')
ax.set_zlabel('Value Function (V)')
ax.set_title('Relationship between Initial Conditions and Value Function')

# Adjust view angle
ax.view_init(elev=30, azim=45)

plt.tight_layout()
plt.savefig('VDP_value_function_plot.png', dpi=300)
plt.show()

# 2D plots to see individual component relationships
plt.figure(figsize=(16, 6))

# Plot V vs x0[0]
plt.subplot(121)
plt.scatter(x0_comp1, v_values, alpha=0.7, c=x0_comp2, cmap='coolwarm')
plt.colorbar(label='x₀[1]')
plt.xlabel('x₀[0]')
plt.ylabel('Value Function (V)')
plt.title('Value Function vs First Component of x₀')

# Plot V vs x0[1]
plt.subplot(122)
plt.scatter(x0_comp2, v_values, alpha=0.7, c=x0_comp1, cmap='coolwarm')
plt.colorbar(label='x₀[0]')
plt.xlabel('x₀[1]')
plt.ylabel('Value Function (V)')
plt.title('Value Function vs Second Component of x₀')

plt.tight_layout()
plt.savefig('VDP_2D_plots.png', dpi=300)
plt.show()

# Create a heatmap to visualize the density of points
from scipy.interpolate import griddata

# Create a grid for interpolation
grid_x, grid_y = np.mgrid[min(x0_comp1):max(x0_comp1):100j, 
                          min(x0_comp2):max(x0_comp2):100j]

# Interpolate V values onto the grid
grid_z = griddata((x0_comp1, x0_comp2), v_values, (grid_x, grid_y), method='cubic')

# Plot the heatmap
plt.figure(figsize=(10, 8))
plt.pcolormesh(grid_x, grid_y, grid_z, cmap='viridis', shading='auto')
plt.colorbar(label='Value Function (V)')
plt.xlabel('x₀[0]')
plt.ylabel('x₀[1]')
plt.title('Value Function Heatmap')
plt.savefig('VDP_heatmap.png', dpi=300)
plt.show()



