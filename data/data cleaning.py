import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import os
import pickle

# Define the file paths for all VDP beta files


combined_data = np.load('data_vdp_legendre_beta=3.npy')

# Filter out NaN values if any
valid_data = combined_data[~np.isnan(combined_data['v'])]
print(f"Valid data points: {valid_data.shape[0]}")

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Extract x0 (which has 2 components) and V
x0_comp1 = valid_data['x'][:, 0]  # First component of x0
x0_comp2 = valid_data['x'][:, 1]  # Second component of x0
v_values = valid_data['v']

# Extract all initial conditions (x0) and save as NumPy file
x0_values = valid_data['x']  # All x0 values
print(f"Extracted {len(x0_values)} x0 values, shape: {x0_values.shape}")

# Save as NumPy file - only x0 values
# x0_npy_path = 'vdp_x0.npy'
# np.save(x0_npy_path, x0_values)
# print(f"Saved x0 values to {x0_npy_path}")

# Print statistics
print("\nStatistics of combined data:")
print(f"Min value: {np.min(v_values)}")
print(f"Max value: {np.max(v_values)}")
print(f"Mean value: {np.mean(v_values)}")
print(f"Std deviation: {np.std(v_values)}")

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
ax.set_title('Combined VDP Data (β=3): Initial Conditions vs Value Function')
# Adjust view angle
ax.view_init(elev=30, azim=45)
plt.tight_layout()

# Save as regular image
plt.savefig('VDP_legendre_plot.png', dpi=300)

# Save the figure as a pickle for interactive use later
pickle_file = 'VDP_legendre_plot.pickle'
with open(pickle_file, 'wb') as f:
    pickle.dump(fig, f)
print(f"Saved interactive plot to {pickle_file}")

# You can later load and interact with the figure using:
# with open('VDP_legendre_plot.pickle', 'rb') as f:
#     fig = pickle.load(f)
# plt.show()

plt.show()

# # 2D plots to see individual component relationships
# plt.figure(figsize=(16, 6))

# # Plot V vs x0[0]
# plt.subplot(121)
# plt.scatter(x0_comp1, v_values, alpha=0.7, c=x0_comp2, cmap='coolwarm')
# plt.colorbar(label='x₀[1]')
# plt.xlabel('x₀[0]')
# plt.ylabel('Value Function (V)')
# plt.title('Combined VDP Data (β=3): Value Function vs First Component of x₀')
# # Plot V vs x0[1]
# plt.subplot(122)
# plt.scatter(x0_comp2, v_values, alpha=0.7, c=x0_comp1, cmap='coolwarm')
# plt.colorbar(label='x₀[0]')
# plt.xlabel('x₀[1]')
# plt.ylabel('Value Function (V)')
# plt.title('Combined VDP Data (β=3): Value Function vs Second Component of x₀')
# plt.tight_layout()
# plt.savefig('VDP_combined_2D_plots.png', dpi=300)
# plt.show()


# # Create a heatmap to visualize the density of points
# # Create a grid for interpolation
# grid_x, grid_y = np.mgrid[min(x0_comp1):max(x0_comp1):100j, 
#                           min(x0_comp2):max(x0_comp2):100j]

# # Interpolate V values onto the grid
# grid_z = griddata((x0_comp1, x0_comp2), v_values, (grid_x, grid_y), method='cubic')
# # Plot the heatmap
# plt.figure(figsize=(10, 8))
# plt.pcolormesh(grid_x, grid_y, grid_z, cmap='viridis', shading='auto')
# plt.colorbar(label='Value Function (V)')
# plt.xlabel('x₀[0]')
# plt.ylabel('x₀[1]')
# plt.title('Combined VDP Data (β=3): Value Function Heatmap')
# plt.savefig('VDP_combined_heatmap.png', dpi=300)
# plt.show()

# # Plot the density of data points
# plt.figure(figsize=(10, 8))
# plt.hist2d(x0_comp1, x0_comp2, bins=50, cmap='jet')
# plt.colorbar(label='Count')
# plt.xlabel('x₀[0]')
# plt.ylabel('x₀[1]')
# plt.title('Density of Sample Points')
# plt.savefig('VDP_combined_density.png', dpi=300)
# plt.show()



