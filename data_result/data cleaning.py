import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
import os
import pickle

# Load and combine datasets
combined_data = np.load('VDP_beta_0.1_grid_30x30.npy')
print(f"Loaded original dataset with {len(combined_data)} points")

# Load fixed points data
# try:
#     fixed_points = np.load('VDP_beta_3_fixed_points_1.npy')
#     print(f"Loaded {len(fixed_points)} fixed points from VDP_beta_3_fixed_points_1.npy")
    
#     # Create a combined dataset with original data and fixed points
#     dtype = original_data.dtype
#     combined_data = np.zeros(len(original_data) + len(fixed_points), dtype=dtype)
#     combined_data[:len(original_data)] = original_data
    
#     # Add fixed points
#     added_count = 0
#     for i, fixed_point in enumerate(fixed_points):
#         # Check if this point exists in original data but has NaN values
#         found = False
#         for j, orig_point in enumerate(original_data):
#             if np.array_equal(fixed_point['x'], orig_point['x']):
#                 if np.isnan(orig_point['v']) or np.isnan(orig_point['dv']).any():
#                     # Replace the NaN entry in original data
#                     combined_data[j] = fixed_point
#                 found = True
#                 break
        
#         if not found:
#             # This is a new point, add it
#             combined_data[len(original_data) + added_count] = fixed_point
#             added_count += 1
    
#     # Trim the array if needed
#     if added_count < len(fixed_points):
#         combined_data = combined_data[:len(original_data) + added_count]
    
#     print(f"Added {added_count} new points to the dataset")
#     print(f"Combined dataset has {len(combined_data)} points")
# except FileNotFoundError:
#     print("VDP_beta_3_fixed_points_1.npy not found. Using only original data.")
#     combined_data = original_data

# Filter out NaN values if any
valid_data = combined_data[~np.isnan(combined_data['v'])]
print(f"Valid data points after filtering: {valid_data.shape[0]}")

# Extract x0 (which has 2 components) and V
x0_comp1 = valid_data['x'][:, 0]  # First component of x0
x0_comp2 = valid_data['x'][:, 1]  # Second component of x0
v_values = valid_data['v']
dv_values = valid_data['dv']  # Extract gradient vectors

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

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create scatter plot
scatter3d = ax.scatter(x0_comp1, x0_comp2, v_values, c=v_values, cmap='viridis', 
                   s=50, alpha=0.8)

# Add a color bar
cbar = fig.colorbar(scatter3d, ax=ax, shrink=0.7)
cbar.set_label('Value Function (V)')

# Set labels and title
ax.set_xlabel('x₀[0]')
ax.set_ylabel('x₀[1]')
ax.set_zlabel('Value Function (V)')
ax.set_title('Combined VDP Data (β=0.1): Initial Conditions vs Value Function')

# Adjust view angle
ax.view_init(elev=30, azim=45)
plt.tight_layout()

# Save as regular image
plt.savefig('VDP_3D_value_function.png', dpi=300)

# Save the figure as a pickle for interactive use later
pickle_file = 'VDP_3D_value_function.pickle'
with open(pickle_file, 'wb') as f:
    pickle.dump(fig, f)
print(f"Saved interactive 3D plot to {pickle_file}")

plt.show()

# Create a 2D plot to visualize gradient vectors
plt.figure(figsize=(12, 10))

# Create a scatter plot of the points
scatter = plt.scatter(x0_comp1, x0_comp2, c=v_values, cmap='viridis', 
                     s=30, alpha=0.6)

# Create a grid of points for more evenly distributed vectors
grid_size = 15  # Adjust this for more or fewer vectors
x_min, x_max = np.min(x0_comp1), np.max(x0_comp1)
y_min, y_max = np.min(x0_comp2), np.max(x0_comp2)
x_grid = np.linspace(x_min, x_max, grid_size)
y_grid = np.linspace(y_min, y_max, grid_size)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

# For each grid point, find the nearest data point and use its gradient
for i in range(grid_size):
    for j in range(grid_size):
        x_point = X_grid[i, j]
        y_point = Y_grid[i, j]
        
        # Find closest data point
        distances = (x0_comp1 - x_point)**2 + (x0_comp2 - y_point)**2
        closest_idx = np.argmin(distances)
        
        # Get gradient at that point
        dx = dv_values[closest_idx, 0]
        dy = dv_values[closest_idx, 1]
        
        # Normalize for better visualization
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude > 0:  # Avoid division by zero
            # Scale inversely with magnitude for better visualization
            scale = 0.15 * (1.0 / (1.0 + 0.5 * magnitude))
            plt.arrow(x_point, y_point, dx*scale, dy*scale, 
                     head_width=0.05, head_length=0.08, fc='red', ec='red', alpha=0.7)

# Add colorbar and labels
cbar = plt.colorbar(scatter)
cbar.set_label('Value Function (V)')
plt.xlabel('x₀[0]')
plt.ylabel('x₀[1]')
plt.title('Value Function with Gradient Vectors')
plt.grid(True, alpha=0.3)

# Save as high-resolution image
plt.savefig('VDP_gradient_vectors_combined_data.png', dpi=300, bbox_inches='tight')
plt.show()

ini = np.load("VDP_beta_0.1_failed_ini_30x30.npy")
print(ini)

# Plot failed initial conditions
plt.figure(figsize=(10, 8))
# Check if ini is 1D or 2D
if len(ini.shape) == 1:
    # If 1D, reshape to ensure it's treated as a collection of points
    ini_points = ini.reshape(-1, 2)
else:
    ini_points = ini

# Plot the failed initial condition points
plt.scatter(ini_points[:, 0], ini_points[:, 1], color='red', marker='x', s=100, label='Failed Initial Conditions')

# Add the existing valid data points for comparison
plt.scatter(x0_comp1, x0_comp2, c=v_values, cmap='viridis', 
           s=30, alpha=0.3, label='Valid Data Points')

plt.colorbar(label='Value Function (V)')
plt.xlabel('x₀[0]')
plt.ylabel('x₀[1]')
plt.title('Failed Initial Conditions Compared to Valid Data')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('VDP_failed_initial_conditions.png', dpi=300, bbox_inches='tight')
plt.show()

# # Previous 3D plot code is commented out
# # Create a 3D scatter plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# 
# # Create scatter plot
# scatter = ax.scatter(x0_comp1, x0_comp2, v_values, c=v_values, cmap='viridis', 
#                     s=50, alpha=0.8)
# 
# # Add a color bar
# cbar = fig.colorbar(scatter, ax=ax, shrink=0.7)
# cbar.set_label('Value Function (V)')
# # Set labels and title
# ax.set_xlabel('x₀[0]')
# ax.set_ylabel('x₀[1]')
# ax.set_zlabel('Value Function (V)')
# ax.set_title('Combined VDP Data (β=3): Initial Conditions vs Value Function')
# # Adjust view angle
# ax.view_init(elev=30, azim=45)
# plt.tight_layout()
# 
# # Save as regular image
# plt.savefig('VDP_legendre_plot.png', dpi=300)
# 
# # Save the figure as a pickle for interactive use later
# pickle_file = 'VDP_legendre_plot.pickle'
# with open(pickle_file, 'wb') as f:
#     pickle.dump(fig, f)
# print(f"Saved interactive plot to {pickle_file}")
# 
# plt.show()

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



