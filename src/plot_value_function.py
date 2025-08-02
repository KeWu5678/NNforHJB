#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the value function V(x) at the best iteration from training history
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import torch

# Path to the training history pickle file
history_file = "data_result/weights/training_history_26.pkl"

# Create output directory for plots
output_dir = "data_result/visualizations"
os.makedirs(output_dir, exist_ok=True)

# Load the training history
print(f"Loading training history from {history_file}...")
with open(history_file, "rb") as f:
    history = pickle.load(f)

# Print basic information about the history
print(f"History contains {len(history['weights'])} iterations")
print(f"Hyperparameters: {history['hyperparameters']}")

# Select the best iteration based on test metrics
test_metrics = history['test_metrics']
best_iteration = np.argmin([m[0] if isinstance(m, (list, np.ndarray)) else float('inf') for m in test_metrics])
print(f"Best iteration: {best_iteration} with {history['neuron_count'][best_iteration]} neurons")

# Extract weights and biases for this iteration
weights = history['weights'][best_iteration]
biases = history['biases'][best_iteration]
activation = history['hyperparameters'].get('activation', 'relu')
power = history['hyperparameters'].get('power', 2)

print(f"Weights shape: {weights.shape}, Biases shape: {biases.shape}")
print(f"Activation: {activation}, Power: {power}")

# Define a function to compute the network output (value function)
def value_function(x):
    """
    Compute the value function for given input points
    
    Args:
        x: Input points of shape (n_points, 2)
    
    Returns:
        Value function output of shape (n_points,)
    """
    # Convert inputs to appropriate numpy arrays
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    
    # Ensure x is a 2D array
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    # Pre-activation: wx + b for each neuron
    # Shape: (n_points, n_neurons)
    pre_activation = np.dot(x, weights.T) + biases
    
    # Apply activation function
    if activation == 'relu':
        activations = np.maximum(0, pre_activation) ** power
    elif activation == 'tanh':
        activations = np.tanh(pre_activation) ** power
    elif activation == 'sigmoid':
        activations = (1 / (1 + np.exp(-pre_activation))) ** power
    else:
        # Default to identity
        activations = pre_activation ** power
    
    # Sum all neuron outputs (weights for output layer are all 1)
    outputs = np.sum(activations, axis=1)
    
    return outputs

# Create a grid of points for visualization
x_min, x_max = -3.0, 3.0
y_min, y_max = -3.0, 3.0
resolution = 100
x1_grid = np.linspace(x_min, x_max, resolution)
x2_grid = np.linspace(y_min, y_max, resolution)
X1, X2 = np.meshgrid(x1_grid, x2_grid)

# Prepare grid points for function evaluation
grid_points = np.column_stack((X1.flatten(), X2.flatten()))

# Compute value function for all grid points
V_values = value_function(grid_points)
V_grid = V_values.reshape(resolution, resolution)

# Create 3D surface plot
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot surface
surf = ax.plot_surface(X1, X2, V_grid, cmap=cm.viridis, alpha=0.8, 
                      linewidth=0, antialiased=True)

# Add a color bar
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('Value Function V(x)')

# Set labels and title
ax.set_xlabel('x₁')
ax.set_ylabel('x₂')
ax.set_zlabel('V(x)')
ax.set_title(f'Value Function at Best Iteration (#{best_iteration})\n{history["neuron_count"][best_iteration]} neurons, Test Error: {test_metrics[best_iteration][0]:.6f}')

# Save the figure
plot_file = os.path.join(output_dir, f"value_function_best_iter_{best_iteration}.png")
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to {plot_file}")

# Also create a 2D contour plot
plt.figure(figsize=(12, 10))
contour = plt.contourf(X1, X2, V_grid, 50, cmap='viridis')
plt.colorbar(contour, label='Value Function V(x)')
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title(f'Value Function Contour at Best Iteration (#{best_iteration})\n{history["neuron_count"][best_iteration]} neurons, Test Error: {test_metrics[best_iteration][0]:.6f}')
plt.grid(True, alpha=0.3)

# Save the contour plot
contour_file = os.path.join(output_dir, f"value_function_contour_best_iter_{best_iteration}.png")
plt.savefig(contour_file, dpi=300, bbox_inches='tight')
print(f"Contour plot saved to {contour_file}")

# Show the plots
plt.show() 