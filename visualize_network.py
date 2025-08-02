#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization script for neural network weights from training history
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import math

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
print(f"Neuron counts per iteration: {history['neuron_count']}")

# Select which iteration to visualize
# Let's find the iteration with the best test metrics
test_metrics = history['test_metrics']
best_iteration = np.argmin([m[0] if isinstance(m, (list, np.ndarray)) else float('inf') for m in test_metrics])
print(f"Best iteration: {best_iteration} with {history['neuron_count'][best_iteration]} neurons")

# Function to visualize a specific iteration
def visualize_iteration(iteration_idx):
    print(f"\nVisualizing iteration {iteration_idx} with {history['neuron_count'][iteration_idx]} neurons")
    
    # Extract weights and biases for this iteration
    weights = history['weights'][iteration_idx]
    biases = history['biases'][iteration_idx]
    
    # Get activation function from hyperparameters
    activation = history['hyperparameters'].get('activation', 'relu')
    power = history['hyperparameters'].get('power', 2)
    
    print(f"Weights shape: {weights.shape}, Biases shape: {biases.shape}")
    print(f"Activation: {activation}, Power: {power}")
    
    # Visualize weight distributions
    plt.figure(figsize=(12, 9))
    
    # Plot weights distribution
    plt.subplot(2, 2, 1)
    plt.hist(weights.flatten(), bins=30, color='blue', alpha=0.7)
    plt.title(f'Weight Distribution (Iteration {iteration_idx})')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Plot bias distribution
    plt.subplot(2, 2, 2)
    plt.hist(biases, bins=20, color='green', alpha=0.7)
    plt.title(f'Bias Distribution (Iteration {iteration_idx})')
    plt.xlabel('Bias Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # If weights are 2D, visualize in weight space
    if weights.shape[1] == 2:
        # 2D scatter plot of weights
        plt.subplot(2, 2, 3)
        plt.scatter(weights[:, 0], weights[:, 1], c=biases, cmap='viridis', alpha=0.8, s=30)
        plt.colorbar(label='Bias Value')
        plt.title(f'2D Weight Space (Iteration {iteration_idx})')
        plt.xlabel('Weight 1')
        plt.ylabel('Weight 2')
        plt.grid(True, alpha=0.3)
        
        # Polar scatter plot of weights
        plt.subplot(2, 2, 4)
        r = np.sqrt(weights[:, 0]**2 + weights[:, 1]**2)
        theta = np.arctan2(weights[:, 1], weights[:, 0])
        sc = plt.scatter(theta, r, c=biases, cmap='viridis', alpha=0.8, s=30)
        plt.colorbar(sc, label='Bias Value')
        plt.title(f'Polar Weight Space (Iteration {iteration_idx})')
        plt.xlabel('Angle θ')
        plt.ylabel('Radius r')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/weight_distribution_iter_{iteration_idx}.png", dpi=300)
    
    # For 2D weights, create a 3D visualization of the neurons
    if weights.shape[1] == 2:
        # Create a grid to visualize neuron activations
        x_min, x_max = -3, 3
        y_min, y_max = -3, 3
        grid_size = 100
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Initialize 3D plot showing all neuron activations
        fig = plt.figure(figsize=(15, 12))
        
        # Number of neurons to visualize (limit to avoid overcrowding)
        num_neurons = min(9, weights.shape[0])
        
        # Create a 3x3 grid of 3D plots, each showing one neuron's activation
        for i in range(num_neurons):
            # Get this neuron's weights and bias
            w1, w2 = weights[i]
            b = biases[i]
            
            # Calculate the activation for this neuron
            Z = np.zeros_like(X)
            for ix in range(grid_size):
                for iy in range(grid_size):
                    x_val = X[ix, iy]
                    y_val = Y[ix, iy]
                    # Pre-activation: wx + b
                    pre_activation = w1 * x_val + w2 * y_val + b
                    
                    # Apply activation function
                    if activation == 'relu':
                        Z[ix, iy] = max(0, pre_activation) ** power
                    elif activation == 'tanh':
                        Z[ix, iy] = np.tanh(pre_activation) ** power
                    elif activation == 'sigmoid':
                        Z[ix, iy] = (1 / (1 + np.exp(-pre_activation))) ** power
                    else:
                        # Default to identity
                        Z[ix, iy] = pre_activation ** power
            
            # Create 3D subplot
            ax = fig.add_subplot(3, 3, i+1, projection='3d')
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
            
            # Set labels and title
            ax.set_xlabel('x₁')
            ax.set_ylabel('x₂')
            ax.set_zlabel(f'{activation}(wx+b)^{power}')
            ax.set_title(f'Neuron {i+1}: w=[{w1:.2f}, {w2:.2f}], b={b:.2f}')
            
            # Add a colorbar
            fig.colorbar(surf, ax=ax, shrink=0.5)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/neuron_activations_iter_{iteration_idx}.png", dpi=300)
    
    # Visualize error evolution over iterations
    plt.figure(figsize=(12, 6))
    
    # Get train and test errors up to this iteration
    train_errors = [loss[0] if isinstance(loss, (list, np.ndarray)) else loss 
                    for loss in history['train_loss'][:iteration_idx+1]]
    test_errors = [metrics[0] if isinstance(metrics, (list, np.ndarray)) else metrics 
                   for metrics in history['test_metrics'][:iteration_idx+1]]
    
    # Plot errors
    plt.semilogy(range(iteration_idx+1), train_errors, 'b-', label='Training Loss')
    plt.semilogy(range(iteration_idx+1), test_errors, 'r-', label='Test Error')
    
    # Add neuron counts
    neuron_counts = history['neuron_count'][:iteration_idx+1]
    plt.plot(range(iteration_idx+1), neuron_counts, 'g--', label='Neuron Count')
    
    plt.xlabel('Iteration')
    plt.ylabel('Value (log scale for errors)')
    plt.title('Error Evolution and Neuron Count Over Iterations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/error_evolution_iter_{iteration_idx}.png", dpi=300)
    
    # Visualize all neurons in a 2D grid (useful for larger networks)
    if weights.shape[1] == 2:
        # Determine grid size for visualization
        grid_cols = min(10, int(math.ceil(math.sqrt(weights.shape[0]))))
        grid_rows = math.ceil(weights.shape[0] / grid_cols)
        
        plt.figure(figsize=(grid_cols*2, grid_rows*2))
        
        # Normalize biases for coloring
        norm = Normalize(vmin=np.min(biases), vmax=np.max(biases))
        
        for i, (w, b) in enumerate(zip(weights, biases)):
            # Calculate position in the grid
            row = i // grid_cols
            col = i % grid_cols
            
            # Create subplot
            ax = plt.subplot(grid_rows, grid_cols, i+1)
            
            # Plot the weight vector as an arrow from origin
            color = cm.viridis(norm(b))
            ax.arrow(0, 0, w[0], w[1], head_width=0.1, head_length=0.1, fc=color, ec=color)
            
            # Set limits and gridlines
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.grid(True, alpha=0.3)
            
            # Add minimal labels
            if row == grid_rows-1:
                ax.set_xlabel('w₁')
            if col == 0:
                ax.set_ylabel('w₂')
            
            ax.set_title(f'Neuron {i+1}\nb={b:.2f}', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/all_neurons_weights_iter_{iteration_idx}.png", dpi=300)
    
    plt.close('all')
    print(f"Visualizations saved to {output_dir}")

# Visualize the best iteration
visualize_iteration(best_iteration)

# Visualize the last iteration
visualize_iteration(len(history['weights'])-1)

print("Visualization complete!") 