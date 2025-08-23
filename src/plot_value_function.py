#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modular plotting functions for VDP value function visualization
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from loguru import logger

def load_training_history(history_file):
    """
    Load training history from pickle file.
    
    Args:
        history_file: Path to the pickle file
        
    Returns:
        Dictionary containing training history
    """
    logger.info(f"Loading training history from {history_file}...")
    with open(history_file, "rb") as f:
        history = pickle.load(f)
    
    logger.info(f"History contains {len(history['weights'])} iterations")
    logger.info(f"Hyperparameters: {history['hyperparameters']}")
    
    return history

def get_best_iteration(history):
    """
    Get the best iteration based on test metrics.
    
    Args:
        history: Training history dictionary
        
    Returns:
        Index of best iteration
    """
    test_metrics = history['test_metrics']
    best_iteration = np.argmin([m[0] if isinstance(m, (list, np.ndarray)) else float('inf') for m in test_metrics])
    logger.info(f"Best iteration: {best_iteration} with {history['neuron_count'][best_iteration]} neurons")
    return best_iteration

def create_value_function(weights, biases, activation='relu', power=2):
    """
    Create a value function from weights and biases.
    
    Args:
        weights: Neural network weights
        biases: Neural network biases
        activation: Activation function ('relu', 'tanh', 'sigmoid')
        power: Power for activation function
        
    Returns:
        Function that computes the value function
    """
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
    
    return value_function

def plot_value_function_3d(weights, biases, activation='relu', power=2, 
                          x_range=(-3.0, 3.0), y_range=(-3.0, 3.0), 
                          resolution=100, title="Value Function", 
                          save_path=None, show_plot=True):
    """
    Create 3D surface plot of the value function.
    
    Args:
        weights: Neural network weights
        biases: Neural network biases
        activation: Activation function
        power: Power for activation function
        x_range: Tuple of (x_min, x_max)
        y_range: Tuple of (y_min, y_max)
        resolution: Grid resolution
        title: Plot title
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
        
    Returns:
        matplotlib figure and axis objects
    """
    # Create value function
    value_func = create_value_function(weights, biases, activation, power)
    
    # Create grid
    x_min, x_max = x_range
    y_min, y_max = y_range
    x1_grid = np.linspace(x_min, x_max, resolution)
    x2_grid = np.linspace(y_min, y_max, resolution)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    
    # Prepare grid points for function evaluation
    grid_points = np.column_stack((X1.flatten(), X2.flatten()))
    
    # Compute value function for all grid points
    V_values = value_func(grid_points)
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
    ax.set_title(title)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"3D plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig, ax

def plot_value_function_contour(weights, biases, activation='relu', power=2,
                               x_range=(-3.0, 3.0), y_range=(-3.0, 3.0),
                               resolution=100, title="Value Function Contour",
                               save_path=None, show_plot=True):
    """
    Create 2D contour plot of the value function.
    
    Args:
        weights: Neural network weights
        biases: Neural network biases
        activation: Activation function
        power: Power for activation function
        x_range: Tuple of (x_min, x_max)
        y_range: Tuple of (y_min, y_max)
        resolution: Grid resolution
        title: Plot title
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
        
    Returns:
        matplotlib figure and axis objects
    """
    # Create value function
    value_func = create_value_function(weights, biases, activation, power)
    
    # Create grid
    x_min, x_max = x_range
    y_min, y_max = y_range
    x1_grid = np.linspace(x_min, x_max, resolution)
    x2_grid = np.linspace(y_min, y_max, resolution)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    
    # Prepare grid points for function evaluation
    grid_points = np.column_stack((X1.flatten(), X2.flatten()))
    
    # Compute value function for all grid points
    V_values = value_func(grid_points)
    V_grid = V_values.reshape(resolution, resolution)
    
    # Create contour plot
    fig, ax = plt.subplots(figsize=(12, 10))
    contour = ax.contourf(X1, X2, V_grid, 50, cmap='viridis')
    plt.colorbar(contour, label='Value Function V(x)')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Contour plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig, ax

def plot_training_results(training_logger, iteration=None, show_plots=True, output_dir="../data_result/visualizations"):
    """
    Convenience function for plotting results directly from notebook after training.
    
    Args:
        training_logger: TrainingLogger object from training
        iteration: Specific iteration to plot (None for best iteration)
        show_plots: Whether to display plots inline in notebook
        output_dir: Directory to save plots
        
    Returns:
        Dictionary with plotting results
    """
    logger.info("=== Plotting Training Results ===")
    
    # Handle TrainingLogger objects
    history = training_logger.history if hasattr(training_logger, 'history') else training_logger
    
    # Get best iteration if not specified
    if iteration is None:
        best_iteration = get_best_iteration(history)
    else:
        best_iteration = iteration
    
    # Extract weights and biases
    weights = history['weights'][best_iteration]
    biases = history['biases'][best_iteration]
    activation = history['hyperparameters'].get('activation', 'relu')
    power = history['hyperparameters'].get('power', 2)
    
    logger.info(f"Plotting iteration {best_iteration}:")
    logger.info(f"Weights shape: {weights.shape}, Biases shape: {biases.shape}")
    logger.info(f"Activation: {activation}, Power: {power}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create title
    test_error = history['test_metrics'][best_iteration][0] if history['test_metrics'][best_iteration] else 0
    title = f"Value Function at Iteration #{best_iteration}\n{history['neuron_count'][best_iteration]} neurons, Test Error: {test_error:.6f}"
    
    # Plot 3D surface
    plot_3d_path = os.path.join(output_dir, f"value_function_3d_iter_{best_iteration}.png")
    fig_3d, ax_3d = plot_value_function_3d(weights, biases, activation, power, title=title, save_path=plot_3d_path, show_plot=show_plots)
    
    # Plot contour
    contour_path = os.path.join(output_dir, f"value_function_contour_iter_{best_iteration}.png")
    fig_contour, ax_contour = plot_value_function_contour(weights, biases, activation, power, title=title, save_path=contour_path, show_plot=show_plots)
    
    # Create value function for return
    value_func = create_value_function(weights, biases, activation, power)
    
    results = {
        'best_iteration': best_iteration,
        'value_function': value_func,
        'plot_paths': {
            '3d': plot_3d_path,
            'contour': contour_path
        },
        'figures': {
            '3d': fig_3d,
            'contour': fig_contour
        },
        'total_iterations': len(history['weights'])
    }
    
    logger.info(f"Plotting Summary:")
    logger.info(f"- Best iteration: {best_iteration}")
    logger.info(f"- Total iterations: {len(history['weights'])}")
    logger.info(f"- 3D plot saved: {plot_3d_path}")
    logger.info(f"- Contour plot saved: {contour_path}")
    
    return results

# Main execution (for standalone use)
if __name__ == "__main__":
    # Path to the training history pickle file
    history_file = "data_result/weights/training_history_26.pkl"
    
    # Load and plot
    history = load_training_history(history_file)
    
    # Create a mock training logger object for testing
    class MockTrainingLogger:
        def __init__(self, history):
            self.history = history
    
    mock_logger = MockTrainingLogger(history)
    results = plot_training_results(mock_logger)
    
    # Show plots
    plt.show()