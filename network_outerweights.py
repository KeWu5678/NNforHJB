#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:30:52 2024

@author: chaoruiz
"""

import os
# Set environment variable to use PyTorch as backend
os.environ["DDE_BACKEND"] = "pytorch"

# Import DeepXDE after setting the environment variable
import deepxde as dde
# Explicitly set the backend to pytorch
dde.config.backend = "pytorch"
print(f"Using backend: {dde.backend.backend_name}")

import numpy as np
from scipy.spatial import KDTree
import torch
from utils import h1_error


def network_outerweights(data, activation,  power, regularization, loss_weights = (1.0, 1.0), inner_weights = None, inner_bias = None, outer_weights = None):
    """
    args:
        inner_weights: (N, 2) - matrix with N rows (number of neurons) and 2 columns (input features)
        inner_bias: (N,) - vector with N bias values, one per neuron
        path: tuple of (x, v, dv)
    return:
        model: the trained model (includes model.losshistory with all loss/metrics)
        weight: the weights of the inner neurons
        bias: the bias of the inner neurons
        output_weight: the weights of the output layer
    """
    training_percentage = 0.9

    # Get the raw data
    ob_x, ob_v, ob_dv = data["x"], data["v"], data["dv"]
    
    # Split data into training and validation sets - no need for permutation as indices are already permuted
    num_samples = len(ob_x)
    split_idx = int(num_samples * training_percentage)
    train_indices = np.arange(split_idx)
    valid_indices = np.arange(split_idx, num_samples)
    
    # Create training and validation datasets
    train_x = ob_x[train_indices].astype(np.float32)
    train_v = ob_v[train_indices].astype(np.float32)
    train_dv = ob_dv[train_indices].astype(np.float32)
    
    valid_x = ob_x[valid_indices].astype(np.float32)
    valid_v = ob_v[valid_indices].astype(np.float32)
    valid_dv = ob_dv[valid_indices].astype(np.float32)
    
    print(f"Training set: {len(train_x)} samples, Validation set: {len(valid_x)} samples")

    # Customized to dimension
    def VdV(x, y, ex):
        y = y[:, 0:1]
        v1 = ex[:, 0:1]
        v2 = ex[:, 1:2]
        
        # Get the target value at point x - convert to tensor
        v_true = value_function(x)
        v_true_tensor = dde.backend.as_tensor(v_true)
        
        # Calculate the derivatives
        dy_dx1 = dde.grad.jacobian(y, x, i=0, j = 0)
        dy_dx2 = dde.grad.jacobian(y, x, i=0, j = 1)
        
        return [
            y - v_true_tensor,  # Match function value directly
            v1 - dy_dx1,        # Match first derivative 
            v2 - dy_dx2,        # Match second derivative
        ]

    geom = dde.geometry.Rectangle([-3, -3], [3, 3])

    def gradient_function(x):
        """Return the auxiliary variables (dV/dx values) for the given points x."""
        # Create KDTree for efficient lookup using the full dataset
        if not hasattr(gradient_function, 'kdtree'):
            gradient_function.kdtree = KDTree(ob_x)
        
        # Handle PyTorch tensors by detaching and converting to numpy
        if hasattr(x, 'detach') and callable(getattr(x, 'detach')):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.array(x)
        
        # Find indices of closest points
        distances, indices = gradient_function.kdtree.query(x_np, k=1)
        # Convert to float32 to match model weights
        return ob_dv[indices].astype(np.float32)

    def value_function(x):
        """Return the value function V at points x."""
        # Use KDTree for efficient and robust matching using the full dataset
        if not hasattr(value_function, 'kdtree'):
            value_function.kdtree = KDTree(ob_x)
        
        # Handle PyTorch tensors by detaching and converting to numpy
        if hasattr(x, 'detach') and callable(getattr(x, 'detach')):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.array(x)
        
        # Find indices of closest points
        _, indices = value_function.kdtree.query(x_np, k=1)
        
        # Return corresponding V values, convert to float32
        return ob_v[indices].reshape(-1, 1).astype(np.float32)  # Make sure it's a column vector

    # Create PDE data with proper test points
    data = dde.data.PDE(
        geom,
        VdV,
        [],
        num_domain=0,
        num_boundary=0,
        anchors=train_x,  # Use only training data as anchors
        auxiliary_var_function=gradient_function, # here is the "ex" in the PDE
        solution=value_function,
        num_test=len(valid_x)  # Generate test points equal to validation set size
    )
    
    # Add validation points as test anchors
    if hasattr(data, 'test_x') and hasattr(data, 'test_y'):
        data.test_x = valid_x
        data.test_y = valid_v.reshape(-1, 1).astype(np.float32)
        if data.auxiliary_var_fn:
            data.test_aux_vars = data.auxiliary_var_fn(data.test_x).astype(np.float32)
    
    if inner_weights is None:
        # if no weights are given, use default 2 neurons
        n = 0
    else:
        # Number of neurons is the first dimension for PyTorch
        n = inner_weights.shape[0]
        print(f"Creating network with {n} neurons")
    
    # Print regularization info for debugging
    if isinstance(regularization, tuple) and regularization[0] == "phi":
        print(f"Using phi regularization with gamma={regularization[1]}, alpha={regularization[2]}")
        # The phi regularization will be applied only to output layer in model.py
    
    # Layer sizes: input dimension, hidden layer size, output dimension
    net = dde.nn.SHALLOW_OUTERWEIGHTS(
        [2] + [n] + [1], activation, "zeros", p = power, inner_weights = inner_weights, 
        inner_bias = inner_bias, outer_weights = outer_weights, regularization = regularization
        )
    model = dde.Model(data, net)
    
    # Define the H1 error metric using a lambda to pass the required parameters
    h1_metric = lambda y_true, y_pred: h1_error(y_true, y_pred, data.test_x, model)
    
    # Set L-BFGS optimizer options before compiling
    dde.optimizers.config.set_LBFGS_options(
        maxiter=50000,     # Increased maximum iterations
        ftol=1e-13,        # Lower function tolerance (more strict)
        gtol=1e-13,        # Lower gradient tolerance (more strict) 
        maxcor=100,        # Increased history size for approximating the Hessian
        maxfun=50000,      # Maximum function evaluations
        maxls=50           # Maximum line search steps
    )
    
    # Add both "mean squared error" and our custom H1 error metric
    model.compile("L-BFGS", loss="mse", metrics=["mean squared error", h1_metric], 
                 loss_weights=[loss_weights[0], loss_weights[1], loss_weights[1]])  # Value matching gets higher weight

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(current_dir, "train_history")
    os.makedirs(model_save_path, exist_ok=True)
    print(f"Training model, saving to {model_save_path}")
    
    # Train the model
    # losshistory and train_state are saved as model.losshistory and model.train_state
    model.train(iterations=20000, batch_size=810, display_every=1000, model_save_path=model_save_path)
    
    # All training and testing errors are available in model.losshistory
    # - model.losshistory.loss_train: training loss
    # - model.losshistory.loss_test: testing loss
    # - model.losshistory.metrics_test: test metrics (MSE between predictions and targets)
    
    # Detach the tensors to remove them from the computation graph before returning
    weight, bias = model.net.get_hidden_params()
    outer_weight = model.net.get_output_params()
    return model, weight.numpy(), bias.numpy(), outer_weight.numpy()
    
if __name__ == "__main__":
    data = np.load("data_result/raw_data/VDP_beta_0.1_grid_30x30.npy")
    weights = np.random.randn(73, 2)
    bias = np.random.randn(73)
    outer_weights = np.random.randn(1, 73)
    regularization = ('l1', 0.0) #('phi', 0.01, 0.01)
    model, weight, bias, output_weight = network_outerweights(data, "relu", 2.0, regularization, loss_weights = (1.0, 1.0), inner_weights=weights, inner_bias=bias, outer_weights=outer_weights)


