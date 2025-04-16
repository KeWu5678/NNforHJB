#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:32:26 2024

@author: chaoruiz
"""

import os
os.environ["DDE_BACKEND"] = "pytorch"

import openloop_legendre as op
import numpy as np
import numpy.polynomial.chebyshev as cheb
import utils 
import deepxde as dde
dde.config.backend = "pytorch"
print(f"Using backend: {dde.backend.backend_name}")

from network_torch import network
from greedy_insertion import insertion


# Define structured array dtype
dtype = [
    ('x', '2float64'),  # 2D float array for the first element
    ('dv', '2float64'), # 2D float array for the second element
    ('v', 'float64')    # 1D float for the third element
]

# Create structured array


"""=====================THE ODE======================"""
beta = 0.1

def VDP(t, y, u):
    """Define the Boundary value problem with control parameter u"""
    return np.vstack([
        y[1],
        -y[0] + y[1] * (1 - y[0] ** 2) + u(t),
        - 2 * y[0] + y[3] * (2 * y[0] * y[1] + 1),
        - 2 * y[1] - y[2] - y[3] * (1 -y[0] ** 2)
    ])

def bc(ya, yb):
    """Boundary conditions"""
    return np.array([
        ya[0] - ini[0],
        ya[1] - ini[1],
        yb[2],
        yb[3]
    ])

def gradient(u, p):
    if len(p) != len(u):
        raise ValueError("p and u must have the same length")
    else:
        n = len(p)
        grad = np.zeros(n)
    for i in range(n):
        grad[i] = p[i] + 2 * beta * u[i]
    return grad

def V(grid, u, y1, y2):
    return utils.L2(grid, u) * beta + 0.5 * (utils.L2(grid, y1) + utils.L2(grid, y2))

def gen_bc(ini_val):
    """Generate boundary conditions based on initial values"""
    def bc_func(ya, yb):
        return np.array([
            ya[0] - ini_val[0],
            ya[1] - ini_val[1],
            yb[2],
            yb[3]
        ])
    return bc_func


if __name__ == "__main__":
    """=====================DATA GENERATION======================"""
    grid = np.linspace(0, 3, 1000)
    guess = np.ones((4, grid.size))
    tol = 1e-5
    max_it = 500
    
    # Create a 30x30 grid of initial conditions
    # Define the range for each dimension
    x1_min, x1_max = -3.0, 3.0  # Range for first component
    x2_min, x2_max = -3.0, 3.0  # Range for second component
    
    # Create 1D arrays for each dimension
    x1_values = np.linspace(x1_min, x1_max, 50)
    x2_values = np.linspace(x2_min, x2_max, 50)
    
    # Create a meshgrid
    X1, X2 = np.meshgrid(x1_values, x2_values)
    
    # Reshape the meshgrid to get a list of all combinations
    x0_values = np.column_stack((X1.flatten(), X2.flatten()))
    
    # Total number of grid points
    N = len(x0_values)
    print(f"Created a 30x30 grid with {N} points")
    
    # Initialize dataset
    dataset = np.zeros(N, dtype=dtype)
    
    # Track failed initial conditions
    failed_ini = []

    print(f"N = {N}")
    for i in range(N):
        # Get initial condition from the grid
        ini = x0_values[i]
        
        # Generate boundary conditions
        bc_func = gen_bc(ini)
        
        print(f"i = {i}")
        print(f"ini = {ini}")
        dv, v = op.OpenLoopOptimizer(VDP, bc_func, V, gradient, grid, guess, tol, max_it).optimize()
        if dv is not None and not np.isnan(v):
            dataset[i] = (ini, dv, v)
        else:
            # Store failed initial condition
            failed_ini.append(ini)
            print(f"Failed to converge for ini = {ini}")
        
    # Use a filename that indicates grid sampling
    output_file = "VDP_beta_3_grid_40x40.npy"
    
    # Save failed initial conditions
    failed_output_file = "VDP_beta_3_failed_ini_40x40.npy"
    if failed_ini:
        failed_ini = np.array(failed_ini)
        print(f"Saving {len(failed_ini)} failed initial conditions to {failed_output_file}")
        np.save(failed_output_file, failed_ini)
    else:
        print("All initial conditions converged successfully")
    
    print(f"Saving results to {output_file}")
    np.save(output_file, dataset)


    # """=====================Greedy Insertion and Training======================"""
    # path = 'data/VDP_beta_3_grid_30x30.npy'# Initialize the weights
    # power = 2.5
    # gamma = 0.01
    # M = 40
    # alpha = 0.1
    # dataset = np.load(path)
    
    # # Data inspection
    # print(f"Loaded dataset from {path}")
    # print(f"Dataset shape: {dataset.shape}")
    # print(f"Dataset dtype: {dataset.dtype}")
    
    # # Check for NaN or empty values
    # nan_count = 0
    # valid_count = 0
    # for i, item in enumerate(dataset):
    #     if np.isnan(item['v']) or np.isnan(item['x']).any() or np.isnan(item['dv']).any():
    #         nan_count += 1
    #         if nan_count < 5:  # Only print first few examples
    #             print(f"NaN values found at index {i}: {item}")
    #     else:
    #         valid_count += 1
            
    # print(f"Valid data points: {valid_count}")
    # print(f"Data points with NaN values: {nan_count}")
    
    # # If we have valid data, print a few examples
    # if valid_count > 0:
    #     print("\nFirst 3 valid data points:")
    #     count = 0
    #     for item in dataset:
    #         if not np.isnan(item['v']) and not np.isnan(item['x']).any() and not np.isnan(item['dv']).any():
    #             print(f"x: {item['x']}, dv: {item['dv']}, v: {item['v']}")
    #             count += 1
    #             if count >= 3:
    #                 break
    
    # # Convert to dictionary format
    # data_dict = {
    #     'x': np.array([item[0] for item in dataset]),
    #     'dv': np.array([item[1] for item in dataset]),
    #     'v': np.array([item[2] for item in dataset])
    # }
    
    # # Check processed data
    # print(f"\nProcessed data shapes:")
    # print(f"x shape: {data_dict['x'].shape}")
    # print(f"dv shape: {data_dict['dv'].shape}")
    # print(f"v shape: {data_dict['v'].shape}")
    
    # # Initialize the model with zero weights
    # model, _, _ = network(data_dict, power, ('phi', gamma, 0)) 
    # weight, bias = insertion(data_dict, model, M)
    # print("Initialization done")
    # print(f"Initial weights shape: {weight.shape}, bias shape: {bias.shape}")
    
    # # Training the model
    # for i in range(30):   
    #     print(f"\n----- Iteration {i} -----")
    #     weight_temp, bias_temp = insertion(data_dict, model, M)
    #     # Convert PyTorch tensors to NumPy arrays if needed
    #     if hasattr(weight_temp, 'numpy'):
    #         print("Converting weight_temp from PyTorch tensor to NumPy array")
    #         weight_temp = weight_temp.numpy()
        
    #     if hasattr(bias_temp, 'numpy'):
    #         print("Converting bias_temp from PyTorch tensor to NumPy array")
    #         bias_temp = bias_temp.numpy()
        
    #     print(f"Iteration {i} - weight_temp shape: {weight_temp.shape}, bias_temp shape: {bias_temp.shape}")
    #     print(f"Iteration {i} - weight shape: {weight.shape}, bias shape: {bias.shape}")
   
    #     weight = np.concatenate((weight, weight_temp), axis=0)
    #     bias = np.concatenate((bias, bias_temp), axis=0)
    #     model, weight, bias = network(data_dict, power, ('phi', gamma, 0.5), inner_weights = weight, inner_bias = bias)
    #     print(f"After concatenation - weight shape: {weight.shape}, bias shape: {bias.shape}")
        
    #     # Check if model has NaN values
    #     test_pred = model.predict(data_dict['x'][:1])
    #     if np.isnan(test_pred).any():
    #         print("WARNING: Model contains NaN values. Stopping training.")
    #         break

    
    

    



