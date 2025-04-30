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
import datetime


from network import network
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
    # """=====================DATA GENERATION======================"""
    # grid = np.linspace(0, 3, 1000)
    # guess = np.ones((4, grid.size))
    # tol = 1e-5
    # max_it = 500
    
    # # Create a 30x30 grid of initial conditions
    # # Define the range for each dimension
    # x1_min, x1_max = -3.0, 3.0  # Range for first component
    # x2_min, x2_max = -3.0, 3.0  # Range for second component
    
    # # Create 1D arrays for each dimension
    # x1_values = np.linspace(x1_min, x1_max, 30)
    # x2_values = np.linspace(x2_min, x2_max, 30)
    
    # # Create a meshgrid
    # X1, X2 = np.meshgrid(x1_values, x2_values)
    
    # # Reshape the meshgrid to get a list of all combinations
    # x0_values = np.column_stack((X1.flatten(), X2.flatten()))
    
    # # Total number of grid points
    # N = len(x0_values)
    # print(f"Created a 30x30 grid with {N} points")
    
    # # Initialize dataset
    # dataset = np.zeros(N, dtype=dtype)
    
    # # Track failed initial conditions
    # failed_ini = []

    # print(f"N = {N}")
    # for i in range(N):
    #     # Get initial condition from the grid
    #     ini = x0_values[i]
        
    #     # Generate boundary conditions
    #     bc_func = gen_bc(ini)
        
    #     print(f"i = {i}")
    #     print(f"ini = {ini}")
    #     dv, v = op.OpenLoopOptimizer(VDP, bc_func, V, gradient, grid, guess, tol, max_it).optimize()
    #     if dv is not None and not np.isnan(v):
    #         dataset[i] = (ini, dv, v)
    #     else:
    #         # Store failed initial condition
    #         failed_ini.append(ini)
    #         print(f"Failed to converge for ini = {ini}")
        
    # # Use a filename that indicates grid sampling
    # output_file = "VDP_beta_3_grid_40x40.npy"
    
    # # Save failed initial conditions
    # failed_output_file = "VDP_beta_3_failed_ini_40x40.npy"
    # if failed_ini:
    #     failed_ini = np.array(failed_ini)
    #     print(f"Saving {len(failed_ini)} failed initial conditions to {failed_output_file}")
    #     np.save(failed_output_file, failed_ini)
    # else:
    #     print("All initial conditions converged successfully")
    
    # print(f"Saving results to {output_file}")
    # np.save(output_file, dataset)


    """=====================Greedy Insertion and Training======================"""
    path = 'data_result/raw_data/VDP_beta_0.1_grid_30x30.npy'# Initialize the weights
    dataset = np.load(path)
    power = 2
    gamma = 0.01
    M = 50 # number greedy insertion selected
    alpha = 1e-05
    regularization = ('phi', gamma, alpha) #('l1', alpha)
    activation = "relu"
    num_iterations = 20
    loss_weights = (1.0, 0.0)
    # Data inspection
    print(f"Loaded dataset from {path}")
    print(f"Dataset shape: {dataset.shape}")
    print(f"Dataset dtype: {dataset.dtype}")

    
    # Convert to dictionary format
    data_dict = {
        'x': np.array([item[0] for item in dataset]),
        'dv': np.array([item[1] for item in dataset]),
        'v': np.array([item[2] for item in dataset])
    }
    
    # Randomly permute the data
    num_samples = len(data_dict['x'])
    permutation = np.random.permutation(num_samples)
    data_dict['x'] = data_dict['x'][permutation]
    data_dict['dv'] = data_dict['dv'][permutation]
    data_dict['v'] = data_dict['v'][permutation]
    
    # Initialize the model with zero weights
    init_weights = np.random.randn(2, 2)
    init_bias = np.random.randn(2)
    model, weight, bias = network(data_dict, activation, power, regularization, loss_weights = loss_weights, inner_weights = init_weights, inner_bias = init_bias) 
    
    print("Initialization done")
    print(f"Initial weights shape: {weight.shape}, bias shape: {bias.shape}")
    
    # Get initial metrics if available
    train_loss = model.losshistory.loss_train[-1] if len(model.losshistory.loss_train) > 0 else None
    test_loss = model.losshistory.loss_test[-1] if len(model.losshistory.loss_test) > 0 else None
    test_metrics = model.losshistory.metrics_test[-1] if len(model.losshistory.metrics_test) > 0 else None
    
    # Simple reporting without formatting
    print(f"Initial train loss: {train_loss}")
    print(f"Initial test loss: {test_loss}")
    print(f"Initial test metrics: {test_metrics}")
    
    # Store losshistory objects to analyze later
    all_losshistory = [model.losshistory]
    neuron_counts = [weight.shape[0]]
    
    # Create a weights history object with metadata
    weights_history = {
        'iteration': [0],
        'neuron_count': [weight.shape[0]],
        'weights': [weight.copy()],
        'biases': [bias.copy()],
        'train_loss': [train_loss],
        'test_loss': [test_loss],
        'test_metrics': [test_metrics],
        'hyperparameters': {
            'activation': activation,
            'power': power,
            'gamma': gamma,
            'alpha': alpha
        }
    }
    
    # Training the model
    for i in range(num_iterations - 1):   
        print(f"\n----- Iteration {i} -----")
        weight_temp, bias_temp = insertion(data_dict, model, M, alpha)
        # Convert PyTorch tensors to NumPy arrays if needed
        if hasattr(weight_temp, 'numpy'):
            print("Converting weight_temp from PyTorch tensor to NumPy array")
            weight_temp = weight_temp.numpy()
        
        if hasattr(bias_temp, 'numpy'):
            print("Converting bias_temp from PyTorch tensor to NumPy array")
            bias_temp = bias_temp.numpy()
        
        print(f"Iteration {i} - inserted weights shape: {weight_temp.shape}, inserted bias shape: {bias_temp.shape}")
        print(f"Iteration {i} - current weights shape: {weight.shape}, current bias shape: {bias.shape}")
   
        weight = np.concatenate((weight, weight_temp), axis=0)
        bias = np.concatenate((bias, bias_temp), axis=0)
        
        # Use a new random seed for each iteration to get different train/test splits
        # np.random.seed(42 + i)
        model, weight, bias, outer_weights = network(data_dict, activation, power, regularization, loss_weights = loss_weights, inner_weights = weight, inner_bias = bias)
        print(f"After concatenation - weight shape: {weight.shape}, bias shape: {bias.shape}")
        
        # Get metrics from the model's losshistory object
        train_loss = model.losshistory.loss_train[-1] if len(model.losshistory.loss_train) > 0 else None
        test_loss = model.losshistory.loss_test[-1] if len(model.losshistory.loss_test) > 0 else None
        test_metrics = model.losshistory.metrics_test[-1] if len(model.losshistory.metrics_test) > 0 else None
        
        # Store information from this iteration
        all_losshistory.append(model.losshistory)
        neuron_counts.append(weight.shape[0])
        
        # Update weights history
        weights_history['iteration'].append(i+1)
        weights_history['neuron_count'].append(weight.shape[0])
        weights_history['weights'].append(weight.copy())
        weights_history['biases'].append(bias.copy())
        weights_history['train_loss'].append(train_loss)
        weights_history['test_loss'].append(test_loss)
        weights_history['test_metrics'].append(test_metrics)
        
        # Print model performance
        if train_loss is not None:
            print(f"Iteration {i+1} - Train loss: {train_loss}")
        if test_loss is not None:
            print(f"Iteration {i+1} - Test loss: {test_loss}")
        if test_metrics is not None:
            print(f"Iteration {i+1} - Test metrics: {test_metrics}")
        
        # Check if model has NaN values
        test_pred = model.predict(data_dict['x'][:1])
        if np.isnan(test_pred).any():
            print("WARNING: Model contains NaN values. Stopping training.")
            break
    
    # Save all weights, biases and loss history in a single file
    weights_dir = "data_result/weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    # Prepare loss history data
    loss_history = [lh.loss_train if hasattr(lh, 'loss_train') else [] for lh in all_losshistory]
    
    # Print debug information
    print("\nDebugging information for saving:")
    print(f"iterations shape: {np.array(weights_history['iteration']).shape}")
    print(f"neuron_counts shape: {np.array(weights_history['neuron_count']).shape}")
    print(f"Number of weight arrays: {len(weights_history['weights'])}")
    if weights_history['weights']:
        print(f"First weight array shape: {weights_history['weights'][0].shape}")
        print(f"Last weight array shape: {weights_history['weights'][-1].shape}")
    print(f"Number of bias arrays: {len(weights_history['biases'])}")
    if weights_history['biases']:
        print(f"First bias array shape: {weights_history['biases'][0].shape}")
        print(f"Last bias array shape: {weights_history['biases'][-1].shape}")
    print(f"train_loss length: {len(weights_history['train_loss'])}")
    print(f"test_loss length: {len(weights_history['test_loss'])}")
    print(f"test_metrics length: {len(weights_history['test_metrics'])}")
    if loss_history:
        print(f"loss_history: {len(loss_history)} items")
        print(f"first loss_history item type: {type(loss_history[0])}")
    
    # Save everything in a single file using pickle
    output_file = os.path.join(weights_dir, "training_history.pkl")
    try:
        # Add loss history to the weights_history dictionary
        weights_history['loss_history'] = loss_history
        
        # Save everything in a single pickle file
        import pickle
        with open(output_file, 'wb') as f:
            pickle.dump(weights_history, f)
        print(f"\nAll training history saved to {output_file}")
    except Exception as e:
        print(f"Error saving data: {e}")
        
        # Fallback to save at least the metadata
        try:
            output_file_npz = os.path.join(weights_dir, "training_history_metadata.npz")
            np.savez(
                output_file_npz,
                iterations=np.array(weights_history['iteration']),
                neuron_counts=np.array(weights_history['neuron_count'])
            )
            print(f"Metadata saved to {output_file_npz}")
        except Exception as e2:
            print(f"Error saving metadata: {e2}")
    
    # Save a more comprehensive metadata file in text format for easier access
    with open(os.path.join(weights_dir, "weights_metadata.txt"), "w") as f:
        f.write(f"Training run at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Hyperparameters: activation={activation}, power={power}, gamma={gamma}, loss_weights={loss_weights}, alpha={alpha}\n\n")
        f.write("Iteration summary:\n")
        for i, count in enumerate(weights_history['neuron_count']):
            train_loss = weights_history['train_loss'][i]
            test_loss = weights_history['test_loss'][i]
            test_metrics = weights_history['test_metrics'][i]
            
            f.write(f"Iteration {i}: {count} neurons\n")
            
            # Write train loss without sum
            if isinstance(train_loss, (list, np.ndarray)):
                f.write(f"  train loss: {train_loss}\n")
            else:
                f.write(f"  train loss: {train_loss}\n")
                
            # Write test loss without sum
            if isinstance(test_loss, (list, np.ndarray)):
                f.write(f"  test loss: {test_loss}\n")
            else:
                f.write(f"  test loss: {test_loss}\n")
                
            # Write test metrics without sum
            if isinstance(test_metrics, (list, np.ndarray)):
                f.write(f"  test metrics: {test_metrics}\n\n")
            else:
                f.write(f"  test metrics: {test_metrics}\n\n")


    
    

    



