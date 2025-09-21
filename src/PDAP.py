#!/usr/bin/env python3
"""
Simple training logger for VDP_Testing notebook
"""

import numpy as np
from loguru import logger
from .greedy_insertion import insertion
from src.greedy_insertion import _sample_uniform_sphere_points


def count_small_weights(weights, threshold):
    """Count weights below threshold."""
    return int(np.sum(np.abs(weights.flatten()) < threshold))


def prune_small_weights(weights, biases, outer_weights, threshold):
    """Prune neurons with small outer weights."""
    # Debug: print shapes
    logger.info(f"prune_small_weights - weights: {weights.shape}, biases: {biases.shape}, outer_weights: {outer_weights.shape}")
    
    # outer_weights has shape (1, n) from the network, need to extract the n values
    if outer_weights.ndim == 2 and outer_weights.shape[0] == 1:
        outer_weights_flat = outer_weights[0, :]  # Extract the (n,) array
    else:
        outer_weights_flat = outer_weights.flatten()
    
    # Find indices of neurons with outer weights above threshold
    keep_mask = np.abs(outer_weights_flat) >= threshold
    
    if np.sum(keep_mask) < len(weights):
        logger.info(f"Pruned {len(weights) - np.sum(keep_mask)} neurons with small weights")
    
    # Keep only neurons above threshold
    weights_pruned = weights[keep_mask]
    biases_pruned = biases[keep_mask]
    
    # Handle outer_weights shape properly - maintain the (1, n) shape
    if outer_weights.ndim == 2 and outer_weights.shape[0] == 1:
        outer_weights_pruned = outer_weights[:, keep_mask]  # Keep (1, n_pruned) shape
    else:
        outer_weights_pruned = outer_weights[keep_mask]
    
    logger.info(f"After pruning - weights: {weights_pruned.shape}, biases: {biases_pruned.shape}, outer_weights: {outer_weights_pruned.shape}")
    
    return weights_pruned, biases_pruned, outer_weights_pruned


def retrain(data_train, data_valid, model_1, model_2, num_iterations, M, threshold):
    
    history = []
    alpha = model_1.alpha
    
    # Track best model across all iterations
    best_iteration = 0
    best_val_loss = float('inf')
    
    # Training loop
    for i in range(num_iterations):
        logger.info(f"Iteration {i} - Starting...")
        W_hidden, b_hidden = _sample_uniform_sphere_points(M)
        model_1.train(data_train, data_valid, inner_weights=W_hidden, inner_bias=b_hidden)
        state_1 = model_1.net.state_dict()
        W_hidden, b_hidden, W_out = state_1['hidden.weight'], state_1['hidden.bias'], state_1['output.weight']
        model_2.train(data_train, data_valid, inner_weights=W_hidden, inner_bias=b_hidden, outer_weights=W_out)
        
        # Count and prune small weights
        state_2 = model_2.net.state_dict()
        small_count = np.sum(np.abs(state_2['output.weight'].flatten()) < threshold)
        logger.info(f"Small weights count: {small_count}, Pruning...")
        W_hidden, b_hidden = prune_small_weights(W_hidden, b_hidden, W_out, threshold)
        
        logger.info(f"Recording...")
        record = {'iteration': int(i), 'artifact': model_2.config, 'num_neurons': W_hidden.shape[0]}
        history.append(record)
        if model_2.config['best_val_loss'] < best_val_loss:
            best_val_loss = model_2.config['best_val_loss']
            best_iteration = i
            logger.info(f"New best model found at iteration {i} with validation loss: {best_val_loss:.6f}")
        
        # Insert neurons and train
        W_to_insert, b_to_insert = insertion(data_train, model_1, M, alpha)
        W_hidden = np.concatenate((W_hidden, W_to_insert), axis=0)
        b_hidden = np.concatenate((b_hidden, b_to_insert), axis=0)
        

    
    
    # Return the best model instead of the final one
    return best_iteration, history