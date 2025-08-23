#!/usr/bin/env python3
"""
Simple training logger for VDP_Testing notebook
"""

import os
import pickle
import numpy as np
from datetime import datetime
from loguru import logger

class TrainingLogger:
    """Simple training logger for greedy insertion experiments."""
    
    def __init__(self, project_root):
        self.project_root = project_root
        self.stats_dir = os.path.join(project_root, 'data_result', 'weights')
        os.makedirs(self.stats_dir, exist_ok=True)
        
        # Training history in format expected by plot_value_function.py
        self.history = {
            'weights': [],      # Inner weights per iteration
            'biases': [],       # Inner biases per iteration  
            'neuron_count': [], # Number of neurons per iteration
            'test_metrics': [], # Test metrics per iteration
            'hyperparameters': {}
        }
        
    def set_hyperparameters(self, activation, power, gamma, alpha):
        """Set hyperparameters."""
        self.history['hyperparameters'] = {
            'activation': activation,
            'power': power, 
            'gamma': gamma,
            'alpha': alpha
        }
        
    def log_iteration(self, iteration, weights, biases, test_loss):
        """Log iteration data."""
        self.history['weights'].append(weights.copy())
        self.history['biases'].append(biases.copy())
        self.history['neuron_count'].append(len(weights))
        self.history['test_metrics'].append([test_loss])
        
        logger.info(f"Iteration {iteration}: {len(weights)} neurons, test_loss={test_loss:.6f}")
    
    def save(self):
        """Save training history."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"training_history_{timestamp}.pkl"
        filepath = os.path.join(self.stats_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.history, f)
            
        logger.info(f"Saved training history to {filepath}")
        return filepath

def extract_test_loss(model_result):
    """Extract test loss from model result."""
    try:
        val_hist = getattr(model_result.losshistory, 'loss_test', [])
        return float(val_hist[-1]) if val_hist else np.nan
    except:
        return np.nan

def count_small_weights(weights, threshold):
    """Count weights below threshold."""
    return int(np.sum(np.abs(weights.flatten()) < threshold))

def run_training_with_logging(data, model_1, model_2, model_result, weight_raw, bias_raw, outerweight_raw,
                            num_iterations, M, alpha, pruning_threshold, power, gamma):
    """
    Run the training loop with improved logging.
    
    Args:
        data: Training data
        model_1, model_2: Model instances
        model_result: Initial model result
        weight_raw, bias_raw, outerweight_raw: Initial weights
        num_iterations, M, alpha, pruning_threshold: Training params
        power, gamma: Hyperparameters
        
    Returns:
        training_logger, final_weights, final_bias, final_outer_weights
    """
    from .greedy_insertion import insertion
    
    # Initialize logger
    project_root = os.path.abspath('..')
    training_logger = TrainingLogger(project_root)
    training_logger.set_hyperparameters('relu', power, gamma, alpha)
    
    # Training loop
    for i in range(num_iterations - 1):
        logger.info(f"Iteration {i} - weights shape: {weight_raw.shape}")
        
        # Train outer weights
        model, weight, bias, outerweights = model_2.train(
            iterations=1000,
            display_every=1000,
            inner_weights=weight_raw,
            inner_bias=bias_raw, 
            outer_weights=outerweight_raw
        )
        
        # Count pruned neurons
        small_count = count_small_weights(outerweight_raw, pruning_threshold)
        logger.info(f"Small weights count: {small_count}")
        
        # Insert neurons
        weight_temp, bias_temp = insertion(data, model_result, M, alpha)
        weights = np.concatenate((weight, weight_temp), axis=0)
        biases = np.concatenate((bias, bias_temp), axis=0)
        
        # Train full model
        model_result, weight_raw, bias_raw, outerweight_raw = model_1.train(
            inner_weights=weights,
            inner_bias=biases
        )
        
        # Log iteration
        test_loss = extract_test_loss(model_result)
        training_logger.log_iteration(i, weight_raw, bias_raw, test_loss)
    
    # Save results
    filepath = training_logger.save()
    logger.info(f"Training completed. History saved to {filepath}")
    
    return training_logger, weight_raw, bias_raw, outerweight_raw