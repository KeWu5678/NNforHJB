#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:32:26 2024

@author: chaoruiz
"""
from src.model import model
from src.model_outerweights import model_outerweights
from src.greedy_insertion import insertion

import numpy as np
from loguru import logger
import torch

# load the data
path = 'data_result/raw_data/VDP_beta_0.1_grid_combined.npy'# Initialize the weights
data = np.load(path)
logger.info(f"Loaded data from {path}, shape: {data.shape}, dtype: {data.dtype}")

# Initialize the parameter
power = 1
gamma = 10.0
M = 20 # number greedy insertion selected
alpha = 1e-5
regularization = (gamma, alpha) 
activation = "relu"
num_iterations = 20
loss_weights = (1.0, 1.0)
pruning_threshold = 1e-3

# Initialize the model with zero neurons (no hidden layer)
init_weights = None
init_bias = None
model_1 = model(data, torch.relu, 2.0, regularization, optimizer='Adam', loss_weights = loss_weights)
model_2 = model_outerweights(data, torch.relu, 2.0, regularization, optimizer='SSN', loss_weights = loss_weights)
model_result, weight, bias, output_weight = model_1.train(
    iterations=5000,
    display_every=1000
)
logger.info("Initialization done"); logger.info(f"Initial weights shape: {weight.shape}, bias shape: {bias.shape}")


# Training the model
for i in range(num_iterations - 1):  
    logger.info(f"Iteration {i} - current weights shape: {weight.shape}, current bias shape: {bias.shape}") 
    # insert M neurons
    weight_temp, bias_temp = insertion(data, model_result, M, alpha)
    # weight_temp and bias_temp are already numpy arrays from insertion()
    weights = np.concatenate((weight, weight_temp), axis=0)
    biases = np.concatenate((bias, bias_temp), axis=0)
    logger.info(f"Iteration {i} - inserted weights shape: {weight_temp.shape}, inserted bias shape: {bias_temp.shape}")
    
    # train 1st model with adam than accelarate with 2nd model with ssn
    model_result, weight_raw, bias_raw, outerweight_raw = model_1.train(inner_weights=weights, inner_bias=biases)
    model, weight, bias, outerweights = model_2.train(inner_weights = weight_raw, inner_bias = bias_raw, outer_weights = outerweight_raw)
            
    # Convert to flat array and count elements with absolute value less than threshold
    outerweights_raw = outerweight_raw.flatten() 
    outerweights = outerweights.flatten()
    small_weights_count = np.sum(np.abs(outerweights_raw) < pruning_threshold)
    small_weights_filtered_count = np.sum(np.abs(outerweights) < pruning_threshold)
    
    logger.info(f"1st model weights shape: {np.shape(outerweight_raw)}, 2nd model weights shape: {np.shape(outerweights)}, Pruned neurons in 2nd model with abs value < {pruning_threshold}: {small_weights_count}")
    
    # Delete neurons whose outer weights are under the pruning threshold
    if outerweights is not None and len(weight_raw) > 0:
        outerweights_abs = np.abs(outerweights) # Get the absolute values of the outer weights
        keep_indices = np.where(outerweights_abs >= pruning_threshold)[0]   # Find indices of outerweights >= threshold
        pruned_count = len(weight_raw) - len(keep_indices)  # Calculate how many neurons would be pruned
        
        # Prune the weights and bias if needed
        if pruned_count > 0 and len(keep_indices) > 0:
            weight = weight_raw[keep_indices]
            bias = bias_raw[keep_indices]
            logger.info(f"Pruning: Removing {pruned_count} neurons with outer weights < {pruning_threshold}, "
                        f"After pruning - weight shape: {weight.shape}, bias shape: {bias.shape}")
        else:
            if pruned_count > 0:
                logger.info(f"Warning: Cannot prune all {pruned_count} neurons - would leave no neurons")
            else:
                logger.info(f"No neurons pruned (all {len(weight_raw)} neurons have outer weights >= {pruning_threshold})")
            pruned_count = 0
            weight = weight_raw # Keep the original weights and biases
            bias = bias_raw
    else:
        pruned_count = 0
        logger.info("No pruning performed (no neurons or outer weights not available)")
        weight = weight_raw
        bias = bias_raw
    
    logger.info(f"After pruning - Final network has {weight.shape[0]} neurons")
    
#     # Get metrics from the model's losshistory object
#     train_loss = model.losshistory.loss_train[-1] if len(model.losshistory.loss_train) > 0 else None
#     test_loss = model.losshistory.loss_test[-1] if len(model.losshistory.loss_test) > 0 else None
#     test_metrics = model.losshistory.metrics_test[-1] if len(model.losshistory.metrics_test) > 0 else None
    
#     # Store information from this iteration
#     all_losshistory.append(model.losshistory)
#     neuron_counts.append(weight.shape[0])
    
    
#     # Print model performance
#     if train_loss is not None:
#         print(f"Iteration {i+1} - Train loss: {train_loss}")
#     if test_loss is not None:
#         print(f"Iteration {i+1} - Test loss: {test_loss}")
#     if test_metrics is not None:
#         print(f"Iteration {i+1} - Test metrics: {test_metrics}")
    
#     # Check if model has NaN values
#     test_pred = model.predict(data['x'][:1])
#     if np.isnan(test_pred).any():
#         print("WARNING: Model contains NaN values. Stopping training.")
#         break

# # Save all weights, biases and loss history in a single file
# weights_dir = "data_result/weights"
# os.makedirs(weights_dir, exist_ok=True)



# # Print debug information
# print("\nDebugging information for saving:")
# print(f"iterations shape: {np.array(weights_history['iteration']).shape}")
# print(f"neuron_counts shape: {np.array(weights_history['neuron_count']).shape}")
# print(f"Number of weight arrays: {len(weights_history['weights'])}")
# if weights_history['weights']:
#     print(f"First weight array shape: {weights_history['weights'][0].shape}")
#     print(f"Last weight array shape: {weights_history['weights'][-1].shape}")
# print(f"Number of bias arrays: {len(weights_history['biases'])}")
# if weights_history['biases']:
#     print(f"First bias array shape: {weights_history['biases'][0].shape}")
#     print(f"Last bias array shape: {weights_history['biases'][-1].shape}")
# print(f"train_loss length: {len(weights_history['train_loss'])}")
# print(f"test_loss length: {len(weights_history['test_loss'])}")
# print(f"test_metrics length: {len(weights_history['test_metrics'])}")
# if loss_history:
#     print(f"loss_history: {len(loss_history)} items")
#     print(f"first loss_history item type: {type(loss_history[0])}")

# Save everything in a single file using pickle
# output_file = os.path.join(weights_dir, "training_history.pkl")
# try:
#     # Add loss history to the weights_history dictionary
#     weights_history['loss_history'] = loss_history
    
#     # Save everything in a single pickle file
#     import pickle
#     with open(output_file, 'wb') as f:
#         pickle.dump(weights_history, f)
#     print(f"\nAll training history saved to {output_file}")
# except Exception as e:
#     print(f"Error saving data: {e}")
    
#     # Fallback to save at least the metadata
#     try:
#         output_file_npz = os.path.join(weights_dir, "training_history_metadata.npz")
#         np.savez(
#             output_file_npz,
#             iterations=np.array(weights_history['iteration']),
#             neuron_counts=np.array(weights_history['neuron_count'])
#         )
#         print(f"Metadata saved to {output_file_npz}")
#     except Exception as e2:
#         print(f"Error saving metadata: {e2}")

# # Save a more comprehensive metadata file in text format for easier access
# with open(os.path.join(weights_dir, "weights_metadata.txt"), "w") as f:
#     f.write(f"Training run at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#     f.write(f"Hyperparameters: activation={activation}, power={power}, gamma={gamma}, loss_weights={loss_weights}, alpha={alpha}\n\n")
#     f.write("Iteration summary:\n")
#     for i, count in enumerate(weights_history['neuron_count']):
#         train_loss = weights_history['train_loss'][i]
#         test_loss = weights_history['test_loss'][i]
#         test_metrics = weights_history['test_metrics'][i]
        
#         pruned_info = ""
#         if i > 0 and 'pruned_neurons' in weights_history:
#             pruned_info = f" (pruned {weights_history['pruned_neurons'][i]} neurons)"
#         else:
#             pruned_info = ""
            
#         f.write(f"Iteration {i}: {count} neurons{pruned_info}\n")
        
#         # Write train loss without sum
#         if isinstance(train_loss, (list, np.ndarray)):
#             f.write(f"  train loss: {train_loss}\n")
#         else:
#             f.write(f"  train loss: {train_loss}\n")
            
#         # Write test loss without sum
#         if isinstance(test_loss, (list, np.ndarray)):
#             f.write(f"  test loss: {test_loss}\n")
#         else:
#             f.write(f"  test loss: {test_loss}\n")
            
#         # Write test metrics without sum
#         if isinstance(test_metrics, (list, np.ndarray)):
#             f.write(f"  test metrics: {test_metrics}\n\n")
#         else:
#             f.write(f"  test metrics: {test_metrics}\n\n")


    
    

    



