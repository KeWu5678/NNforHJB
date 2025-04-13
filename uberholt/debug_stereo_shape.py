import numpy as np
import torch
import sys
import os
from greedy_insertion import stereo, insertion
import deepxde as dde

# Add path for deepxde access if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Monkey patch the stereo function to print the shape
original_stereo = stereo

def debug_stereo(z):
    print("DEBUG - z shape in stereo:", z.shape if hasattr(z, 'shape') else "No shape attribute")
    print("DEBUG - z type:", type(z))
    print("DEBUG - z content preview:", z)
    
    if isinstance(z, torch.Tensor):
        z = z.detach().numpy()
    
    if not isinstance(z, np.ndarray):
        print("ERROR - z is not a numpy array!")
        z = np.array(z)
    
    # If z is 1D array of length 2, reshape it to (2,1)
    if z.ndim == 1 and len(z) == 2:
        print("DEBUG - Reshaping z from 1D to (2,1)")
        z = z.reshape(2, 1)
    
    # If z is (n, 2) instead of (2, n), transpose it
    if z.ndim == 2 and z.shape[1] == 2 and z.shape[0] != 2:
        print(f"DEBUG - Transposing z from {z.shape} to {z.shape[::-1]}")
        z = z.T
    
    try:
        return original_stereo(z)
    except Exception as e:
        print(f"ERROR in stereo: {e}")
        print(f"z shape: {z.shape}")
        print(f"z: {z}")
        raise e

# Load the data for testing
try:
    path = 'data/VDP_beta_3_patch1.npy'
    dataset = np.load(path)
    print(f"Loaded dataset with shape: {dataset.shape}")
    
    # Create a mini model for testing
    print("Creating test model...")
    from network import network
    model, _, _ = network(dataset, 2.5, ('phi', 4, 0))
    print("Model created")
    
    # Test insertion with debug
    from greedy_insertion import insertion as original_insertion
    
    # Monkey patch the stereo function
    import greedy_insertion
    greedy_insertion.stereo = debug_stereo
    
    # Run insertion with a small M
    print("Running insertion with M=1...")
    try:
        weight, bias = greedy_insertion.insertion(dataset, model, 1)
        print("Insertion successful!")
        print(f"Weight shape: {weight.shape}, bias shape: {bias.shape}")
    except Exception as e:
        print(f"Insertion failed with error: {e}")
    
except Exception as e:
    print(f"Error: {e}")

print("\nChecking mean_l2_relative_error function...")
# Check the mean_l2_relative_error function from deepxde
print(dde.losses.mean_l2_relative_error) 