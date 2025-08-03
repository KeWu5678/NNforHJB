#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to retry failed initial conditions using solutions from nearby successful points.
This loads existing results rather than regenerating the entire dataset.
"""

import os
import numpy as np
from scipy.spatial.distance import cdist
import openloop_optimizer as op
from test_VDP import VDP, V, gradient, gen_bc

# Load the existing dataset and failed initial conditions
dataset = np.load("data_result/VDP_beta_3_grid_30x30.npy")
failed_ini = np.load("data_result/VDP_beta_3_failed_ini.npy")

print(f"Loaded dataset with {len(dataset)} points")
print(f"Loaded {len(failed_ini)} failed initial conditions")

# Extract successful results from dataset
successful_results = {}
for item in dataset:
    ini = tuple(item['x'])
    # Check if this entry has valid results (not NaN)
    if not np.isnan(item['v']) and not np.isnan(item['dv']).any():
        successful_results[ini] = (item['dv'], item['v'])

print(f"Found {len(successful_results)} successful results in the dataset")

# Convert successful initial conditions to array for distance calculation
successful_ini = np.array([np.array(k) for k in successful_results.keys()])

# Setup for optimization
grid = np.linspace(0, 3, 1000)
tol = 5e-4
max_it = 500
beta = 3  # From test_VDP.py

# Track which failed conditions get fixed
fixed_ini = []
fixed_results = []

print(f"\n===== Retrying {len(failed_ini)} failed initial conditions with better guesses =====")

for j, fail_point in enumerate(failed_ini):
    print(f"Retrying failed point {j+1}/{len(failed_ini)}: {fail_point}")
    
    # Find nearest successful initialization point
    if len(successful_ini) > 0:
        # Calculate distances from this failed point to all successful points
        distances = cdist([fail_point], successful_ini, 'euclidean')[0]
        
        # Get index of nearest point
        nearest_idx = np.argmin(distances)
        nearest_point = tuple(successful_ini[nearest_idx])
        nearest_distance = distances[nearest_idx]
        
        print(f"  Using solution from nearest point {nearest_point} (distance: {nearest_distance:.4f})")
        
        # Get the successful solution to use as a better guess
        dv_nearest, v_nearest = successful_results[nearest_point]
        
        # Create a better initial guess for BVP based on the nearest successful solution
        better_guess = np.ones((4, grid.size))
        # Initialize adjoint variables with gradients from nearest solution
        better_guess[2, :] = dv_nearest[0]  # Initialize y3 with gradient from nearest solution
        better_guess[3, :] = dv_nearest[1]  # Initialize y4 with gradient from nearest solution
        
        # Try again with the better guess
        bc_func = gen_bc(fail_point)
        dv, v = op.OpenLoopOptimizer(VDP, bc_func, V, gradient, grid, better_guess, tol, max_it).optimize()
        
        if dv is not None and not np.isnan(v):
            print(f"  SUCCESS! Point {fail_point} now converged with value {v}")
            # Store result
            fixed_ini.append(fail_point)
            fixed_results.append((fail_point, dv, v))
            # Add to successful results for potential use by later failed points
            successful_results[tuple(fail_point)] = (dv, v)
            successful_ini = np.append(successful_ini, [fail_point], axis=0)
        else:
            print(f"  Still failed to converge for {fail_point}")
    else:
        print("  No successful points to use as reference.")

print(f"\nFixed {len(fixed_ini)} out of {len(failed_ini)} failed initial conditions")

# Save results
if fixed_results:
    # Create a structured array to hold the new results
    dtype = [
        ('x', '2float64'),  # 2D float array for initial condition
        ('dv', '2float64'), # 2D float array for gradient
        ('v', 'float64')    # 1D float for value
    ]
    
    fixed_data = np.zeros(len(fixed_results), dtype=dtype)
    for i, (ini, dv, v) in enumerate(fixed_results):
        fixed_data[i] = (ini, dv, v)
    
    # Save the new results
    np.save("VDP_beta_3_fixed_points_1.npy", fixed_data)
    print(f"Saved {len(fixed_results)} fixed points to VDP_beta_3_fixed_points_1.npy")
    
    # Update the list of remaining failed points
    remaining_failed = [point for point in failed_ini if not any(np.array_equal(point, fixed) for fixed in fixed_ini)]
    if remaining_failed:
        np.save("VDP_beta_3_remaining_failed_1.npy", np.array(remaining_failed))
        print(f"Saved {len(remaining_failed)} remaining failed points to VDP_beta_3_remaining_failed_1.npy")
else:
    print("No points were fixed. The original failed_ini.npy file remains unchanged.")

# Optionally, create a combined dataset with all successful points
if fixed_results:
    # Combine the original dataset with the fixed points
    combined = np.zeros(len(dataset) + len(fixed_results), dtype=dtype)
    
    # Copy original dataset
    combined[:len(dataset)] = dataset
    
    # Add fixed points
    # First, identify indices in the original dataset that correspond to failed points
    # and replace those entries instead of appending
    valid_count = 0
    for i, (ini, dv, v) in enumerate(fixed_results):
        # Find if this ini exists in the original dataset with NaN values
        found = False
        for j, item in enumerate(dataset):
            if np.array_equal(item['x'], ini) and (np.isnan(item['v']) or np.isnan(item['dv']).any()):
                # Replace the invalid entry
                combined[j] = (ini, dv, v)
                found = True
                break
        
        if not found:
            # This is a new point, append it
            combined[len(dataset) + valid_count] = (ini, dv, v)
            valid_count += 1
    
    # Trim the array if we didn't add all fixed points (because some replaced existing entries)
    if valid_count < len(fixed_results):
        combined = combined[:len(dataset) + valid_count]
    
    # Save combined dataset
    np.save("VDP_beta_3_grid_30x30_complete.npy", combined)
    print(f"Saved combined dataset with {len(combined)} points to VDP_beta_3_grid_30x30_complete.npy") 