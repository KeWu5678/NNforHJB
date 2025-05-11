import numpy as np
import matplotlib.pyplot as plt
import re
import os
import pickle
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def extract_data(file_path):
    """Extract neuron count, L2 error, and H1 error from metadata file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract gamma value from hyperparameters
    gamma_match = re.search(r'gamma=(\d*\.?\d*)', content)
    gamma = float(gamma_match.group(1)) if gamma_match and gamma_match.group(1) else None
    
    # Special handling for specific files
    if "weights_metadata_26.txt" in file_path:
        gamma = 10.0
    elif "weights_metadata_27.txt" in file_path:
        gamma = 0.01
    elif "weights_metadata_28.txt" in file_path:
        gamma = 0.0
    
    print(f"Using gamma={gamma} for {os.path.basename(file_path)}")
    
    # Extract loss_weights to identify if fitting with gradient or not
    loss_weights_match = re.search(r'loss_weights=\((.*?)\)', content)
    loss_weights = loss_weights_match.group(1) if loss_weights_match else None
    using_gradient = "(1.0, 1.0)" in loss_weights if loss_weights else False
    
    # Extract iteration data
    iterations = re.findall(r'Iteration (\d+): (\d+) neurons.*?test metrics: \[(.*?), (.*?)\]', content, re.DOTALL)
    
    neurons = []
    l2_errors = []
    h1_errors = []
    
    for iteration, neuron_count, l2_error, h1_error in iterations:
        # Only include data points where neurons > 0
        if int(neuron_count) > 0:
            neurons.append(int(neuron_count))
            l2_errors.append(float(l2_error))
            h1_errors.append(float(h1_error))
    
    print(f"Extracted {len(neurons)} data points from {os.path.basename(file_path)}")
    return neurons, l2_errors, h1_errors, gamma, using_gradient

# Read data for comparison between different gamma values
neurons_26, l2_errors_26, h1_errors_26, gamma_26, _ = extract_data("data_result/weights/weights_metadata_26.txt")
neurons_27, l2_errors_27, h1_errors_27, gamma_27, _ = extract_data("data_result/weights/weights_metadata_27.txt")
neurons_28, l2_errors_28, h1_errors_28, gamma_28, _ = extract_data("data_result/weights/weights_metadata_28.txt")
neurons_29, l2_errors_29, h1_errors_29, gamma_29, _ = extract_data("data_result/weights/weights_metadata_29.txt")

# Read data for comparison between fitting with and without gradient
neurons_14, l2_errors_14, h1_errors_14, gamma_14, using_gradient_14 = extract_data("data_result/weights/weights_metadata_14.txt")
neurons_18, l2_errors_18, h1_errors_18, gamma_18, using_gradient_18 = extract_data("data_result/weights/weights_metadata_18.txt")

# =============================================================================
# PLOT 1: Comparison between fitting with and without gradient (no penalty)
# Shows how gradient fitting affects L2 and H1 errors when no penalty is used
# =============================================================================

# NEW PLOT: COMPARING WEIGHTS_METADATA_29 AND WEIGHTS_METADATA_24 (NO PENALTY)
# Read data for comparison between fitting with and without gradient (no penalty)
neurons_29, l2_errors_29, h1_errors_29, gamma_29, _ = extract_data("data_result/weights/weights_metadata_29.txt")
neurons_24, l2_errors_24, h1_errors_24, gamma_24, _ = extract_data("data_result/weights/weights_metadata_24.txt")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot L2 errors
ax1.plot(neurons_29, l2_errors_29, 'g-', label='With Gradient (1.0, 1.0)')
ax1.plot(neurons_24, l2_errors_24, 'm-', label='Without Gradient (1.0, 0.0)')
ax1.set_xlabel('Number of Neurons')
ax1.set_ylabel('L2 Error')
ax1.set_title('L2 Error vs. Number of Neurons')
ax1.legend()
ax1.grid(True)
ax1.set_xlim(0, 400)  # Set x-axis limits based on data

# Plot H1 errors
ax2.plot(neurons_29, h1_errors_29, 'g-', label='With Gradient (1.0, 1.0)')
ax2.plot(neurons_24, h1_errors_24, 'm-', label='Without Gradient (1.0, 0.0)')
ax2.set_xlabel('Number of Neurons')
ax2.set_ylabel('H1 Error')
ax2.set_title('H1 Error vs. Number of Neurons')
ax2.legend()
ax2.grid(True)
ax2.set_xlim(0, 400)  # Set x-axis limits based on data

# Set log scale for y-axis on both plots
ax1.set_yscale('log')
ax2.set_yscale('log')

# Add a main title
plt.suptitle('Testing Error: Fitting With vs. Without Gradient (no penalty)', fontsize=14)
plt.tight_layout()

# Save the figure
plt.savefig('data_result/plot/Fitting With vs. Without Gradient_no_penalty.png', dpi=300)
print(f"Plot saved to data_result/plot/Fitting With vs. Without Gradient_no_penalty.png")

plt.show()

# =============================================================================
# PLOT 2: Comparison between fitting with and without gradient for nonconvex penalty (γ=0.01)
# Shows how gradient fitting affects L2 and H1 errors when using nonconvex penalty
# =============================================================================

# Second plot: Comparison between fitting with and without gradient
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot L2 errors
ax1.plot(neurons_14, l2_errors_14, 'g-', label='Without Gradient (1.0, 0.0)')
ax1.plot(neurons_18, l2_errors_18, 'm-', label='With Gradient (1.0, 1.0)')
ax1.set_xlabel('Number of Neurons')
ax1.set_ylabel('L2 Error')
ax1.set_title('L2 Error vs. Number of Neurons')
ax1.legend()
ax1.grid(True)
ax1.set_xlim(0, 100)  # Set x-axis limits from 0 to 100 (adjust based on data)

# Plot H1 errors
ax2.plot(neurons_14, h1_errors_14, 'g-', label='Without Gradient (1.0, 0.0)')
ax2.plot(neurons_18, h1_errors_18, 'm-', label='With Gradient (1.0, 1.0)')
ax2.set_xlabel('Number of Neurons')
ax2.set_ylabel('H1 Error')
ax2.set_title('H1 Error vs. Number of Neurons')
ax2.legend()
ax2.grid(True)
ax2.set_xlim(0, 100)  # Set x-axis limits from 0 to 100 (adjust based on data)

# Set log scale for y-axis on both plots
ax1.set_yscale('log')
ax2.set_yscale('log')

# Add a main title
plt.suptitle('Testing Error: Fitting With vs. Without Gradient for nonconvex penalty (γ=0.01)', fontsize=14)
plt.tight_layout()

# Save the figure
plt.savefig('data_result/plot/Fitting With vs. Without Gradient_nonconvex.png', dpi=300)
print(f"Plot saved to data_result/plot/error_vs_neurons_gradient_comparison.png")

# =============================================================================
# PLOT 3: Comparison between different penalty terms (without gradient)
# Shows how different penalty terms affect L2 and H1 errors when not using gradient
# =============================================================================

# NEW PLOT: COMPARING WEIGHTS_METADATA_14, WEIGHTS_METADATA_15, WEIGHTS_METADATA_25, AND WEIGHTS_METADATA_24
# Read data for comparison between L1, gamma=5.0, gamma=0.01, and no penalty (all without gradient)
neurons_14, l2_errors_14, h1_errors_14, gamma_14, _ = extract_data("data_result/weights/weights_metadata_14.txt")
neurons_15, l2_errors_15, h1_errors_15, gamma_15, _ = extract_data("data_result/weights/weights_metadata_15.txt")
neurons_25, l2_errors_25, h1_errors_25, gamma_25, _ = extract_data("data_result/weights/weights_metadata_25.txt")
neurons_24, l2_errors_24, h1_errors_24, gamma_24, _ = extract_data("data_result/weights/weights_metadata_24.txt")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot L2 errors
ax1.plot(neurons_15, l2_errors_15, 'b-', label=f'γ={gamma_15}')
ax1.plot(neurons_25, l2_errors_25, 'r-', label=f'γ={gamma_25}')
ax1.plot(neurons_14, l2_errors_14, 'g-', label=f'L1')
ax1.plot(neurons_24, l2_errors_24, 'm-', label=f'no penalty')
ax1.set_xlabel('Number of Neurons')
ax1.set_ylabel('L2 Error')
ax1.set_title('L2 Error vs. Number of Neurons')
ax1.legend()
ax1.grid(True)
ax1.set_xlim(0, 100)  # Set x-axis limits to 100 neurons

# Plot H1 errors
ax2.plot(neurons_15, h1_errors_15, 'b-', label=f'γ={gamma_15}')
ax2.plot(neurons_25, h1_errors_25, 'r-', label=f'γ={gamma_25}')
ax2.plot(neurons_14, h1_errors_14, 'g-', label=f'L1')
ax2.plot(neurons_24, h1_errors_24, 'm-', label=f'no penalty')
ax2.set_xlabel('Number of Neurons')
ax2.set_ylabel('H1 Error')
ax2.set_title('H1 Error vs. Number of Neurons')
ax2.legend()
ax2.grid(True)
ax2.set_xlim(0, 100)  # Set x-axis limits to 100 neurons

# Set log scale for y-axis on both plots
ax1.set_yscale('log')
ax2.set_yscale('log')

# Add a main title
plt.suptitle('Cost without Gradient: comparing different penalty terms', fontsize=14)
plt.tight_layout()

# Save the figure
plt.savefig('data_result/plot/Cost without gradient: comparing different penalty terms.png', dpi=300)
print(f"Plot saved to data_result/plot/Cost without gradient: comparing different penalty terms.png")

plt.show()

# =============================================================================
# PLOT 4: Cost with gradient: compare different penalty terms (γ=5.0, γ=0.01, l1, no penalty)
# Shows how different penalty terms (γ=5.0, γ=0.01, l1, no penalty) affect L2 and H1 errors
# =============================================================================

# First plot: Comparison between gamma values
# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot L2 errors with normal lines but make them thicker for visibility
ax1.plot(neurons_26, l2_errors_26, 'b-', linewidth=3, marker='o', markersize=8, label=f'γ={gamma_26}')
ax1.plot(neurons_27, l2_errors_27, 'r-', linewidth=3, marker='s', markersize=8, label=f'γ={gamma_27}')
ax1.plot(neurons_28, l2_errors_28, 'g-', linewidth=3, marker='^', markersize=8, label=f'L1')
ax1.plot(neurons_29, l2_errors_29, 'm-', linewidth=3, marker='*', markersize=8, label=f'no penalty')
ax1.set_xlabel('Number of Neurons')
ax1.set_ylabel('L2 Error')
ax1.set_title('L2 Error vs. Number of Neurons')
ax1.legend()
ax1.grid(True)
ax1.set_xlim(0, 400)  # Set x-axis limits from 0 to 400

# Plot H1 errors with normal lines but make them thicker for visibility
ax2.plot(neurons_26, h1_errors_26, 'b-', linewidth=3, marker='o', markersize=8, label=f'γ={gamma_26}')
ax2.plot(neurons_27, h1_errors_27, 'r-', linewidth=3, marker='s', markersize=8, label=f'γ={gamma_27}')
ax2.plot(neurons_28, h1_errors_28, 'g-', linewidth=3, marker='^', markersize=8, label=f'L1')
ax2.plot(neurons_29, h1_errors_29, 'm-', linewidth=3, marker='*', markersize=8, label=f'no penalty')
ax2.set_xlabel('Number of Neurons')
ax2.set_ylabel('H1 Error')
ax2.set_title('H1 Error vs. Number of Neurons')
ax2.legend()
ax2.grid(True)
ax2.set_xlim(0, 400)  # Set x-axis limits from 0 to 400

# Set log scale for y-axis on both plots
ax1.set_yscale('log')
ax2.set_yscale('log')

# Add a main title
plt.suptitle('Cost with Gradient: compare different penalty terms', fontsize=14)
plt.tight_layout()

# Save the figure
plt.savefig('data_result/plot/Cost with gradient: compare different penalty terms.png', dpi=300)
print(f"Plot saved to data_result/plot/Cost with gradient: compare different penalty terms.png")

# =============================================================================
# PLOT 5: Weight space visualization in polar coordinates
# Shows the distribution of weights in 2D space for different penalty terms
# =============================================================================

# Load pickle files
print("\nLoading pickle files for weight analysis...")
with open('data_result/weights/training_history_26.pkl', 'rb') as f:
    training_history_26 = pickle.load(f)

with open('data_result/weights/training_history_27.pkl', 'rb') as f:
    training_history_27 = pickle.load(f)

with open('data_result/weights/training_history_28.pkl', 'rb') as f:
    training_history_28 = pickle.load(f)

print("Successfully loaded pickle files")

# Extract inner weights and biases
weights_26_iter7 = None
biases_26_iter7 = None
weights_27_iter8 = None
biases_27_iter8 = None
weights_28_iter7 = None
biases_28_iter7 = None

if 7 < len(training_history_26['weights']):
    weights_26_iter7 = training_history_26['weights'][7]
    biases_26_iter7 = training_history_26['biases'][7]
    neurons_26_iter7 = training_history_26['neuron_count'][7]
    print(f"Extracted weights from training_history_26, iteration 7: {weights_26_iter7.shape}, neurons: {neurons_26_iter7}")
else:
    print(f"Error: training_history_26 doesn't have iteration 7. Max: {len(training_history_26['weights'])-1}")

if 8 < len(training_history_27['weights']):
    weights_27_iter8 = training_history_27['weights'][8]
    biases_27_iter8 = training_history_27['biases'][8]
    neurons_27_iter8 = training_history_27['neuron_count'][8]
    print(f"Extracted weights from training_history_27, iteration 8: {weights_27_iter8.shape}, neurons: {neurons_27_iter8}")
else:
    print(f"Error: training_history_27 doesn't have iteration 8. Max: {len(training_history_27['weights'])-1}")

if 7 < len(training_history_28['weights']):
    weights_28_iter7 = training_history_28['weights'][7]
    biases_28_iter7 = training_history_28['biases'][7]
    neurons_28_iter7 = training_history_28['neuron_count'][7]
    print(f"Extracted weights from training_history_28, iteration 7: {weights_28_iter7.shape}, neurons: {neurons_28_iter7}")
else:
    print(f"Error: training_history_28 doesn't have iteration 7. Max: {len(training_history_28['weights'])-1}")

# =============== POLAR COORDINATE VISUALIZATION  =================
if weights_26_iter7 is not None and weights_27_iter8 is not None and weights_28_iter7 is not None:
    # Extract the gamma values from the metadata files
    _, _, _, gamma_26, _ = extract_data("data_result/weights/weights_metadata_26.txt")
    _, _, _, gamma_27, _ = extract_data("data_result/weights/weights_metadata_27.txt")
    _, _, _, gamma_28, _ = extract_data("data_result/weights/weights_metadata_28.txt")
    
    # Create figure for polar coordinate plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': 'polar'})
    
    # 1. Polar plot for Training 26
    # Compute angles and radii in weight space (2D)
    xy_norms_26 = np.sqrt(weights_26_iter7[:, 0]**2 + weights_26_iter7[:, 1]**2)
    angles_26 = np.arctan2(weights_26_iter7[:, 1], weights_26_iter7[:, 0])
    
    # Plot in polar coordinates
    axes[0].scatter(angles_26, xy_norms_26, color='blue', alpha=0.8, s=60)
    axes[0].set_title(f'Weight Space at Optimal Iteration\nTraining 26 (γ={gamma_26})\nNeurons: {neurons_26_iter7}', fontsize=12)
    axes[0].grid(True, alpha=0.5)
    
    # 2. Polar plot for Training 27
    # Compute angles and radii in weight space (2D)
    xy_norms_27 = np.sqrt(weights_27_iter8[:, 0]**2 + weights_27_iter8[:, 1]**2)
    angles_27 = np.arctan2(weights_27_iter8[:, 1], weights_27_iter8[:, 0])
    
    # Plot in polar coordinates
    axes[1].scatter(angles_27, xy_norms_27, color='red', alpha=0.8, s=60)
    axes[1].set_title(f'Weight Space at Optimal Iteration\nTraining 27 (γ={gamma_27})\nNeurons: {neurons_27_iter8}', fontsize=12)
    axes[1].grid(True, alpha=0.5)
    
    # 3. Polar plot for Training 28
    # Compute angles and radii in weight space (2D)
    xy_norms_28 = np.sqrt(weights_28_iter7[:, 0]**2 + weights_28_iter7[:, 1]**2)
    angles_28 = np.arctan2(weights_28_iter7[:, 1], weights_28_iter7[:, 0])
    
    # Plot in polar coordinates
    axes[2].scatter(angles_28, xy_norms_28, color='green', alpha=0.8, s=60)
    axes[2].set_title(f'Weight Space at Optimal Iteration\nTraining 28 (γ={gamma_28})\nNeurons: {neurons_28_iter7}', fontsize=12)
    axes[2].grid(True, alpha=0.5)
    
    # Adjust spacing without main title
    plt.subplots_adjust(wspace=0.3)
    
    # Save the figure
    plt.savefig('data_result/plot/weights_polar_analysis.png', dpi=300)
    print(f"Polar coordinate analysis saved to data_result/plot/weights_polar_analysis.png")
    
    # Show plot
    plt.show()

# Show all plots
plt.show()





