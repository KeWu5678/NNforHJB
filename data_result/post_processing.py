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
    if "weights_metadata_20.txt" in file_path:
        gamma = 5.0
    elif "weights_metadata_21.txt" in file_path:
        gamma = 0.01
    elif "weights_metadata_22.txt" in file_path:
        gamma = 0.0
    
    print(f"Using gamma={gamma} for {os.path.basename(file_path)}")
    
    # Extract loss_weights to identify if fitting with gradient or not
    loss_weights_match = re.search(r'loss_weights=\((.*?)\)', content)
    loss_weights = loss_weights_match.group(1) if loss_weights_match else None
    using_gradient = "(1.0, 1.0)" in loss_weights if loss_weights else False
    
    # Extract iteration data
    iterations = re.findall(r'Iteration (\d+): (\d+) neurons\n.*?test metrics: \[(.*?), (.*?)\]', content, re.DOTALL)
    
    neurons = []
    l2_errors = []
    h1_errors = []
    
    for iteration, neuron_count, l2_error, h1_error in iterations:
        neurons.append(int(neuron_count))
        l2_errors.append(float(l2_error))
        h1_errors.append(float(h1_error))
    
    return neurons, l2_errors, h1_errors, gamma, using_gradient

# Read data for comparison between different gamma values
neurons_20, l2_errors_20, h1_errors_20, gamma_20, _ = extract_data("data_result/weights/weights_metadata_20.txt")
neurons_21, l2_errors_21, h1_errors_21, gamma_21, _ = extract_data("data_result/weights/weights_metadata_21.txt")
neurons_22, l2_errors_22, h1_errors_22, gamma_22, _ = extract_data("data_result/weights/weights_metadata_22.txt")

# Read data for comparison between fitting with and without gradient
neurons_14, l2_errors_14, h1_errors_14, gamma_14, using_gradient_14 = extract_data("data_result/weights/weights_metadata_14.txt")
neurons_18, l2_errors_18, h1_errors_18, gamma_18, using_gradient_18 = extract_data("data_result/weights/weights_metadata_18.txt")

# First plot: Comparison between gamma values
# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot L2 errors - removed the 'o' marker
ax1.plot(neurons_20, l2_errors_20, 'b-', label=f'γ={gamma_20}')
ax1.plot(neurons_21, l2_errors_21, 'r-', label=f'γ={gamma_21}')
ax1.plot(neurons_22, l2_errors_22, 'g-', label=f'γ={gamma_22}')
ax1.set_xlabel('Number of Neurons')
ax1.set_ylabel('L2 Error')
ax1.set_title('L2 Error vs. Number of Neurons')
ax1.legend()
ax1.grid(True)
ax1.set_xlim(0, 400)  # Set x-axis limits from 0 to 400

# Plot H1 errors - removed the 'o' marker
ax2.plot(neurons_20, h1_errors_20, 'b-', label=f'γ={gamma_20}')
ax2.plot(neurons_21, h1_errors_21, 'r-', label=f'γ={gamma_21}')
ax2.plot(neurons_22, h1_errors_22, 'g-', label=f'γ={gamma_22}')
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
plt.suptitle('Fitting with gradient: sparcity inducing effect', fontsize=14)
plt.tight_layout()

# Save the figure
plt.savefig('data_result/plot/Fitting with gradient: sparcity inducing effect.png', dpi=300)
print(f"Plot saved to data_result/plot/error_vs_neurons_gamma_comparison.png")

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
plt.suptitle('Error Metrics: Fitting With vs. Without Gradient (γ=0.01)', fontsize=14)
plt.tight_layout()

# Save the figure
plt.savefig('data_result/plot/Fitting With vs. Without Gradient.png', dpi=300)
print(f"Plot saved to data_result/plot/error_vs_neurons_gradient_comparison.png")

# =============================================================================
# NEW CODE FOR EXTRACTING INNER WEIGHTS AND PLOTTING 3D SPHERE
# =============================================================================

# Load pickle files
print("\nLoading pickle files for weight analysis...")
with open('data_result/weights/training_history_20.pkl', 'rb') as f:
    training_history_20 = pickle.load(f)

with open('data_result/weights/training_history_21.pkl', 'rb') as f:
    training_history_21 = pickle.load(f)

with open('data_result/weights/training_history_22.pkl', 'rb') as f:
    training_history_22 = pickle.load(f)

print("Successfully loaded pickle files")

# Extract inner weights and biases
weights_20_iter7 = None
biases_20_iter7 = None
weights_21_iter8 = None
biases_21_iter8 = None
weights_22_iter7 = None
biases_22_iter7 = None

if 7 < len(training_history_20['weights']):
    weights_20_iter7 = training_history_20['weights'][7]
    biases_20_iter7 = training_history_20['biases'][7]
    neurons_20_iter7 = training_history_20['neuron_count'][7]
    print(f"Extracted weights from training_history_20, iteration 7: {weights_20_iter7.shape}, neurons: {neurons_20_iter7}")
else:
    print(f"Error: training_history_20 doesn't have iteration 7. Max: {len(training_history_20['weights'])-1}")

if 8 < len(training_history_21['weights']):
    weights_21_iter8 = training_history_21['weights'][8]
    biases_21_iter8 = training_history_21['biases'][8]
    neurons_21_iter8 = training_history_21['neuron_count'][8]
    print(f"Extracted weights from training_history_21, iteration 8: {weights_21_iter8.shape}, neurons: {neurons_21_iter8}")
else:
    print(f"Error: training_history_21 doesn't have iteration 8. Max: {len(training_history_21['weights'])-1}")

if 7 < len(training_history_22['weights']):
    weights_22_iter7 = training_history_22['weights'][7]
    biases_22_iter7 = training_history_22['biases'][7]
    neurons_22_iter7 = training_history_22['neuron_count'][7]
    print(f"Extracted weights from training_history_22, iteration 7: {weights_22_iter7.shape}, neurons: {neurons_22_iter7}")
else:
    print(f"Error: training_history_22 doesn't have iteration 7. Max: {len(training_history_22['weights'])-1}")

# =============== POLAR COORDINATE VISUALIZATION  =================
if weights_20_iter7 is not None and weights_21_iter8 is not None and weights_22_iter7 is not None:
    # Extract the gamma values from the metadata files
    _, _, _, gamma_20, _ = extract_data("data_result/weights/weights_metadata_20.txt")
    _, _, _, gamma_21, _ = extract_data("data_result/weights/weights_metadata_21.txt")
    _, _, _, gamma_22, _ = extract_data("data_result/weights/weights_metadata_22.txt")
    
    # Create figure for polar coordinate plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': 'polar'})
    
    # 1. Polar plot for Training 20
    # Compute angles and radii in weight space (2D)
    xy_norms_20 = np.sqrt(weights_20_iter7[:, 0]**2 + weights_20_iter7[:, 1]**2)
    angles_20 = np.arctan2(weights_20_iter7[:, 1], weights_20_iter7[:, 0])
    
    # Plot in polar coordinates
    axes[0].scatter(angles_20, xy_norms_20, color='blue', alpha=0.8, s=60)
    axes[0].set_title(f'Weight Space in Polar Coordinates\nTraining 20 (γ={gamma_20})\nNeurons: {neurons_20_iter7}', fontsize=12)
    axes[0].grid(True, alpha=0.5)
    
    # 2. Polar plot for Training 21
    # Compute angles and radii in weight space (2D)
    xy_norms_21 = np.sqrt(weights_21_iter8[:, 0]**2 + weights_21_iter8[:, 1]**2)
    angles_21 = np.arctan2(weights_21_iter8[:, 1], weights_21_iter8[:, 0])
    
    # Plot in polar coordinates
    axes[1].scatter(angles_21, xy_norms_21, color='red', alpha=0.8, s=60)
    axes[1].set_title(f'Weight Space in Polar Coordinates\nTraining 21 (γ={gamma_21})\nNeurons: {neurons_21_iter8}', fontsize=12)
    axes[1].grid(True, alpha=0.5)
    
    # 3. Polar plot for Training 22
    # Compute angles and radii in weight space (2D)
    xy_norms_22 = np.sqrt(weights_22_iter7[:, 0]**2 + weights_22_iter7[:, 1]**2)
    angles_22 = np.arctan2(weights_22_iter7[:, 1], weights_22_iter7[:, 0])
    
    # Plot in polar coordinates
    axes[2].scatter(angles_22, xy_norms_22, color='green', alpha=0.8, s=60)
    axes[2].set_title(f'Weight Space in Polar Coordinates\nTraining 22 (γ={gamma_22})\nNeurons: {neurons_22_iter7}', fontsize=12)
    axes[2].grid(True, alpha=0.5)
    
    # Add a main title
    plt.suptitle('Analysis of Weights in Polar Coordinates at Optimal Iteration', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save the figure
    plt.savefig('data_result/plot/weights_polar_analysis.png', dpi=300)
    print(f"Polar coordinate analysis saved to data_result/plot/weights_polar_analysis.png")
    
    # Show plot
    plt.show()

# Show all plots
plt.show()

# =============================================================================
# NEW PLOT: COMPARING WEIGHTS_METADATA_14 AND WEIGHTS_METADATA_15
# =============================================================================

# Read data for comparison between gamma=0.01 and gamma=5.0
neurons_14, l2_errors_14, h1_errors_14, gamma_14, _ = extract_data("data_result/weights/weights_metadata_14.txt")
neurons_15, l2_errors_15, h1_errors_15, gamma_15, _ = extract_data("data_result/weights/weights_metadata_15.txt")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot L2 errors
ax1.plot(neurons_14, l2_errors_14, 'b-', label=f'γ={gamma_14}')
ax1.plot(neurons_15, l2_errors_15, 'r-', label=f'γ={gamma_15}')
ax1.set_xlabel('Number of Neurons')
ax1.set_ylabel('L2 Error')
ax1.set_title('L2 Error vs. Number of Neurons')
ax1.legend()
ax1.grid(True)
ax1.set_xlim(0, 55)  # Set x-axis limits based on data

# Plot H1 errors
ax2.plot(neurons_14, h1_errors_14, 'b-', label=f'γ={gamma_14}')
ax2.plot(neurons_15, h1_errors_15, 'r-', label=f'γ={gamma_15}')
ax2.set_xlabel('Number of Neurons')
ax2.set_ylabel('H1 Error')
ax2.set_title('H1 Error vs. Number of Neurons')
ax2.legend()
ax2.grid(True)
ax2.set_xlim(0, 55)  # Set x-axis limits based on data

# Set log scale for y-axis on both plots
ax1.set_yscale('log')
ax2.set_yscale('log')

# Add a main title
plt.suptitle('fitting without gradient: sparcity inducing effect', fontsize=14)
plt.tight_layout()

# Save the figure
plt.savefig('data_result/plot/fitting without gradient: sparcity inducing effect.png', dpi=300)
print(f"Plot saved to data_result/plot/error_vs_neurons_gamma_comparison_14_15.png")

plt.show()





