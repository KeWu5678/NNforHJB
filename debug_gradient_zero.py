import numpy as np
import torch
from unittest.mock import MagicMock

def stereo(z):
    """
    Standard stereographic projection function.
    z shape: (2, n)
    returns: a (2, n), b (1, n)
    """
    denominator = (1 + np.sum(z**2, axis=0)).reshape(1, -1)
    a = (2 * z) / denominator
    b = (1 - np.sum(z**2, axis=0)) / denominator
    return a, b

def inverse_stereo(a, b):
    """
    Inverse stereographic projection: Map from the unit ball (a, b) back to R^2 (z)
    a shape: (2, n)
    b shape: (1, n)
    returns: z (2, n)
    """
    b = b.reshape(1, -1)
    return a / (1 + b)

def sample_z_inverse(num_samples=1):
    """
    Sample z values using the inverse method for uniform distribution on the sphere
    """
    # Sample uniformly on unit sphere
    theta = np.random.uniform(0, 2*np.pi, num_samples)
    phi = np.arccos(2 * np.random.uniform(0, 1, num_samples) - 1)
    
    # Convert to 3D Cartesian coordinates
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    # Convert to a, b representation
    a = np.vstack([x, y])
    b = z.reshape(1, -1)
    
    # Get z using inverse projection
    return inverse_stereo(a, b)

def debug_gradient():
    """
    Debug why gradient is returning 0
    """
    # Generate synthetic data
    np.random.seed(42)
    K = 10  # Number of data points
    X_train = np.random.uniform(-1, 1, (K, 2))
    V_train = np.random.uniform(-1, 1, (K, 1))
    dV_train = np.random.uniform(-1, 1, (K, 2))
    
    # Create a mock model with a deliberate difference from V_train
    # This ensures v_pred - V_train is non-zero
    model = MagicMock()
    v_pred_values = V_train + 0.2 * np.random.uniform(-1, 1, (K, 1))
    model.predict.return_value = v_pred_values
    
    # Function to get predicted gradients with deliberate difference from dV_train
    def get_dV_pred(x):
        return dV_train + 0.2 * np.random.uniform(-1, 1, (K, 2))
    
    # ReLU function
    def relu(x):
        return np.maximum(0, x)
    
    # Test gradient calculation with original gradient function
    def gradient_orig(z, p=2):
        """Original gradient function (similar to your code)"""
        z = z.reshape(2, 1)
        print(f"\nOriginal gradient with z = {z.flatten()}")
        
        a, b = stereo(z)
        print(f"a shape: {a.shape}, b shape: {b.shape}")
        print(f"a = {a.flatten()}, b = {b.flatten()}")
        
        # Calculate activations and print stats
        activations = X_train @ a + b
        print(f"activations shape: {activations.shape}")
        print(f"activations: {activations.flatten()}")
        print(f"activations: min={np.min(activations):.4f}, max={np.max(activations):.4f}")
        print(f"positive activations: {np.sum(activations > 0)} out of {K}")
        
        # Apply ReLU
        kernel = relu(activations)
        print(f"kernel values after ReLU: {kernel.flatten()}")
        print(f"non-zero kernel elements: {np.count_nonzero(kernel)} out of {K}")
        
        if np.count_nonzero(kernel) == 0:
            print("⚠️ All kernel values are zero after ReLU! This is why gradient=0")
            print("   This happens because all activations are negative")
            return 0
        
        # Get predictions
        v_pred = model.predict(X_train)
        print(f"V_train to v_pred difference: {np.linalg.norm(V_train - v_pred):.4f}")
        
        # Calculate term1
        diff = (v_pred - V_train)
        print(f"v_pred - V_train: {diff.flatten()}")
        term1 = np.sum(kernel ** p * diff)
        print(f"term1 = {term1:.4f}")
        
        # Calculate term2
        dV_pred = get_dV_pred(X_train)
        print(f"dV_pred shape: {dV_pred.shape}")
        print(f"dV_train shape: {dV_train.shape}")
        print(f"Norm of dV_pred - dV_train: {np.linalg.norm(dV_pred - dV_train):.4f}")
        
        coeff = (dV_pred - dV_train) * np.tile(a.T, (K, 1))
        multi_kernel = np.repeat(p * (kernel ** (p - 1)), 2, axis=1)
        print(f"coeff shape: {coeff.shape}, multi_kernel shape: {multi_kernel.shape}")
        
        term2 = np.sum(coeff * multi_kernel)
        print(f"term2 = {term2:.4f}")
        
        # Final result
        result = term1 + term2
        print(f"Final gradient = {result:.4f}")
        return result
    
    # Modified gradient function to try different approaches
    def gradient_mod(z, p=2, shift_b=0.5):
        """Modified gradient with b shift to get more positive activations"""
        z = z.reshape(2, 1)
        print(f"\nModified gradient with z = {z.flatten()}, b_shift = {shift_b}")
        
        a, b = stereo(z)
        # Shift b to make more activations positive
        b = b + shift_b
        print(f"a = {a.flatten()}, b = {b.flatten()} (after shift)")
        
        # Calculate activations
        activations = X_train @ a + b
        print(f"activations: min={np.min(activations):.4f}, max={np.max(activations):.4f}")
        print(f"positive activations: {np.sum(activations > 0)} out of {K}")
        
        kernel = relu(activations)
        print(f"non-zero kernel elements: {np.count_nonzero(kernel)} out of {K}")
        
        if np.count_nonzero(kernel) == 0:
            print("⚠️ All kernel values are zero after ReLU! This is why gradient=0")
            return 0
        
        v_pred = model.predict(X_train)
        term1 = np.sum(kernel ** p * (v_pred - V_train))
        
        dV_pred = get_dV_pred(X_train)
        coeff = (dV_pred - dV_train) * np.tile(a.T, (K, 1))
        multi_kernel = np.repeat(p * (kernel ** (p - 1)), 2, axis=1)
        term2 = np.sum(coeff * multi_kernel)
        
        result = term1 + term2
        print(f"Final gradient = {result:.4f}")
        return result

    # Test different z values
    print("TESTING WITH UNIFORM SAMPLING (likely problematic):")
    z_values_uniform = [
        np.random.uniform(-1, 1, (2, 1)),
        np.random.uniform(-3, 3, (2, 1)),
        np.random.uniform(-5, 5, (2, 1)),
    ]
    
    for z in z_values_uniform:
        gradient_orig(z)
    
    print("\nTESTING WITH INVERSE SAMPLING (should be better):")
    z_values_inverse = sample_z_inverse(3)
    
    for i in range(3):
        z = z_values_inverse[:, i:i+1]
        gradient_orig(z)
    
    print("\nTESTING WITH B SHIFT PARAMETER:")
    # Try with different b shifts to make activations positive
    b_shifts = [0.2, 0.5, 1.0, 2.0]
    z = sample_z_inverse(1)
    
    for shift in b_shifts:
        gradient_mod(z, shift_b=shift)
    
    print("\nDETAILED RECOMMENDATIONS:")
    print("1. If your gradient is zero, it's likely because all activations (X_train @ a + b) are negative.")
    print("2. With negative activations, ReLU outputs all zeros, making the entire gradient zero.")
    print("3. Solutions:")
    print("   a. Use the inverse sampling method for z (implemented in this script)")
    print("   b. Add a shift to b in your stereo function: b = b + shift_value")
    print("   c. Normalize your input data (X_train) if its range is causing issues")
    print("4. Implementation:")
    print("   - Start with b_shift = 0.5 and adjust based on your data")
    print("   - Print activation statistics to verify you have >0 positive activations")
    
    print("\nPROPOSED FIX FOR YOUR CODE:")
    print("def stereo(z):")
    print("    denominator = (1 + np.sum(z**2, axis=0)).reshape(1, -1)")
    print("    a = (2 * z) / denominator")
    print("    b = (1 - np.sum(z**2, axis=0)) / denominator")
    print("    return a, b")

if __name__ == "__main__":
    debug_gradient() 