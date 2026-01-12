import numpy as np
import torch
from unittest.mock import MagicMock

# Function to test gradient calculation
def test_gradient():
    # Create synthetic data
    K = 10  # Number of data points
    X_train = np.random.uniform(-1, 1, (K, 2))
    V_train = np.random.uniform(-1, 1, (K, 1))
    dV_train = np.random.uniform(-1, 1, (K, 2))
    
    # Create a mock model
    model = MagicMock()
    model.predict.return_value = V_train + 0.1  # Add small difference from V_train
    
    # Mock the gradient operator
    def gradient_op(inputs, outputs):
        # Just return random gradients
        return torch.rand_like(inputs)
    
    # Function to get predicted gradients
    def get_dV_pred(x):
        return dV_train + 0.1  # Add small difference from dV_train
    
    # ReLU function
    def relu(x):
        return np.maximum(0, x)
    
    # Stereo function (from your code)
    def stereo(z):
        """
        here z is given as a (2, 1) np.array
        return shape: (2, 1), (1, 1)
        """
        denominator = (1 + np.sum(z**2, axis=0)).reshape(1, -1)
        return [
            (2 * z) / denominator, 
            (1 - np.sum(z**2, axis=0)) / denominator
            ]
    
    # Gradient function (similar to yours)
    def gradient(z, p=2):
        z = z.reshape(2, 1)
        print(f"\nTesting gradient with z = {z.flatten()}")
        
        a, b = stereo(z)  # shape = (2, 1), (1, 1)
        print(f"a = {a.flatten()}, b = {b.flatten()}")
        
        # Calculate activations and print stats
        activations = X_train @ a + b
        print(f"activations: min={np.min(activations):.4f}, max={np.max(activations):.4f}")
        print(f"positive activations: {np.sum(activations > 0)} out of {K}")
        
        kernel = relu(activations)  # shape = (K, 1)
        print(f"non-zero kernel elements: {np.count_nonzero(kernel)} out of {K}")
        
        # Get predictions
        v_pred = model.predict(X_train)  # shape = (K, 1)
        print(f"V_train to v_pred difference: {np.linalg.norm(V_train - v_pred):.4f}")
        
        # Calculate first term
        term1 = np.sum(kernel ** p * (v_pred - V_train))
        print(f"term1 = {term1:.4f}")
        
        # Calculate second term
        multi_kernel = np.repeat(p * (kernel ** (p - 1)), 2, axis=1)  # shape = (K, 2)
        dV_pred = get_dV_pred(X_train)  # shape = (K, 2)
        
        # Two ways to handle the repeat
        try:
            # Try the original way
            coeff_orig = (dV_pred - dV_train) * a.T.repeat(K, axis=0)  # shape = (K, 2)
            term2_orig = np.sum(coeff_orig * multi_kernel)
            print(f"term2 (original) = {term2_orig:.4f}")
        except Exception as e:
            print(f"Original repeat failed: {e}")
        
        # Alternative using tile
        coeff_tile = (dV_pred - dV_train) * np.tile(a.T, (K, 1))  # shape = (K, 2)
        term2_tile = np.sum(coeff_tile * multi_kernel)
        print(f"term2 (tile) = {term2_tile:.4f}")
        
        # Calculate final result
        result_orig = term1 + (term2_orig if 'term2_orig' in locals() else 0)
        result_tile = term1 + term2_tile
        
        print(f"Final gradient (original): {result_orig:.4f}")
        print(f"Final gradient (tile): {result_tile:.4f}")
        
        return result_tile
    
    # Test with different z values
    z_values = [
        np.array([0.1, 0.1]),    # Small values
        np.array([10.0, 10.0]),  # Large values
        np.array([0.5, -0.5]),   # Mixed signs
        np.array([50.0, -20.0])  # Extreme values
    ]
    
    # Try all the values
    for z in z_values:
        grad = gradient(z)
        print(f"Gradient result for z={z}: {grad:.6f}\n" + "-"*50)
    
    # Try with b shifted to be more positive
    print("\nTrying with shifted b values:")
    
    def stereo_shifted(z):
        denominator = (1 + np.sum(z**2, axis=0)).reshape(1, -1)
        a = (2 * z) / denominator
        b_orig = (1 - np.sum(z**2, axis=0)) / denominator
        b = b_orig + 0.5  # Shift to make more positive
        return a, b
    
    def gradient_shifted(z, p=2):
        z = z.reshape(2, 1)
        print(f"\nTesting shifted gradient with z = {z.flatten()}")
        
        a, b = stereo_shifted(z)
        print(f"a = {a.flatten()}, b = {b.flatten()}")
        
        activations = X_train @ a + b
        print(f"activations: min={np.min(activations):.4f}, max={np.max(activations):.4f}")
        print(f"positive activations: {np.sum(activations > 0)} out of {K}")
        
        kernel = relu(activations)
        print(f"non-zero kernel elements: {np.count_nonzero(kernel)} out of {K}")
        
        v_pred = model.predict(X_train)
        term1 = np.sum(kernel ** p * (v_pred - V_train))
        
        multi_kernel = np.repeat(p * (kernel ** (p - 1)), 2, axis=1)
        dV_pred = get_dV_pred(X_train)
        
        coeff = (dV_pred - dV_train) * np.tile(a.T, (K, 1))
        term2 = np.sum(coeff * multi_kernel)
        
        result = term1 + term2
        print(f"Final gradient (shifted): {result:.4f}")
        
        return result
    
    # Test the shifted version with the same z values
    for z in z_values:
        grad = gradient_shifted(z)
        print(f"Shifted gradient result for z={z}: {grad:.6f}\n" + "-"*50)

if __name__ == "__main__":
    test_gradient() 