import numpy as np

def stereo(z):
    """
    Here z is given as a (2, 1) np.array
    Return shape: (2, 1), (1, 1)
    """
    print(f"Input z shape: {z.shape}")
    denominator = (1 + np.sum(z**2, axis=0)).reshape(1, -1)  # Reshape to (1, n) where n is number of points
    
    a = (2 * z) / denominator  # This should maintain z's shape (2, n)
    b = (1 - np.sum(z**2, axis=0)) / denominator  # This will be (1, n)
    
    print(f"Output a shape: {a.shape}, b shape: {b.shape}")
    return a, b

# Test with a (2, 1) array
test_z = np.array([[1], [2]])
print("Testing with shape (2, 1):")
a, b = stereo(test_z)
print(f"a: {a}")
print(f"b: {b}")
print("\n")

# Test with multiple points
test_z2 = np.array([[1, 3], [2, 4]]) # 2 points with coordinates (1,2) and (3,4)
print("Testing with shape (2, 2) for multiple points:")
a, b = stereo(test_z2)
print(f"a: {a}")
print(f"b: {b}")
print("\n")

# Test matrix multiplication with the output
# Simulate X_train @ a + b
X_train = np.array([[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]])  # K=3, features=2
print(f"X_train shape: {X_train.shape}")

a_single, b_single = stereo(test_z[:, 0:1])  # Use first point only
print(f"Matrix multiplication X_train @ a_single shape: {(X_train @ a_single).shape}")
print(f"Result after adding b_single: {(X_train @ a_single + b_single).shape}")
print("\n") 