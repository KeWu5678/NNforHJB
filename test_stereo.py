import numpy as np

def stereo(z):
    """
    Here z is given as a (2, 1) np.array
    Return shape: (2, 1), (1, 1)
    """
    print(f"Input z shape: {z.shape}")
    
    a = (2 * z) / (1 + np.sum(z**2, axis=1))
    b = (1 - np.sum(z**2, axis=1)) / (1 + np.sum(z**2, axis=1))
    
    print(f"a shape: {a.shape}")
    print(f"b shape: {b.shape}")
    
    return [a, b]

# Test with a (2, 1) array
test_z = np.array([[1], [2]])
print("Testing with shape (2, 1):")
a, b = stereo(test_z)
print("\n")

# Let's test with a different shape to understand what's happening
test_z2 = np.array([[1, 2]]).T  # This is also a (2, 1) array but created differently
print("Testing with shape (2, 1) created differently:")
a, b = stereo(test_z2)
print("\n")

# Test with shape for multiple points case
test_z3 = np.random.randn(2, 5)  # This would be like having 5 points
print("Testing with shape (2, 5) for multiple points:")
a, b = stereo(test_z3)
print("\n") 