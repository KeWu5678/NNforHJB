import numpy as np

# Setup
N = 5
R = 10
np.random.seed(42)

# Create random points
Z = np.random.rand(N, 2) * R
print(f"Z shape: {Z.shape}")
print(Z)

# Calculate sum of squares
s = np.sum(Z ** 2, axis=1)
print(f"\nSum shape: {s.shape}")
print(s)

# Try B calculation without reshape
try:
    B = (1 - s) / (1 + s)
    print("\nB shape (without reshape):", B.shape)
    print(B)
    print("Note: This works because both numerator and denominator are shape (N,)")
except Exception as e:
    print(f"\nError: {e}")

# Trying to use B with Z (where broadcasting matters)
try:
    # This would fail if we needed B to interact with Z
    result = Z * B
    print("\nZ * B shape:", result.shape)
    print("First few results:", result[:2])
except Exception as e:
    print(f"\nError when using with Z: {e}")
    
# Now with reshape
B_reshaped = (1 - s) / (1 + s)
B_reshaped = B_reshaped.reshape(-1, 1)
print(f"\nB reshaped shape: {B_reshaped.shape}")

# This will work with broadcasting
result_fixed = Z * B_reshaped
print(f"\nZ * B_reshaped shape: {result_fixed.shape}")
print(result_fixed) 