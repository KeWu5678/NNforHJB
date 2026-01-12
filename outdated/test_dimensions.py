import numpy as np

# Setup
N = 5  # Small for clarity
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

# Try the operation as written
try:
    A = (2 * Z) / (1 + s)
    print("\nA shape:", A.shape)
    print(A)
except Exception as e:
    print(f"\nError: {e}")

# Fix with reshape for broadcasting
s_reshaped = s.reshape(-1, 1)
print(f"\nReshaped sum shape: {s_reshaped.shape}")
print(s_reshaped)

# Try the corrected operation
A_fixed = (2 * Z) / (1 + s_reshaped)
print(f"\nFixed A shape: {A_fixed.shape}")
print(A_fixed) 