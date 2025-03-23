import numpy as np

# Setup
N = 5
R = 10
np.random.seed(42)

# Create random points
Z = np.random.rand(N, 2) * R
print(f"Z shape: {Z.shape}")

# Calculate sum of squares
s = np.sum(Z ** 2, axis=1)
print(f"Sum shape: {s.shape}")

# Calculate A and B (similar to original code)
# For A, reshape is needed
A = (2 * Z) / (1 + s.reshape(-1, 1))
print(f"A shape: {A.shape}")

# For B, no reshape needed if we're just going to column_stack
B = (1 - s) / (1 + s)
print(f"B shape: {B.shape}")

# Now column_stack them
W = np.column_stack((A, B))
print(f"W shape: {W.shape}")
print("First few rows of W:")
print(W[:3])

print("\nExplanation:")
print("1. A needs reshape because we're operating with Z (shape (N,2))")
print("2. B doesn't need reshape because (1-s)/(1+s) results in shape (N,)")
print("3. column_stack automatically converts 1D arrays to columns")
print("4. Final W has shape (N, 3): 2 columns from A, 1 from B") 