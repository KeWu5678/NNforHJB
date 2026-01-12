import numpy as np

# Create a sample 2D array
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print("Original array X:")
print(X)
print(f"Shape: {X.shape}")  # (3, 3)

# Summing along axis=1 (rows)
row_sums = np.sum(X, axis=1)
print("\nSum along axis=1 (rows):")
print(row_sums)
print(f"Shape: {row_sums.shape}")  # (3,)

# Why does NumPy return a 1D array (N,) instead of (N,1)?
print("\nWhy does NumPy return a 1D (N,) array?")
print("1. Historical reasons: NumPy was designed to reduce dimensions when possible")
print("2. Convenience: Makes it easier to index results with a single index")
print("3. Memory efficiency: Stores only the necessary data")

# Broadcasting example showing why reshape is needed
try:
    # Try to multiply the original 2D array with row sums
    result = X * row_sums
    print("\nThis shouldn't work...")
except ValueError as e:
    print(f"\nError when multiplying X * row_sums: {e}")

# Fix with reshape
row_sums_reshaped = row_sums.reshape(-1, 1)
print("\nAfter reshape to (N,1):")
print(row_sums_reshaped)
print(f"Shape: {row_sums_reshaped.shape}")

# Now it works with broadcasting
result = X * row_sums_reshaped
print("\nMultiplying X * row_sums_reshaped works:")
print(result)
print(f"Shape: {result.shape}")

# Matrix operations demonstration
A = np.array([[1, 2], [3, 4], [5, 6]])  # Shape (3, 2)
b = np.array([10, 20, 30])  # Shape (3,)

print("\nMatrix operations:")
print(f"A shape: {A.shape}")
print(f"b shape: {b.shape}")

# For certain operations, 1D arrays are treated specially:
try:
    # Matrix multiplication with 1D array is treated as a special case
    # This works without reshape for matrix operations
    c = A.T @ b  # Equivalent to np.matmul(A.T, b)
    print(f"\nA.T @ b result: {c}")
    print(f"Shape: {c.shape}")
except ValueError as e:
    print(f"\nError in matrix multiplication: {e}")

# Why this inconsistency?
print("\nWhy the inconsistency in NumPy?")
print("1. Different operations have different broadcasting rules")
print("2. Matrix operations (@, dot, matmul) have special cases for 1D arrays")
print("3. Element-wise operations (*, +, etc.) follow strict broadcasting rules")

# Row vs column vector
print("\nRow vs Column Vectors:")
row_vector = np.array([1, 2, 3])  # Shape (3,)
col_vector = row_vector.reshape(-1, 1)  # Shape (3, 1)
print(f"1D array shape: {row_vector.shape}")
print(f"Column vector shape: {col_vector.shape}")
print(f"Transpose of column vector shape: {col_vector.T.shape}")  # (1, 3)

print("\nConclusion:")
print("- NumPy operations that reduce dimensions return 1D arrays to save memory")
print("- For element-wise operations with 2D arrays, you need to reshape 1D arrays")
print("- Matrix operations have special handling for 1D arrays")
print("- It's often safer to explicitly reshape to ensure correct broadcasting") 