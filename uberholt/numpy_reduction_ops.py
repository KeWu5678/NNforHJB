import numpy as np

# Create a sample 2D array
X = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

print(f"Original array X shape: {X.shape}")
print(X)

print("\nDemonstrating various NumPy reduction operations along axis=1 (rows):")
print("-" * 70)

# Sum
row_sums = np.sum(X, axis=1)
print(f"np.sum(X, axis=1) shape: {row_sums.shape}")
print(f"Values: {row_sums}")

# Mean
row_means = np.mean(X, axis=1)
print(f"\nnp.mean(X, axis=1) shape: {row_means.shape}")
print(f"Values: {row_means}")

# Max
row_max = np.max(X, axis=1)
print(f"\nnp.max(X, axis=1) shape: {row_max.shape}")
print(f"Values: {row_max}")

# Min
row_min = np.min(X, axis=1)
print(f"\nnp.min(X, axis=1) shape: {row_min.shape}")
print(f"Values: {row_min}")

# Standard deviation
row_std = np.std(X, axis=1)
print(f"\nnp.std(X, axis=1) shape: {row_std.shape}")
print(f"Values: {row_std}")

# Product
row_prod = np.prod(X, axis=1)
print(f"\nnp.prod(X, axis=1) shape: {row_prod.shape}")
print(f"Values: {row_prod}")

# Variance
row_var = np.var(X, axis=1)
print(f"\nnp.var(X, axis=1) shape: {row_var.shape}")
print(f"Values: {row_var}")

# Median
row_median = np.median(X, axis=1)
print(f"\nnp.median(X, axis=1) shape: {row_median.shape}")
print(f"Values: {row_median}")

print("\nTesting non-reduction operations:")
print("-" * 70)

# Cumulative sum (doesn't reduce dimension)
row_cumsum = np.cumsum(X, axis=1)
print(f"np.cumsum(X, axis=1) shape: {row_cumsum.shape}")
print(row_cumsum)

print("\nOperations on arrays of different dimensions:")
print("-" * 70)

# 3D array example
Y = np.arange(24).reshape(2, 3, 4)
print(f"3D array Y shape: {Y.shape}")
print(Y)

# Sum along different axes
sum_axis0 = np.sum(Y, axis=0)
sum_axis1 = np.sum(Y, axis=1)
sum_axis2 = np.sum(Y, axis=2)

print(f"\nnp.sum(Y, axis=0) shape: {sum_axis0.shape}")
print(f"\nnp.sum(Y, axis=1) shape: {sum_axis1.shape}")
print(f"\nnp.sum(Y, axis=2) shape: {sum_axis2.shape}")

print("\nConclusion:")
print("-" * 70)
print("1. Almost all NumPy reduction operations return a 1D array with shape (N,)")
print("2. The dimension you operate along is removed from the output shape")
print("3. When using the result with operations that require specific shapes for broadcasting,")
print("   you'll need to reshape the result to make it explicit (e.g., reshape(-1, 1))")
print("4. Non-reduction operations (like cumsum) preserve dimensions") 