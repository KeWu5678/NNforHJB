import numpy as np

# Create a 1D array
a = np.array([1, 2, 3, 4, 5])
print(f"Original array: {a}")
print(f"Shape: {a.shape}")  # Will show (5,)

# Reshape to column vector
a_col = a.reshape(-1, 1)
print(f"\nAfter reshape(-1, 1):\n{a_col}")
print(f"New shape: {a_col.shape}")  # Will show (5, 1)

# The -1 is a "wildcard" that tells NumPy to calculate that dimension automatically
print("\nExplanation of the -1 parameter:")
print("- The -1 tells NumPy: 'figure out this dimension based on the array size'")
print("- reshape(-1, 1) means: 'make this a 2D array with 1 column and calculate the rows'")

# Another example starting with a larger array
b = np.arange(10)
print(f"\nAnother array: {b}")
print(f"Shape: {b.shape}")  # Will show (10,)

# Various reshapes
print("\nDifferent reshape examples:")
print(f"reshape(2, 5):\n{b.reshape(2, 5)}")  # Explicitly 2 rows, 5 columns
print(f"reshape(-1, 2):\n{b.reshape(-1, 2)}")  # 2 columns, rows calculated (5)
print(f"reshape(5, -1):\n{b.reshape(5, -1)}")  # 5 rows, columns calculated (2)

# Broadcasting example with reshape(-1, 1)
print("\nBroadcasting example:")
x = np.array([1, 2, 3])  # 1D array
y = np.array([[4], [5], [6]])  # Column vector

try:
    # This will fail - can't broadcast 1D with 2D properly for certain operations
    result1 = x * y
    print(f"x * y = {result1}")
except Exception as e:
    print(f"Error without reshape: {e}")

# With reshape
x_col = x.reshape(-1, 1)  # Now a column vector
print(f"\nx reshaped:\n{x_col}")
result2 = x_col * y  # This works with broadcasting
print(f"\nx_col * y =\n{result2}") 