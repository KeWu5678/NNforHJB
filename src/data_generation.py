import numpy as np
import utils
import openloop_optimizer as op


# Define structured array dtype
dtype = [
    ('x', '2float64'),  # 2D float array for the first element
    ('dv', '2float64'), # 2D float array for the second element
    ('v', 'float64')    # 1D float for the third element
]

# Create structured array


"""=====================THE ODE======================"""
beta = 0.1

def VDP(t, y, u):
    """Define the Boundary value problem with control parameter u"""
    return np.vstack([
        y[1],
        -y[0] + y[1] * (1 - y[0] ** 2) + u(t),
        - 2 * y[0] + y[3] * (2 * y[0] * y[1] + 1),
        - 2 * y[1] - y[2] - y[3] * (1 -y[0] ** 2)
    ])

def bc(ya, yb):
    """Boundary conditions"""
    return np.array([
        ya[0] - ini[0],
        ya[1] - ini[1],
        yb[2],
        yb[3]
    ])

def gradient(u, p):
    if len(p) != len(u):
        raise ValueError("p and u must have the same length")
    else:
        n = len(p)
        grad = np.zeros(n)
    for i in range(n):
        grad[i] = p[i] + 2 * beta * u[i]
    return grad

def V(grid, u, y1, y2):
    return utils.L2(grid, u) * beta + 0.5 * (utils.L2(grid, y1) + utils.L2(grid, y2))

def gen_bc(ini_val):
    """Generate boundary conditions based on initial values"""
    def bc_func(ya, yb):
        return np.array([
            ya[0] - ini_val[0],
            ya[1] - ini_val[1],
            yb[2],
            yb[3]
        ])
    return bc_func


"""=====================DATA GENERATION======================"""
grid = np.linspace(0, 3, 1000)
guess = np.ones((4, grid.size))
tol = 1e-5
max_it = 500

# Define the range for each dimension
x1_min, x1_max = -3.0, 3.0  # Range for first component
x2_min, x2_max = -3.0, 3.0  # Range for second component

# Create a 60x60 grid but exclude the points that would be in the 30x30 grid

# First create a 60x60 grid
x1_full = np.linspace(x1_min, x1_max, 60)
x2_full = np.linspace(x2_min, x2_max, 60)

# The 30x30 grid points would be at every other index in the 60x60 grid
# Create boolean masks to identify points to keep (True) or exclude (False)
x1_mask = np.ones(60, dtype=bool)
x2_mask = np.ones(60, dtype=bool)

# Set every other index (0, 2, 4, ...) to False to exclude those points
x1_mask[::2] = False
x2_mask[::2] = False

# Create arrays with only the points we want to keep
x1_values = x1_full[~x1_mask]  # These are the odd-indexed points (1, 3, 5, ...)
x2_values = x2_full[~x2_mask]  # These are the odd-indexed points (1, 3, 5, ...)

# For verification, create the 30x30 grid we're excluding
x1_excluded = x1_full[::2]  # These are the even-indexed points (0, 2, 4, ...)
x2_excluded = x2_full[::2]  # These are the even-indexed points (0, 2, 4, ...)

print(f"Full 60x60 grid points in x1: {len(x1_full)}")
print(f"Excluded points in x1 (30x30 grid): {len(x1_excluded)}")
print(f"Remaining points in x1: {len(x1_values)}")

# Create a meshgrid of the points we're keeping
X1, X2 = np.meshgrid(x1_values, x2_values)

# Reshape the meshgrid to get a list of all combinations
x0_values = np.column_stack((X1.flatten(), X2.flatten()))

# Create a meshgrid of the points we're excluding for verification
X1_excl, X2_excl = np.meshgrid(x1_excluded, x2_excluded)
x0_excluded = np.column_stack((X1_excl.flatten(), X2_excl.flatten()))

# Verify that none of our points match the excluded points
min_distances = []
for point in x0_values[:10]:  # Check just a subset for efficiency
    distances = np.sqrt(np.sum((x0_excluded - point)**2, axis=1))
    min_distances.append(np.min(distances))

min_distance = np.min(min_distances)
print(f"Minimum distance between new and excluded points: {min_distance:.6f}")
print(f"This should be greater than zero if points are properly excluded")

# Total number of grid points
N = len(x0_values)
print(f"Created a {len(x1_values)}x{len(x2_values)} grid with {N} points")

# Initialize dataset
dataset = np.zeros(N, dtype=dtype)

# Track failed initial conditions
failed_ini = []

print(f"N = {N}")
for i in range(N):
    # Get initial condition from the grid
    ini = x0_values[i]
    
    # Generate boundary conditions
    bc_func = gen_bc(ini)
    
    print(f"i = {i}")
    print(f"ini = {ini}")
    dv, v = op.OpenLoopOptimizer(VDP, bc_func, V, gradient, grid, guess, tol, max_it).optimize()
    if dv is not None and not np.isnan(v):
        dataset[i] = (ini, dv, v)
    else:
        # Store failed initial condition
        failed_ini.append(ini)
        print(f"Failed to converge for ini = {ini}")
    
# Use a filename that indicates grid sampling
output_file = "VDP_beta_0.1_grid_30x30_odd_indices.npy"

# Save failed initial conditions
failed_output_file = "VDP_beta_0.1_failed_ini_30x30_odd_indices.npy"
if failed_ini:
    failed_ini = np.array(failed_ini)
    print(f"Saving {len(failed_ini)} failed initial conditions to {failed_output_file}")
    np.save(failed_output_file, failed_ini)
else:
    print("All initial conditions converged successfully")

print(f"Saving results to {output_file}")
np.save(output_file, dataset)