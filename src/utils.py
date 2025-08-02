import numpy as np
import scipy.special as sp
from numpy.polynomial.chebyshev import Chebyshev
from scipy.spatial import KDTree
import torch

def gen(identifier, coeffs, geom):
    """
    args:
        identifier: str
            "cheby"
            "legendre"
            "bspline"
    """
    if identifier == "legendre":
        n = len(coeffs)
        def f(x):
            return sum(coeffs[i] * sp.legendre(i)(x) for i in range(n))
        return f
    
    """In the case of piecewise interpolation, geom should be the grid"""
    if identifier == "pw":
        def f(x):
            conditions = [(x >= geom[i]) & (x < geom[i+1]) for i in range(len(geom) - 1)]
            return np.piecewise(x, conditions, coeffs)
    return f

def project(identifier, geom_old, geom_new, f_old):
    """
    Given old parametrization of f, return new parametrization of f
    """
    if identifier == "pw":
        f_new = np.interp(geom_new, geom_old, f_old)
        return f_new
    

def L2(grid, x):
    """
    Input:
        x:  1 x grid.size
            Coefficients of the vector
    Output:
            L2 norm of x
    """
    
    if len(grid) != len(x):
        raise ValueError("grid and x must have the same size")
    else:
        n = len(grid)
    norm = 0
    for i in range(n - 1):
        mesh = grid[i + 1] - grid[i]
        norm += mesh * (x[i] ** 2)
    return norm

# Define a custom H1 error metric for DeepXDE
# def h1_error(y_true, y_pred, x, model):
#     """
#     Calculate H1 error (L2 error of values + L2 error of gradients)
    
#     Args:
#         y_true: True values
#         y_pred: Predicted values
#         x: Input points
#         model: The DeepXDE model
#     """
#     # L2 error of values - already computed as MSE by default
#     value_error = dde.metrics.mean_squared_error(y_true, y_pred)
    
#     # Get gradients from auxiliary variables (true gradients)
#     grad_true = model.data.test_aux_vars
    
#     # Calculate predicted gradients using autodiff
#     inputs = dde.backend.as_tensor(x)
#     inputs.requires_grad_()
#     outputs = model.net(inputs)
    
#     # Calculate gradient with respect to inputs
#     grad_pred = []
#     for i in range(2):  # For 2D problem
#         grad_i = dde.grad.jacobian(outputs, inputs, i=0, j=i)
#         grad_pred.append(grad_i)
    
#     grad_pred = torch.cat(grad_pred, dim=1)
    
#     # Calculate L2 error of gradients
#     gradient_error = torch.mean((grad_pred - torch.tensor(grad_true))**2)
    
#     # H1 error = L2 error of values + L2 error of gradients
#     h1_error = value_error + gradient_error
    
#     # Round to 8 digits
#     return round(h1_error.item(), 8)

def remove_duplicates(points, tolerance=1e-3):
    """
    Remove duplicate points from a numpy array of shape (n, d)
    """
    if len(points) <= 1:
        return points
    tree = KDTree(points)
    pairs = tree.query_pairs(tolerance)
    if not pairs:
        return points
    duplicate_indices = set()
    for i, j in pairs:
        duplicate_indices.add(j)
    unique_indices = [i for i in range(len(points)) if i not in duplicate_indices]
    return points[unique_indices]

def gen_legendre(coefficients, domain=(0, 3)):
    """
    Create a function using Legendre polynomial basis on a custom interval.
    
    Args:
        coefficients: List/array of coefficients for Legendre polynomials
        domain: Domain interval (default: [0, 3])
        
    Returns:
        function: A function that evaluates the Legendre expansion at any point t
    """
    n = len(coefficients)
    a, b = domain
    
    def func(t):
        # Map from custom domain to [-1, 1] which is the standard domain for Legendre polynomials
        t_mapped = (2*t - (a+b)) / (b-a)
        
        # Handle arrays properly
        if isinstance(t, np.ndarray):
            result = np.zeros_like(t, dtype=float)
            for i in range(n):
                P_i = sp.legendre(i)
                result += coefficients[i] * P_i(t_mapped)
            return result
        else:
            # Evaluate the sum of Legendre polynomials with given coefficients
            result = 0.0
            for i in range(n):
                P_i = sp.legendre(i)
                result += coefficients[i] * P_i(t_mapped)
            return result
    
    return func

def fit_legendre(x_data, y_data, n, domain=(0, 3), report_error=False):
    """
    Fit discrete data points with n Legendre basis functions.
    
    Args:
        x_data: Array of x-coordinates of data points
        y_data: Array of y-coordinates of data points
        n: Number of Legendre basis functions to use
        domain: Domain interval (default: [-3, 3])
        report_error: Whether to print fit quality metrics (default: False)
        
    Returns:
        coefficients: Array of coefficients for Legendre polynomials
    """
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length")
    
    # Map data to standard Legendre domain [-1, 1]
    a, b = domain
    x_mapped = (2*x_data - (a+b)) / (b-a)
    
    # Create design matrix A where A[i,j] = P_j(x_i)
    A = np.zeros((len(x_data), n))
    for j in range(n):
        P_j = sp.legendre(j)
        A[:, j] = P_j(x_mapped)
    
    # Solve the least squares problem: min ||A*coeffs - y_data||^2
    coeffs, residuals, rank, s = np.linalg.lstsq(A, y_data, rcond=None)
    
    # Report fit quality if requested
    if report_error:
        # Calculate fitted values
        y_fit = A @ coeffs
        
        # Calculate errors
        errors = y_data - y_fit
        max_error = np.max(np.abs(errors))
        mean_error = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        
        # Calculate R² (coefficient of determination)
        ss_total = np.sum((y_data - np.mean(y_data))**2)
        ss_residual = np.sum(errors**2)
        r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
        
        print(f"Legendre fit with {n} basis functions:")
        print(f"  Max absolute error: {max_error:.6e}")
        print(f"  Mean absolute error: {mean_error:.6e}")
        print(f"  RMSE: {rmse:.6e}")
        print(f"  R² (coefficient of determination): {r_squared:.6f}")
        if len(residuals) > 0:
            print(f"  Sum of squared residuals: {residuals[0]:.6e}")
        print(f"  Matrix rank: {rank}")
        
        # Check if potentially underdetermined
        if rank < min(A.shape):
            print("  Warning: Matrix is rank deficient. Consider using fewer basis functions.")
        
        # Check for potential numerical issues
        condition_number = np.linalg.cond(A)
        if condition_number > 1e10:
            print(f"  Warning: Design matrix is ill-conditioned (condition number: {condition_number:.2e}).")
            print("  Consider using fewer basis functions or regularization.")
        
    return coeffs

# def hyperbolic_cross_indices(d, n, s=4):
#     """
#     Generate hyperbolic cross index set for d-dimensional polynomials.
    
#     The hyperbolic cross index set is defined as:
#     J = J(s) = { i = (i₁, i₂, ..., iₙ) ∈ ℕ₀ⁿ : ∏(iⱼ + 1) ≤ s + 1 }
    
#     Args:
#         d: Dimension of the approximation space (n in the formula)
#         n: Maximum polynomial degree to consider (if needed)
#         s: Parameter controlling the cardinality of the index set
        
#     Returns:
#         indices: List of d-tuples, each containing indices for the hyperbolic cross
#     """
#     from itertools import product
    
#     # The hyperbolic cross criterion uses ∏(iⱼ + 1) ≤ s + 1
#     # The maximum index in any dimension is at most s
#     max_degree_needed = min(n, s) if n is not None else s
    
#     # Generate indices efficiently by checking the criterion directly
#     hyperbolic_indices = []
    
#     # Create an n-dimensional grid of indices, filtering by the hyperbolic cross criterion
#     all_indices = list(product(range(max_degree_needed + 1), repeat=d))
    
#     # Filter indices based on the correct criterion: ∏(iⱼ + 1) ≤ s + 1
#     for idx in all_indices:
#         # Calculate product of (j_k + 1)
#         product = 1
#         for j in idx:
#             product *= (j + 1)
        
#         # Keep indices satisfying the hyperbolic cross criterion
#         if product <= s + 1:
#             hyperbolic_indices.append(idx)
    
#     # For debugging
#     print(f"Hyperbolic cross J({s}) has {len(hyperbolic_indices)} indices for d={d}")
    
#     return hyperbolic_indices

# def gen_hyperbolic_legendre(coefficients, indices, domain=(-1, 1)):
#     """
#     Create a d-dimensional function using Legendre polynomial basis on hyperbolic cross.
    
#     Args:
#         coefficients: List of coefficients for the hyperbolic cross basis
#         indices: List of d-tuples from hyperbolic_cross_indices
#         domain: Domain interval for each dimension (default: [-1, 1])
        
#     Returns:
#         function: A function that evaluates the sparse Legendre expansion at any point x
#     """
#     if len(coefficients) != len(indices):
#         raise ValueError("Number of coefficients must match number of basis terms")
    
#     # Extract dimension from indices
#     d = len(indices[0]) if indices else 1
    
#     # Domain scaling parameters
#     a, b = domain
    
#     def func(x):
#         """
#         Evaluate sparse Legendre expansion at point x.
        
#         Args:
#             x: Point of shape (d,) or array of points with shape (n_points, d)
            
#         Returns:
#             Value of the function at x
#         """
#         x = np.asarray(x)
        
#         # Handle single point vs. multiple points
#         if x.ndim == 1:
#             if x.shape[0] != d:
#                 raise ValueError(f"Input dimension {x.shape[0]} doesn't match function dimension {d}")
            
#             # Map from custom domain to [-1, 1] for each dimension
#             x_mapped = (2*x - (a+b)) / (b-a)
            
#             # Compute basis functions and sum up
#             result = 0.0
#             for i, idx in enumerate(indices):
#                 # Compute product of 1D Legendre polynomials
#                 basis_val = 1.0
#                 for dim, degree in enumerate(idx):
#                     P = sp.legendre(degree)
#                     basis_val *= P(x_mapped[dim])
                
#                 result += coefficients[i] * basis_val
                
#             return result
            
#         else:  # Handle multiple points
#             if x.shape[1] != d:
#                 raise ValueError(f"Input dimension {x.shape[1]} doesn't match function dimension {d}")
            
#             # Map from custom domain to [-1, 1] for each dimension
#             x_mapped = (2*x - (a+b)) / (b-a)
            
#             # Initialize result array
#             result = np.zeros(x.shape[0])
            
#             # Compute basis functions and sum up
#             for i, idx in enumerate(indices):
#                 # Compute product of 1D Legendre polynomials
#                 basis_vals = np.ones(x.shape[0])
#                 for dim, degree in enumerate(idx):
#                     P = sp.legendre(degree)
#                     basis_vals *= P(x_mapped[:, dim])
                
#                 result += coefficients[i] * basis_vals
#             return result
#     return func

# def hyperbolic_design_matrix(x_data, indices, domain=(-1, 1)):
#     """
#     Create design matrix for hyperbolic cross Legendre basis.
    
#     Args:
#         x_data: Array of data points with shape (n_points, d)
#         indices: List of d-tuples from hyperbolic_cross_indices
#         domain: Domain interval for each dimension (default: [-1, 1])
        
#     Returns:
#         A: Design matrix of shape (n_points, len(indices))
#     """
#     x_data = np.asarray(x_data)
#     n_points = x_data.shape[0]
#     d = x_data.shape[1]
#     n_basis = len(indices)
    
#     # Domain scaling parameters
#     a, b = domain
    
#     # Map data to standard Legendre domain [-1, 1]
#     x_mapped = (2*x_data - (a+b)) / (b-a)
    
#     # Create design matrix A where A[i,j] is the j-th basis function evaluated at x_i
#     A = np.zeros((n_points, n_basis))
    
#     for j, idx in enumerate(indices):
#         # Compute product of 1D Legendre polynomials
#         basis_vals = np.ones(n_points)
#         for dim, degree in enumerate(idx):
#             P = sp.legendre(degree)
#             basis_vals *= P(x_mapped[:, dim])
        
#         A[:, j] = basis_vals
    
#     return A

# def fit_hyperbolic_legendre(x_data, y_data, d, n, s=4, domain=(-1, 1), report_error=False):
#     """
#     Fit data using hyperbolic cross Legendre basis.
    
#     Args:
#         x_data: Array of x-coordinates of data points with shape (n_points, d)
#         y_data: Array of y-coordinates of data points with shape (n_points,)
#         d: Dimension of input space
#         n: Maximum polynomial degree to consider
#         s: Parameter controlling the cardinality of the index set (s=4 gives |J(s)|≈52)
#         domain: Domain interval for each dimension (default: [-1, 1])
#         report_error: Whether to print fit quality metrics (default: False)
        
#     Returns:
#         coefficients: Array of coefficients for the hyperbolic cross basis
#         indices: List of d-tuples representing the hyperbolic cross indices
#     """
#     if len(x_data) != len(y_data):
#         raise ValueError("x_data and y_data must have the same length")
    
#     # Generate hyperbolic cross indices
#     indices = hyperbolic_cross_indices(d, n, s)
#     print(f"Using hyperbolic cross basis with {len(indices)} terms for dimension {d}")
    
#     # Create design matrix
#     A = hyperbolic_design_matrix(x_data, indices, domain)
    
#     # Solve the least squares problem: min ||A*coeffs - y_data||^2
#     coeffs, residuals, rank, s_vals = np.linalg.lstsq(A, y_data, rcond=None)
    
#     # Report fit quality if requested
#     if report_error:
#         # Calculate fitted values
#         y_fit = A @ coeffs
        
#         # Calculate errors
#         errors = y_data - y_fit
#         max_error = np.max(np.abs(errors))
#         mean_error = np.mean(np.abs(errors))
#         rmse = np.sqrt(np.mean(errors**2))
        
#         # Calculate R² (coefficient of determination)
#         ss_total = np.sum((y_data - np.mean(y_data))**2)
#         ss_residual = np.sum(errors**2)
#         r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
        
#         print(f"Hyperbolic Legendre fit with {len(indices)} basis functions:")
#         print(f"  Max absolute error: {max_error:.6e}")
#         print(f"  Mean absolute error: {mean_error:.6e}")
#         print(f"  RMSE: {rmse:.6e}")
#         print(f"  R² (coefficient of determination): {r_squared:.6f}")
#         if len(residuals) > 0:
#             print(f"  Sum of squared residuals: {residuals[0]:.6e}")
#         print(f"  Matrix rank: {rank}")
        
#         # Check if potentially underdetermined
#         if rank < min(A.shape):
#             print("  Warning: Matrix is rank deficient. Consider reducing the basis size.")
        
#         # Check for potential numerical issues
#         condition_number = np.linalg.cond(A)
#         if condition_number > 1e10:
#             print(f"  Warning: Design matrix is ill-conditioned (condition number: {condition_number:.2e}).")
    
#     return coeffs, indices

# def test_hyperbolic_cross():
#     """Test function to verify the correctness of hyperbolic_cross_indices."""
#     from itertools import product
    
#     print("Testing hyperbolic cross index generation:")
    
#     # Test cases with exact expected counts
#     test_cases = [
#         (2, 4, 15),   # d=2, s=4 should give |J(s)| = 15
#         (2, 9, 40),   # d=2, s=9 should give |J(s)| = 40
#         (3, 3, 20)    # d=3, s=3 should give |J(s)| = 20
#     ]
    
#     for d, s, expected in test_cases:
#         indices = hyperbolic_cross_indices(d, None, s)
#         print(f"For d={d}, s={s}: Generated {len(indices)} indices, expected {expected}")
        
#         # Print first few indices sorted by total degree
#         sorted_indices = sorted(indices, key=sum)
#         print(f"First few indices: {sorted_indices[:min(5, len(sorted_indices))]}")
        
#         # Verify the boundary case
#         boundary_in = False
#         boundary_out = False
        
#         for idx in indices:
#             prod = 1
#             for j in idx:
#                 prod *= (j + 1)
#             if prod == s + 1:
#                 boundary_in = True
#                 print(f"Boundary case included: {idx} with product = {s+1}")
#                 break
        
#         # Try to find a case just outside the boundary
#         for total_degree in range(d, d*s + 1):
#             # Generate indices with a specific total degree
#             for idx in sorted([i for i in product(range(s+1), repeat=d) if sum(i) == total_degree], 
#                              key=lambda x: [x[i] for i in range(d)]):
#                 prod = 1
#                 for j in idx:
#                     prod *= (j + 1)
#                 if prod == s + 2:  # Just outside the boundary
#                     boundary_out = True
#                     print(f"Boundary case excluded: {idx} with product = {s+2}")
#                     break
#             if boundary_out:
#                 break
                
#     # Plot the index set for 2D case (if matplotlib is available)
#     try:
#         import matplotlib.pyplot as plt
#         import numpy as np
        
#         d, s = 2, 9  # Use s=9 for a more interesting visualization
#         indices = hyperbolic_cross_indices(d, None, s)
        
#         x = [idx[0] for idx in indices]
#         y = [idx[1] for idx in indices]
        
#         plt.figure(figsize=(8, 6))
#         plt.scatter(x, y, marker='o')
#         plt.grid(True)
#         plt.xlabel('i₁')
#         plt.ylabel('i₂')
#         plt.title(f'Hyperbolic Cross Index Set J({s}) for d={d}')
        
#         # Draw hyperbola boundary (i₁+1)*(i₂+1) = s+1
#         x_curve = np.linspace(0, s, 1000)
#         y_curve = (s+1)/(x_curve + 1) - 1
#         mask = y_curve >= 0
#         plt.plot(x_curve[mask], y_curve[mask], 'r-', label=f'(i₁+1)(i₂+1) = {s+1}')
        
#         plt.legend()
#         plt.axis([0, s, 0, s])
#         plt.savefig('hyperbolic_cross.png')
#         print("Saved visualization to 'hyperbolic_cross.png'")
        
#     except (ImportError, NameError) as e:
#         print(f"Visualization skipped: {str(e)}")
    
#     return True

# def H1_error(y_true, y_pred, grad_y_true, grad_y_pred):
#     """
#     Calculate H1 (Sobolev) error between true and predicted functions.
    
#     H1 error is the sum of the L2 error of the function values
#     and the L2 error of the gradients.
    
#     Args:
#         y_true: Array of true function values
#         y_pred: Array of predicted function values
#         grad_y_true: Array of true gradient values, shape (n_points, n_dims)
#         grad_y_pred: Array of predicted gradient values, shape (n_points, n_dims)
        
#     Returns:
#         h1_error: The H1 error
#         l2_error: The L2 error of function values (component of H1)
#         gradient_error: The L2 error of gradients (component of H1)
#     """
#     # Ensure all inputs are numpy arrays
#     y_true = np.asarray(y_true)
#     y_pred = np.asarray(y_pred)
#     grad_y_true = np.asarray(grad_y_true)
#     grad_y_pred = np.asarray(grad_y_pred)
    
#     # Calculate L2 error for function values
#     l2_error = np.sqrt(np.mean((y_true - y_pred)**2))
    
#     # Calculate L2 error for gradients
#     # This computes the squared norm of the difference for each point
#     sq_grad_diff = np.sum((grad_y_true - grad_y_pred)**2, axis=1)
#     gradient_error = np.sqrt(np.mean(sq_grad_diff))
    
#     # H1 error is the sum of the two components
#     h1_error = l2_error + gradient_error
    
#     return h1_error, l2_error, gradient_error
#     test_hyperbolic_cross()
    
#     print("\nDemonstrating value and gradient fitting with hyperbolic Legendre basis:")
#     try:
#         data = np.load("data_result/raw_data/VDP_beta_0.1_grid_30x30.npy")
        
#         # Extract coordinates and values
#         x_data = data["x"][:100]  # Take first 100 points
#         v_data = data["v"][:100]  # Values
#         dv_data = data["dv"][:100]  # Gradients
        
#         print(f"Loaded data: {x_data.shape[0]} points of dimension {x_data.shape[1]}")
        
#         # Define domain based on data range
#         domain = (np.min(x_data), np.max(x_data))
#         print(f"Data domain: {domain}")
        
#         # Fit hyperbolic Legendre basis with s=5
#         d = x_data.shape[1]  # Dimension
#         n = 100  # Max polynomial degree
#         s = 16   # Hyperbolic cross parameter
        
#         # 1. Fit using only value information
#         print("\n====== Fitting with values only ======")
#         indices = hyperbolic_cross_indices(d, n, s)
#         A_values = hyperbolic_design_matrix(x_data, indices, domain)
        
#         # Solve least squares problem for values only
#         value_coeffs, residuals, rank, s_vals = np.linalg.lstsq(A_values, v_data, rcond=None)
        
#         # Generate the function from the fitted coefficients
#         value_func = gen_hyperbolic_legendre(value_coeffs, indices, domain)
        
#         # Predict values
#         v_pred_values_only = np.array([value_func(x) for x in x_data])
        
#         # To get gradient predictions, we would need to compute derivatives of the basis functions
#         # For simplicity, let's use a numerical approximation
#         def numerical_gradient(func, x, h=1e-6):
#             grad = np.zeros(len(x))
#             for i in range(len(x)):
#                 x_plus = x.copy()
#                 x_plus[i] += h
#                 x_minus = x.copy()
#                 x_minus[i] -= h
#                 grad[i] = (func(x_plus) - func(x_minus)) / (2*h)
#             return grad
        
#         # Compute gradients numerically for the value-only fit
#         dv_pred_values_only = np.array([numerical_gradient(value_func, x) for x in x_data])
        
#         # 2. Now implement a fit that includes gradient information
#         print("\n====== Fitting with values and gradients ======")
        
#         # Create an augmented design matrix that includes gradient information
#         # For each basis function, we need its value and its gradient with respect to each dimension
        
#         # First, let's create the gradient design matrices - one for each dimension
#         A_gradients = []
#         for dim in range(d):
#             # This matrix will contain the derivatives of each basis function with respect to dim
#             A_grad_dim = np.zeros((len(x_data), len(indices)))
            
#             # For each basis function (column in the design matrix)
#             for j, idx in enumerate(indices):
#                 # Calculate the derivative of the current basis function with respect to dim
#                 for i, x in enumerate(x_data):
#                     # For numerical derivative, perturb the point slightly in dimension dim
#                     h = 1e-6
#                     x_plus = x.copy()
#                     x_plus[dim] += h
#                     x_minus = x.copy()
#                     x_minus[dim] -= h
                    
#                     # Compute derivative using finite difference
#                     basis_plus = np.ones(1)
#                     basis_minus = np.ones(1)
                    
#                     # Map to standard domain
#                     a, b = domain
#                     x_plus_mapped = (2*x_plus - (a+b)) / (b-a)
#                     x_minus_mapped = (2*x_minus - (a+b)) / (b-a)
                    
#                     # Evaluate the basis function at perturbed points
#                     for d_idx, degree in enumerate(idx):
#                         P = sp.legendre(degree)
#                         basis_plus *= P(x_plus_mapped[d_idx])
#                         basis_minus *= P(x_minus_mapped[d_idx])
                    
#                     # Finite difference approximation of derivative
#                     A_grad_dim[i, j] = (basis_plus[0] - basis_minus[0]) / (2*h)
            
#             A_gradients.append(A_grad_dim)
        
#         # Create the augmented design matrix [A_values; A_grad_dim1; A_grad_dim2; ...]
#         A_augmented = np.vstack([A_values] + A_gradients)
        
#         # Create the augmented target vector [v_data; dv_data_dim1; dv_data_dim2; ...]
#         y_augmented = np.concatenate([v_data] + [dv_data[:, i] for i in range(d)])
        
#         # Solve the augmented least squares problem
#         combined_coeffs, combined_residuals, combined_rank, combined_s = np.linalg.lstsq(A_augmented, y_augmented, rcond=None)
        
#         # Generate the function from the combined coefficients
#         combined_func = gen_hyperbolic_legendre(combined_coeffs, indices, domain)
        
#         # Predict values using the combined fit
#         v_pred_combined = np.array([combined_func(x) for x in x_data])
        
#         # Compute gradients numerically for the combined fit
#         dv_pred_combined = np.array([numerical_gradient(combined_func, x) for x in x_data])
        
#         # 3. Evaluate and compare both approaches
#         print("\n====== Error Comparison ======")
        
#         # Calculate errors for values-only fit
#         h1_values_only, l2_values_only, grad_err_values_only = H1_error(
#             v_data, v_pred_values_only, dv_data, dv_pred_values_only
#         )
        
#         # Calculate errors for combined fit
#         h1_combined, l2_combined, grad_err_combined = H1_error(
#             v_data, v_pred_combined, dv_data, dv_pred_combined
#         )
        
#         print(f"Values-only fit:")
#         print(f"  H1 error: {h1_values_only:.6e}")
#         print(f"  L2 error (values): {l2_values_only:.6e}")
#         print(f"  Gradient error: {grad_err_values_only:.6e}")
        
#         print(f"\nCombined fit (values + gradients):")
#         print(f"  H1 error: {h1_combined:.6e}")
#         print(f"  L2 error (values): {l2_combined:.6e}")
#         print(f"  Gradient error: {grad_err_combined:.6e}")
        
#         print(f"\nImprovement ratios (Combined vs Values-only):")
#         print(f"  H1 error ratio: {h1_combined/h1_values_only:.4f} (< 1 means combined is better)")
#         print(f"  L2 error ratio: {l2_combined/l2_values_only:.4f}")
#         print(f"  Gradient error ratio: {grad_err_combined/grad_err_values_only:.4f}")
        
#         print("\nConclusion:")
#         if h1_combined < h1_values_only:
#             print("  The combined fit (including gradient information) performs better in H1 norm.")
#             print("  This demonstrates the value of including gradient information in the fitting process.")
#         else:
#             print("  The values-only fit performs better in this example.")
#             print("  This could be due to noise in the gradient data or numerical issues.")
        
#     except Exception as e:
#         print(f"Could not run demonstration: {str(e)}")
#         import traceback
#         traceback.print_exc()


