import numpy as np
import scipy.special as sp
from numpy.polynomial.chebyshev import Chebyshev
from scipy.spatial import KDTree
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



