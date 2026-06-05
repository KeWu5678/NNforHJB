import math
import numpy as np
import scipy.special as sp
from numpy.polynomial import legendre as npleg
from numpy.polynomial.chebyshev import Chebyshev
from scipy.spatial import KDTree
import torch

# The non-convex penalty / proximal kernels now live in the self-contained
# ``src.SSN`` package.  They are re-exported here so existing
# ``from .utils import _phi`` style imports keep working.  (``SSN`` imports
# nothing from ``utils``, so this does not create a cycle.)
from .SSN.penalty import (
    _phi,
    _dphi,
    _ddphi,
    _penalty_grad,
    _nonconvex_correction,
    _nonconvex_correction_dd,
)
from .SSN.prox import (
    _compute_prox,
    _compute_prox_q_half,
    _compute_prox_q_twothirds,
    _compute_dprox,
    _phi_prox,
)


def stereo(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Stereographic projection from R^2 to the unit sphere S^2.

    Args:
        z: torch tensor with shape (2, n)

    Returns:
        (a, b) where:
        - a has shape (2, n)
        - b has shape (1, n)
    """
    if not isinstance(z, torch.Tensor):
        raise TypeError(f"stereo expects torch.Tensor, got {type(z)!r}")
    if z.ndim != 2 or z.shape[0] != 2:
        raise ValueError("z must have shape (2, n)")

    z2_sum = (z * z).sum(dim=0, keepdim=True)  # (1, n)
    denominator = 1 + z2_sum                   # (1, n)
    a = (2 * z) / denominator                  # (2, n)
    b = (1 - z2_sum) / denominator             # (1, n)
    return a, b

def invese_stereo(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: 
    return a/2

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
    
    grid = np.asarray(grid, dtype=float)
    x = np.asarray(x, dtype=float)
    if grid.shape[0] != x.shape[0]:
        raise ValueError("grid and x must have the same size")
    # Left-Riemann sum of x**2 over the (non-uniform) grid mesh.
    mesh = np.diff(grid)
    return float(np.dot(mesh, x[:-1] ** 2))

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
    coefficients = np.asarray(coefficients, dtype=float)
    a, b = domain
    # Precompute the affine map t -> (2t - (a+b)) / (b-a) onto Legendre's [-1, 1].
    scale = 2.0 / (b - a)
    shift = (a + b) / (b - a)

    def func(t):
        # legval sums coefficients[i] * P_i in one vectorized C call and handles
        # both scalar and array inputs, so no Python-level loop is needed.
        return npleg.legval(scale * np.asarray(t, dtype=float) - shift, coefficients)

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
    x_mapped = (2 * np.asarray(x_data, dtype=float) - (a + b)) / (b - a)

    # Vandermonde matrix A[i, j] = P_j(x_i), built in one vectorized call.
    A = npleg.legvander(x_mapped, n - 1)

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