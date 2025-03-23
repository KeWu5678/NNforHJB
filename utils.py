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