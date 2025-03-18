#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:32:26 2024

@author: chaoruiz
"""

import OpenLoop as op
import numpy as np
import numpy.polynomial.chebyshev as cheb
import discretization as dis


"""
DATA GENERATION
"""

# GLOBAL PARAMETER


# Define structured array dtype
dtype = [
    ('x', '2float64'),  # 2D float array for the first element
    ('dv', '2float64'), # 2D float array for the second element
    ('v', 'float64')    # 1D float for the third element
]

# Create structured array


"""=====================THE ODE======================"""
beta = 3

def VDP(t, y, u):
    """Define the Boundary value problem with control parameter u"""
    return np.vstack([
        y[1],
        -y[0] + y[1] * (1 - y[0] ** 2) + u(t),
        -y[3] - 2 * y[0],
        2 * y[0] * y[1] * y[2] + y[0] ** 2 * y[3] + y[2] - y[3] - 2 * y[1]
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
    return dis.L2(grid, u) * beta + 0.5 * (dis.L2(grid, y1) + dis.L2(grid, y2))

"""=====================OPTIMIZATION======================"""
grid = np.linspace(0, 3, 1000)
guess = np.ones((4, grid.size))
tol = 1e-6
N = 2000
dataset = np.zeros(N, dtype=dtype)
max_it = 500


def gen_bc(ini):
    def bc(ya, yb):
        """Boundary conditions"""
        return np.array([
            ya[0] - ini[0],
            ya[1] - ini[1],
            yb[2],
            yb[3]
        ])
    return bc



if __name__ == "__main__":
    for i in range(N):
        ini = np.random.uniform(0, 3, 2)
        bc = gen_bc(ini)
        print(f"i = {i}")
        print(f"ini = {ini}")
        dv, v = op.OpenLoopOptimizer(VDP, bc, V, gradient, grid, guess, tol, max_it).optimize()
        if dv is not None and not np.isnan(v):
            dataset[i] = (ini, dv, v)
        
    np.save("VDP_beta_3.npy", dataset)



