#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:32:26 2024

@author: chaoruiz
"""

import OpenLoop as op
import numpy as np


"""
DATA GENERATION
"""

# GLOBAL PARAMETER
span = [0, 3]
grid = np.linspace(span[0], span[1], 1000)
guess = np.ones((4, grid.size))
beta = 2
tol = 1e-4
N = 50
max_it = 100000

# Define structured array dtype
dtype = [
    ('x0', '2float64'),  # 2D float array for the first element
    ('dV0', '2float64'), # 2D float array for the second element
    ('V', 'float64')    # 1D float for the third element
]

# Create structured array
samples = np.zeros(N, dtype=dtype)

def optimizer(ini):
    """Test the OpenLoopOptimizer class"""
    optimizer = op.OpenLoopOptimizer(span, grid, guess, beta, tol, ini, max_it)
    dV, V = optimizer.optimize()
    return dV, V




if __name__ == "__main__":
    for i in range(N):
        ini = np.random.uniform(-3, 3, 2)
        print(f"i = {i}")
        print(f"ini = {ini}")
        dV, V = optimizer(ini)
        samples[i] = (ini, dV, V)
        
        




