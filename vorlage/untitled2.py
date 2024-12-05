#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:03:47 2024

@author: chaoruiz
"""

import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

########################## GLOBAL VAL ########################################
## the grid
span = [0, 3]
grid = np.linspace(span[0], span[1], 100000)
guess = np.ones((4, grid.size))
## the boundary conditions
degree = 10  # interpolation degree of the control space
beta = 2  ## ODE parameter
tol = 0.001  # termination criteria
ini = [3, 3]


########################   THE ODE   ##########################################
"""
Define the Boudary value problem with the control parameter u
Input 
    t: dummy
    y: dummy
    u: function of u
"""
def BVP_VDP(t, y):
    y1, y2, p1, p2 = y  # Unpack the variables

    # Dynamics for y (states)
    dydt = np.zeros(4)  # Initialize the derivatives
    dydt[0] = y2
    dydt[1] = -y1 + y2 * (1 - y1**2) - (1 / 2 * beta ) * (p1 + p2)

    # Dynamics for p (costates)
    dydt[2] = -p2 - 2 * y1 
    dydt[3] = 2 * p1 * y1 * y2 + p2 * y1**2 + p1 - p2 - 2 * y2

    return dydt


def bc(ya, yb):
    return np.array([
        ya[0] - ini[0],
        ya[1] - ini[1],
        yb[2],
        yb[3]
        ])



##########################


sol = solve_bvp(BVP_VDP, bc, grid, guess, tol = tol)
u = np.zeros(grid.size)
u = - (1 / 2 * beta ) * (sol.y[2, :] + sol.y[3, :])


t = sol.t
plt.xscale('linear')  # Linear scale for x-axis
plt.yscale('linear')     # Logarithmic scale for y-axis
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.plt(t, u)
