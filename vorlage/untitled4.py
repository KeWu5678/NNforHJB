#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 20:09:09 2024

@author: chaoruiz
"""

import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

########################## GLOBAL VARIABLES ##################################
span = [0, 3]  # Integration interval
grid = np.linspace(span[0], span[1], 100000)  # Finer grid for performance
ini = [1, 2]  # Initial boundary conditions
beta = 2  # ODE parameter
tol = 1e-3  # Tolerance for BVP solver

##########################   THE ODE   #######################################
def BVP_VDP(t, y):
    
    return np.vstack([
        y[1], 
        -y[0] + y[1] * (1 - y[0] ** 2) - (1 / 2 * beta) * (y[2] + y[3]),
        -y[3] - 2 * y[0], 
        2 * y[2] * y[0] * y[1] + y[3] * y[0] ** 2 + y[2] - y[3] - 2 * y[1]
        ])
    # dydt[0] = y2
    # dydt[1] = -y1 + y2 * (1 - y1**2) - (1 / 2 * beta) * (p1 + p2)
    # dydt[2] = -p2 - 2 * y1
    # dydt[3] = 2 * p1 * y1 * y2 + p2 * y1**2 + p1 - p2 - 2 * y2

    # return dydt

########################## BOUNDARY CONDITIONS ################################
def bc(ya, yb):
    # Residuals for boundary conditions
    return np.array([
        ya[0] - ini[0],  # y1(0) = ini[0]
        ya[1] - ini[1],  # y2(0) = ini[1]
        yb[2],           # p1(3) = 0
        yb[3]            # p2(3) = 0
    ])

########################## COST FUNCTIONS #############################
"""
# Given the control and solution and TVBVP, evaluate the gradient.

# input:  
#     x:      1 x grid.size array
#             Control current
#     grid:   1 x grid.size array
#             linspace where x is evaluated

# return:   
#     norm:   float
#             The L2 norm of x.     
# """
def L2(x, grid):
    norm = 0  # Initialize norm

    for i in range(grid.size - 1):  # Avoid out-of-bounds error
        mesh = grid[i + 1] - grid[i]  # Mesh interval
        x_sq = x[i] ** 2 # Left evaluation
        norm += mesh * x_sq  # Compute measure for this interval
    return norm  # Return after the loop is complete


def Cost(y1, y2, u, grid):
    return L2(y1, grid) + L2(y2, grid) + beta * L2(u, grid)
    
    

########################## SOLVER ############################################
# Initial guess for the solution
guess = np.ones((4, grid.size))  # Initial guess for 4 variables over the grid

# Solve the BVP

sol = solve_bvp(BVP_VDP, bc, grid, guess, tol = tol)
u = np.zeros(grid.size)
u = - (1 / 2 * beta ) * (sol.y[2] + sol.y[3])
grid = sol.x
print(Cost(sol.y[0], sol.y[1], u, grid))


# t = sol.x
# plt.figure(figsize=(10, 6))
# plt.plot(t, sol.y[0], label="y1 (state)")
# plt.plot(t, sol.y[1], label="y2 (state)")
# plt.plot(t, sol.y[2], label="p1 (costate)")
# plt.plot(t, sol.y[3], label="p2 (costate)")
# plt.xscale('linear')  # Linear scale for x-axis
# plt.yscale('linear')     # Logarithmic scale for y-axis
# plt.grid(True, which="both", linestyle='--', linewidth=0.5)
# plt.plot(t, u)
