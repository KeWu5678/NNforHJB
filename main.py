#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:32:26 2024

@author: chaoruiz
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.pyplot as plt


########################## GLOBAL VAL ########################################
## the grid
span = [0, 3]
grid = np.linspace(span[0], span[1], 100000)
guess = [1, 1]
## the boundary conditions
degree = 10  # interpolation degree of the control space
beta = 2  ## ODE parameter
tol = 0.001  # termination criteria
ini = [1, 2]


########################## TVBVP SLOVER ######################################
"""
Solve the TVBVP by shooting methods. The space of the control is interpolate by
the piecewise constant functions with value given by the array u.

input:  
    f:      functions; 
            The ODE
    u:      1xgrid.size array; 
            The control function
    span:   1x2 array; 
            the time span 
    ini:    1x2 array; 
            initial contition of ode
    guess:  1x2 array; 
            guess of the initial condition of the adjoint p
    tol:    1x2 array; 
            tolerance of the root-finding
    grid:   1d array 
            grid of the evaluation

return:   
    grad:   1 x grid.size array
            gradient evaluated on the grid
    
"""
def shooting(f, span, ini, guess, grid, tol):
   
    ##  Define the operator form guess of initial condition to residual of bc 
    ##  return: pb: 1x2 array;
    def residual(guess):
        guess_expand = np.zeros(4)
        guess_expand = [*ini, *guess]
        sol = solve_ivp(f, span, guess_expand, method='Radau', t_eval=grid, dense_output=False)
        pb = sol.y[2:4, -1]
        return pb

    result = root(residual, guess, tol=tol)
    
    if not result.success:
        raise RuntimeError("Root finding failed: " + result.message)

    correct_ini = result.x
    # Solve the IVP with the correct slopes and evaluate on the fixed grid
    bvp_sol = solve_ivp(f, span, [*ini, *correct_ini], t_eval=grid)
    
    return bvp_sol





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





##########################


sol = shooting(BVP_VDP, span, ini, guess, grid, tol)
u = np.zeros(grid.size)
u = - (1 / 2 * beta ) * (sol.y[2, :] + sol.y[3, :])


t = sol.t
plt.xscale('linear')  # Linear scale for x-axis
plt.yscale('linear')     # Logarithmic scale for y-axis
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label="y1 (state)")
plt.plot(sol.t, sol.y[1], label="y2 (state)")
plt.plot(sol.t, sol.y[2], label="p1 (costate)")
plt.plot(sol.t, sol.y[3], label="p2 (costate)")
# plt.ylim(-5, 10000)
# plt.xlim(2, 3)
# plt.grid(True, which="both", linestyle='--', linewidth=0.5)
# plt.plot(t, u)




