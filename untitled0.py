#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 21:26:23 2024

@author: chaoruiz
"""


import numpy as np
from scipy.integrate import solve_bvp


def BVP_VDP(t, y, u):
   """
   Define the Boundary value problem with the control parameter u
   Input 
       t: dummy
       y: dummy
       u: function of u
   """
   return np.vstack([
       y[1],
       -y[0] + y[1] * (1 - y[0] ** 2) + u(t),
       -y[3] - 2 * y[0],
       2 * y[2] * y[0] * y[1] + y[3] * y[0] ** 2 + y[2] - y[3] - 2 * y[1]
   ])

def bc(ya, yb, ini):
   # Residuals for boundary conditions
   return np.array([
       ya[0] - ini[0],  # y1(0) = ini[0]  
       ya[1] - ini[1],  # y2(0) = ini[1]
       yb[2],          # p1(3) = 0
       yb[3]           # p2(3) = 0
   ])

def L2(x, grid):
   """
   Given the control and solution and TVBVP, evaluate the gradient.

   input:  
       x:      1 x grid.size array
               Control current
       grid:   1 x grid.size array
               linspace where x is evaluated

   return:   
       norm:   float
               The L2 norm of x.     
   """
   norm = 0
   for i in range(grid.size - 1):
       mesh = grid[i + 1] - grid[i]
       x_sq = x[i] ** 2
       norm += mesh * x_sq
   return norm

def sol_BVP(ode, bc, span, ini, u_coe, guess, grid, tol):
   """
   Solve the TVBVP by sol_BVP methods. The space of the control is interpolate by
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
   def u(t):
       conditions = [(t >= grid[i]) & (t < grid[i+1]) for i in range(grid.size - 1)]
       return np.piecewise(t, conditions, u_coe)
   
   sol = solve_bvp(lambda t, y: ode(t, y, u), lambda ya, yb: bc(ya, yb, ini), grid, guess, tol=1e-6)
   return sol

def Grad_VDP(sol, u, beta):
   """
   Describe the gradient function of the system
   Parameter:  
       sol:    OdeResult
               Solution of the TVBVP
       u:      1 x grid.size
               Coefficient of the control
   """
   grid = sol.x
   grad = np.zeros(grid.size)
   p2 = sol.y[3, :]
   
   for i in range(grid.size - 1):
       grad[i] = p2[i] + 2 * beta * u[i]
   
   return grad

"""
Solve the OpenLoop Problem and return the tuple: {V , \nabla V, x}
Input:
    span:   1 x 2 array
            span of the ODE
    grid:   linspace
            Grid where the ODE is evaluated
    guess:  1 x grid.size array
            Initial guess for the BVP
    beta:   float
            Parameter for the VDP system
    tol:    float
            Termination criteria for the Optimization System
    ini:    1 x 2 array
            The initial condition for the VDP
"""
def OpenLoop(span, grid, guess, beta, tol, ini):
    u0 = np.zeros(grid.size)
    alpha = 1/20
    sol = sol_BVP(BVP_VDP, bc, span, ini, u0, guess, grid, tol)
    G0 = -alpha * Grad_VDP(sol, u0, beta)
    u1 = u0 - alpha * G0
    sol = sol_BVP(BVP_VDP, bc, span, ini, u1, guess, grid, tol)
    G1 = Grad_VDP(sol, u1, beta)
    k = 1
    while L2(G1, grid) >= tol:
        print(f" k = {k}")
        print(f" alpha = {alpha}")
        print(f" norm u = {L2(u1,grid)}")
        print(f" norm G = {L2(G1,grid)}")
        
        u0, G0 = u1, G1
        u1 = u1 - alpha * G1
        sol = sol_BVP(BVP_VDP, bc, span, ini, u1, guess, grid, tol)
        G1 = Grad_VDP(sol, u1, beta)      
        k += 1
    p = np.array([
        sol.y[2, 0], 
        sol.y[3, 0]
    ]) 
    
    
    
    return p