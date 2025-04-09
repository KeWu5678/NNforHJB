#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:30:52 2024

@author: chaoruiz
"""


import numpy as np
import utils 
from VDP import vdp
import matplotlib.pyplot as plt
import time

def OpenLoopOptimize(solver, ini, V, gradient, grid, tol = 1e-6, max_it = 1000):
    """
    Initialize OpenLoop optimizer
    
        Args:
            grid:   linspace of span
                    Grid for the initial evaluation 
            ini:  1 x grid.size array
                    Initial guess for BVP
            tol:    float
                    Tolerance of optimization
            solver(grid, u): callable
                    return the grid and function value on the grid
            gradient: callable
                    given the state and control, return the gradient on the grid
        """

    # Start timing
    total_start = time.time()
    
    def BBstep(u0, u1, G0, G1):
        s = u1 - u0
        y = G1 - G0
        
        sy = np.dot(s, y)
        ss = np.dot(s, s)
        yy = np.dot(y, y)
        
        epsilon = 1e-10  # Prevent division by zero
        BB1 = sy / (ss + epsilon)
        BB2 = yy / (sy + epsilon)
        return BB1, BB2
   

    """Run optimization loop"""
    # Initialize
    num_basis = 10
    u0_coeff = np.zeros(num_basis)
    u0 = utils.gen_legendre(u0_coeff)
    
    # Initial solver call with zero control
    print("\nInitial solver call...")
    solver_start = time.time()
    t, y_sol = solver(u0, ini, grid)
    solver_time = time.time() - solver_start
    print(f"Initial solver time: {solver_time:.2f} seconds")
    
    # Extract p values from the solution
    p_values = y_sol[3, :]
    
    # Compute gradient
    print("\nComputing initial gradient...")
    grad_start = time.time()
    G0 = gradient(u0(grid), p_values)
    grad_time = time.time() - grad_start
    print(f"Initial gradient computation time: {grad_time:.2f} seconds")
    
    # Get coefficients for the gradient
    print("\nFitting initial Legendre coefficients...")
    fit_start = time.time()
    G0_coeff = utils.fit_legendre(grid, G0, num_basis, report_error=True)
    fit_time = time.time() - fit_start
    print(f"Initial Legendre fitting time: {fit_time:.2f} seconds")
    
    # Create initial control function with small negative gradient
    step_size = 0.1
    u1_coeff = -step_size * G0_coeff  # Using coefficients directly
    u1 = utils.gen_legendre(u1_coeff)
    
    # Solve with the new control
    print("\nSecond solver call...")
    solver_start = time.time()
    t, y_sol = solver(u1, ini, grid)
    solver_time = time.time() - solver_start
    print(f"Second solver time: {solver_time:.2f} seconds")
    
    p_values = y_sol[3, :]
    G1 = gradient(u1(grid), p_values)
    G1_coeff = utils.fit_legendre(grid, G1, num_basis)
    
    # Initial step size
    alpha = 0.01
    
    # Optimization loop
    k = 1
    store = True
    G_norm = utils.L2(grid, G1)
    print(f"\nInitial gradient norm: {G_norm:.6e}")
    
    # Track timing for each iteration
    iteration_times = []
    
    while G_norm >= tol and k < max_it:
        try:
            iter_start = time.time()
            
            # Compute BB step sizes
            # bb_step = BBstep(u0_coeff, u1_coeff, G0_coeff, G1_coeff)
            
            # # Use alternating BB step sizes
            # if k % 2 == 0 and not np.isnan(bb_step[0]) and not np.isinf(bb_step[0]) and bb_step[0] > 1e-10:
            #     alpha = 1.0 / bb_step[0]
            # elif not np.isnan(bb_step[1]) and not np.isinf(bb_step[1]) and bb_step[1] > 1e-10:
            #     alpha = 1.0 / bb_step[1]
            # else:
            #     # Use a small fixed step size if BB steps are unstable
            #     alpha = 0.01
            
            alpha = 1/10

            
            # Update coefficients using gradient descent
            u2_coeff = u1_coeff - alpha * G1_coeff
            
            # Save current values for next iteration
            u0_coeff, G0_coeff = u1_coeff, G1_coeff
            
            # Create new control function and solve
            u1 = utils.gen_legendre(u2_coeff)
            u1_coeff = u2_coeff
            
            # Time the solver call
            solver_start = time.time()
            t, y_sol = solver(u1, ini, grid)
            solver_time = time.time() - solver_start
            
            p_values = y_sol[3, :]
            G1 = gradient(u1(grid), p_values)
            G1_coeff = utils.fit_legendre(grid, G1, num_basis)
            
            # Compute gradient norm
            G_norm = utils.L2(grid, G1)
            
            # Record iteration time
            iter_time = time.time() - iter_start
            iteration_times.append(iter_time)
            
            # Print iteration info
            k += 1
            print(f"\nIteration {k}:")
            print(f"  alpha = {alpha}")
            print(f"  norm G = {G_norm}")
            print(f"  iteration time = {iter_time:.2f} seconds")
            print(f"  solver time = {solver_time:.2f} seconds")
            
            # Check for NaNs or extreme values indicating instability
            if np.isnan(G_norm) or np.isinf(G_norm) or np.any(np.isnan(u1_coeff)) or np.any(np.isinf(u1_coeff)):
                print("Warning: Numerical instability detected. Terminating optimization.")
                store = False
                break
            
        except Exception as e:
            print(f"Error in iteration {k}: {str(e)}")
            store = False
            break
    
    # Final result
    if store:
        # Extract initial costate values
        p0 = np.array([y_sol[2, 0], y_sol[3, 0]])
        
        # Compute cost
        V_value = V(grid, u1(grid), y_sol[0, :], y_sol[1, :])
        
        # Print timing summary
        total_time = time.time() - total_start
        avg_iter_time = np.mean(iteration_times) if iteration_times else 0
        print(f"\nOptimization completed in {total_time:.2f} seconds")
        print(f"Average iteration time: {avg_iter_time:.2f} seconds")
        print(f"Total iterations: {k}")
        print(f"Final cost = {V_value}")
        print(f"Final norm G = {G_norm}")
        
        return p0, V_value
    else:
        print("maximum iteration reached")
        return None, None
    
def gradient(u, p):
    beta = 0.1
    if len(p) != len(u):
        raise ValueError("p and u must have the same length")
    else:
        n = len(p)
        grad = np.zeros(n)
    for i in range(n):
        grad[i] = p[i] + 2 * beta * u[i]
    return grad

def V(grid, u, y1, y2):
    beta = 0.1  # Define beta here to avoid undefined variable error
    return utils.L2(grid, u) * beta + 0.5 * (utils.L2(grid, y1) + utils.L2(grid, y2))
    
if __name__ == "__main__":
    ini = [1.1, 1.1]
    grid = np.linspace(0, 3, 1001)
    OpenLoopOptimize(vdp, ini, V, gradient, grid, tol=1e-4, max_it=100)