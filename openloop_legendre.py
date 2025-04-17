#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:30:52 2024

@author: chaoruiz
"""


import numpy as np
import utils 
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

class OpenLoopOptimizer:
    def __init__(self, ODE, bc, V, gradient, grid, guess, tol, max_it):
        """
        Initialize OpenLoop optimizer
        
        Args:
            grid:   linspace of span
                    Grid for the initial evaluation 
            guess:  1 x grid.size array
                    Initial guess for BVP
            tol:    float
                    Tolerance of optimization
        """
        self.grid = grid
        self.guess = guess
        self.tol = tol
        self.max_it = max_it
        self.ODE = ODE
        self.bc = bc
        self.gradient = gradient
        self.V = V




    def sol_BVP(self, u):
        sol = solve_bvp(
            lambda t, y: self.ODE(t, y, u),
            self.bc,
            self.grid,
            self.guess,
            tol=1e-7,
            max_nodes = 10000
        )
        return sol
    
    def BBstep(self, u0, u1, G0, G1):
        s = u1 - u0
        y = G1 - G0
        
        sy = np.dot(s, y)
        ss = np.dot(s, s)
        yy = np.dot(y, y)
        
        epsilon = 1e-20  # Prevent division by zero
        BB1 = sy / (ss + epsilon)
        BB2 = yy / (sy + epsilon)
        return BB1, BB2
   

    def optimize(self):
        """Run optimization loop"""
        # Initialize
        num_basis = 30
        u0_coeff = np.zeros(num_basis)
        u0 = utils.gen_legendre(u0_coeff)
        
        print("Solving initial BVP...")
        sol0 = self.sol_BVP(u0)
        G0 = self.gradient(u0(sol0.x), sol0.y[3,:])
        grid0 = sol0.x
        G0_coeff = utils.fit_legendre(grid0, G0, num_basis)

        u1_coeff = - (1/10) * G0_coeff
        u1 = utils.gen_legendre(u1_coeff)
        sol1 = self.sol_BVP(u1)
        G1 = self.gradient(u1(sol1.x), sol1.y[3,:])
        grid1 = sol1.x
        G1_coeff = utils.fit_legendre(grid1, G1, num_basis)
        
        # Initial step size (will be adapted)
        alpha = 0.1

        # Optimization loop
        k = 1
        store = True
        while utils.L2(grid1, G1) >= self.tol and store:
            # Get BB step sizes with safety checks
            bb_step = self.BBstep(u0_coeff, u1_coeff, G0_coeff, G1_coeff)
            
            # Safely compute step size with checks
            if k % 2 == 0:
                if abs(bb_step[0]) > 1e-20 and not np.isnan(bb_step[0]) and not np.isinf(bb_step[0]):
                    alpha = 1 / bb_step[0]
                else:
                    # Fallback if BB step is problematic
                    alpha = 0.1  # Default safe step size
            else:
                if abs(bb_step[1]) > 1e-20 and not np.isnan(bb_step[1]) and not np.isinf(bb_step[1]):
                    alpha = 1 / bb_step[1]
                else:
                    # Fallback if BB step is problematic
                    alpha = 0.01  # Default safe step size
            
            # Bound step size for stability
            alpha = min(max(alpha, 5e-3), 1000.0)

                
            # Update coefficients - IMPORTANT: work with coefficients, not function values
            u_coeff_temp = u1_coeff - alpha * G1_coeff
            
            # Save current values before updating
            u0_coeff, G0_coeff = u1_coeff, G1_coeff
            
            # Create new control function with updated coefficients
            u1 = utils.gen_legendre(u_coeff_temp)
            u1_coeff = u_coeff_temp  # Store coefficients for next iteration
            
            try:
                # Solve BVP with new control
                sol1 = self.sol_BVP(u1)
                grid1 = sol1.x
                G1 = self.gradient(u1(grid1), sol1.y[3,:])
                G1_coeff = utils.fit_legendre(grid1, G1, num_basis)
                
                # Update iteration counter
                k += 1
                print(f" k = {k}")
                print(f" alpha = {alpha}")
                print(f" norm G = {utils.L2(sol1.x, G1)}")
                
                # Check for numerical instability
                if np.isnan(utils.L2(sol1.x, G1)) or np.isinf(utils.L2(sol1.x, G1)):
                    print("Warning: Numerical instability detected. Terminating.")
                    store = False
                    break
                    
            except Exception as e:
                print(f"Error in iteration {k}: {str(e)}")
                store = False
                break
            
            # Check max iterations
            if k >= self.max_it:
                store = False
                print("Maximum iterations reached")
                print(f" norm G = {utils.L2(sol1.x, G1)}")
                break
        # Final result
        if store == True:
            p = np.array([
                sol1.y[2, 0],
                sol1.y[3, 0]
            ])
            V = self.V(sol1.x, u1(sol1.x), sol1.y[0,:], sol1.y[1,:])
            print(f" cost = {V}")
            print(f" norm G = {utils.L2(sol1.x, G1)}")

            # fig = plt.figure(figsize=(10, 8))
            # ax = fig.add_subplot(111, projection='3d')
            
            # # Plot trajectory
            # t = sol1.x
            # x = sol1.y[0, :]
            # y = sol1.y[1, :]
            # ax.plot(t, x, y, label='Solution Trajectory')
            
            # # Labels and title
            # ax.set_xlabel('t')
            # ax.set_ylabel('x(t)')
            # ax.set_zlabel('y(t)')
            # ax.set_title('Solution Trajectory in Phase Space')
            
            # # Add legend
            # ax.legend()
            
            # plt.show()    
            return p, V
        else:
            print("maximum iteration reached")
            return None, None
    


# if __name__ == "__main__":
#     beta = 2
#     # ini = np.random.rand(2)
#     ini = [0.54491032, 0.52831127]
#     print(f" ini = {ini}")

# def VDP(t, y, u):
#     """Define the Boundary value problem with control parameter u"""
#     return np.vstack([
#         y[1],
#         -y[0] + y[1] * (1 - y[0] ** 2) + u(t),
#         -y[3] - 2 * y[0],
#         2 * y[0] * y[1] * y[2] + y[0] ** 2 * y[3] + y[2] - y[3] - 2 * y[1]
#     ])


# def bc(ya, yb):
#     """Boundary conditions"""
    
#     return np.array([
#         ya[0] - ini[0],
#         ya[1] - ini[1],
#         yb[2],
#         yb[3]
#     ])

# def gradient(u, p):
#     if len(p) != len(u):
#         raise ValueError("p and u must have the same length")
#     else:
#         n = len(p)
#         grad = np.zeros(n)
#     for i in range(n):
#         grad[i] = p[i] + 2 * beta * u[i]
#     return grad

# def V(grid, u, y1, y2):
#     return utils.L2(grid, u) * beta + 0.5 * (utils.L2(grid, y1) + utils.L2(grid, y2))

# grid = np.linspace(0, 3, 1000)
# guess = np.ones((4, grid.size)) 
# tol = 1e-6
# max_it = 1000
# optimizer = OpenLoopOptimizer(VDP, bc, V, gradient, grid, guess, tol, max_it)
# p, V = optimizer.optimize()
# print(f" p = {p}")
# print(f" V = {V}")