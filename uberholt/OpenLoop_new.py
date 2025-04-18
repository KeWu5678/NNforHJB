#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:30:52 2024

@author: chaoruiz
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

class OpenLoopOptimizer:
    def __init__(self, span, grid, guess, beta, tol, ini):
        """
        Initialize OpenLoop optimizer
        
        Args:
            span:   1 x 2 array 
                    Span of the ODE
            grid:   linspace of span
                    Grid for the initial evaluation 
            guess:  1 x grid.size array
                    Initial guess for BVP
            beta:   float
                    Parameter in the cost function
            tol:    float
                    Tolerance
            ini:    1 x 2 array
                    Initial condition for VDP
        """
        self.span = span
        self.grid = grid
        self.guess = guess
        self.beta = beta
        self.tol = tol
        self.ini = ini

    def BVP_VDP(self, t, y, u):
        """Define the Boundary value problem with control parameter u"""
        return np.vstack([
            y[1],
            -y[0] + y[1] * (1 - y[0] ** 2) + u(t),
            -y[3] - 2 * y[0],
            2 * y[0] * y[1] * y[2] + y[0] ** 2 * y[3] + y[2] - y[3] - 2 * y[1]
        ])

    def bc(self, ya, yb):
        """Boundary conditions"""
        return np.array([
            ya[0] - self.ini[0],
            ya[1] - self.ini[1],
            yb[2],
            yb[3]
        ])

    def L2(self, x):
        """
        Compute the L2 norm over the span
        Input:
            x:  1 x grid.size
                Coefficients of the vector
        Output:
            L2 norm of x
        """
        
        norm = 0
        for i in range(self.grid.size - 1):
            mesh = self.grid[i + 1] - self.grid[i]
            x_sq = x[i] ** 2
            norm += mesh * x_sq
        return norm

    def sol_BVP(self, u):
        """
        Solve the TVBVP. Given the piecewise constant control u. Return the 
        solution class with sol_bvp
        
        Input:
            u:      1 x grid.size array
                    Coefficient vector of the piecewise control
            
        Return:
            sol:    class (solution)
        
        """
        def u_func(t):
            conditions = [(t >= self.grid[i]) & (t < self.grid[i+1]) 
                         for i in range(self.grid.size - 1)]
            return np.piecewise(t, conditions, u)
        
        sol = solve_bvp(
            lambda t, y: self.BVP_VDP(t, y, u_func),
            self.bc,
            self.grid,
            self.guess,
            tol=1e-6,
            max_nodes = 10000
        )
        return sol
    
    def BBstep(self,u0, u1, G0, G1, grid0, grid1):
        # Interpolate u1 and G1 onto grid0
        u1_interp = np.interp(grid0, grid1, u1)
        G1_interp = np.interp(grid0, grid1, G1)
        
        s = u1_interp - u0
        y = G1_interp - G0
        
        sy = 0
        ss = 0
        for i in range(len(grid0)-1):
            mesh = grid0[i + 1] - grid0[i]
            sy += s[i] * y[i] * mesh
            ss += s[i] * s[i] * mesh
        
        return sy/ss, ss/sy
   
    def Grad_VDP(self, sol, u):
        """C
        ompute gradient: sol.y is evaluated on different grid as self.grid. 
        The function interpolate the solution of sol and return gradient vector
        evaluated on self.grid
        Input:
            sol:    class of solution
                    Solution of ode at control u
            u:      1 x grid.size
                    The piecewise constant control
        
        Outout:
            grad:   1 x grid.size array
                    Interpolated gradient of J(u) 
        """
        p_grid = sol.x
        p_interp = np.interp(self.grid, p_grid, sol.y[3,:])
        grad = np.zeros(self.grid.size)
        
        for i in range(self.grid.size - 1):
            grad[i] = p_interp[i] + 2 * self.beta * u[i]
        return grad

    def optimize(self):
        """Run optimization loop"""
        # Initialize
        u0 = np.zeros(self.grid.size)
        sol0 = self.sol_BVP(u0)
        G0 = - self.Grad_VDP(sol0, u0)
        
        # First step
        u1 = - (1/10) * G0
        sol1 = self.sol_BVP(u1)
        G1 = self.Grad_VDP(sol1, u1)
        
        
        alpha = 1/20
        # Optimization loop
        k = 1
        while self.L2(G1) >= self.tol:
            # if k % 2 == 0:
            #     alpha = 1 / self.BBstep(u0, u1, G0, G1, sol0.x, sol1.x)[0]
            # else:
            #     alpha = 1 / self.BBstep(u0, u1, G0, G1, sol0.x, sol1.x)[1]
            
            u0, G0, sol0 = u1, G1, sol1
            u1 = u1 - alpha * G1
            sol1 = self.sol_BVP(u1)
            G1 = self.Grad_VDP(sol1, u1)
            k += 1
            print(f" k = {k}")
            print(f" alpha = {alpha}")
            print(f" norm u = {self.L2(u1)}")
            print(f" norm G = {self.L2(G1)}")
            
        # Final result
        p = np.array([
            sol1.y[2, 0],
            sol1.y[3, 0]
        ])
        
        cost = self.L2(u1) * self.beta + 0.5 * (self.L2(sol1.y[0, :]) + self.L2(sol1.y[1, :]))
        print(f" cost = {cost}")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory
        t = sol1.x
        x = sol1.y[0, :]
        y = sol1.y[1, :]
        ax.plot(t, x, y, label='Solution Trajectory')
        
        # Labels and title
        ax.set_xlabel('t')
        ax.set_ylabel('x(t)')
        ax.set_zlabel('y(t)')
        ax.set_title('Solution Trajectory in Phase Space')
        
        # Add legend
        ax.legend()
        
        plt.show()    
        return p, cost


        

def test_optimizer():
    """Test the OpenLoopOptimizer class"""
    span = [0, 3]
    grid = np.linspace(span[0], span[1], 1000)
    guess = np.ones((4, grid.size))
    beta = 2
    tol = 1e-4
    ini = [-1, 1]
    
    optimizer = OpenLoopOptimizer(span, grid, guess, beta, tol, ini)
    p, sol = optimizer.optimize()
    return p, sol

if __name__ == "__main__":
    p, sol = test_optimizer()
    print("Final p:", p)