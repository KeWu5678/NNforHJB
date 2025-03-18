#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:30:52 2024

@author: chaoruiz
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import discretization as dis
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
            tol=1e-6,
            max_nodes = 10000
        )
        return sol
    
    def BBstep(self, u0, u1, G0, G1):
        sy, ss, yy = 0, 0, 0
        s = u1 - u0
        y = G1 - G0
        
        sy = np.dot(s,y)
        ss = np.dot(s,s)
        yy = np.dot(y,y)
        
        BB1 = sy/ss
        BB2 = yy/sy
        return BB1, BB2
   

    def optimize(self):
        """Run optimization loop"""
        # Initialize
        u0 = dis.gen("pw", np.zeros(self.grid.size), self.grid)
        sol0 = self.sol_BVP(u0)
        G0 = self.gradient(u0(sol0.x), sol0.y[3,:])
        
        # Store the grid where G0 is evaluated
        grid0 = sol0.x

        u1_coeff = - (1/10) * G0
        u1 = dis.gen("pw", u1_coeff, sol0.x)
        sol1 = self.sol_BVP(u1)
        G1 = self.gradient(u1(sol1.x), sol1.y[3,:])

        # alpha = 1/30
        # Optimization loop
        k = 1
        store = True
        while dis.L2(sol1.x, G1) >= self.tol and store:
            bb_step = self.BBstep(u0(sol1.x), u1(sol1.x), np.interp(sol1.x, grid0, G0), G1)
            if k % 2 == 0:
                alpha = 1 / bb_step[0]
            else:
                alpha = 1 / bb_step[1]
            
            u2_coeff = u1(sol1.x) - alpha * G1
            u0, G0, grid0 = u1, G1, sol1.x

            u1 = dis.gen("pw", u2_coeff, sol1.x)
            sol1 = self.sol_BVP(u1)
            G1 = self.gradient(u1(sol1.x), sol1.y[3,:])

            k += 1

            # print(f" k = {k}")
            # print(f" alpha = {alpha}")
            # print(f" norm G = {dis.L2(sol1.x, G1)}")
            
            if k == self.max_it:
                store = False
                break
        # Final result
        if store == True:
            p = np.array([
                sol1.y[2, 0],
                sol1.y[3, 0]
            ])
            V = self.V(sol1.x, u1(sol1.x), sol1.y[0,:], sol1.y[1,:])
            print(f" cost = {V}")
            print(f" norm G = {dis.L2(sol1.x, G1)}")

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
    


