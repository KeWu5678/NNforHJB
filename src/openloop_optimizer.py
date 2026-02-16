#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:30:52 2024

@author: chaoruiz
"""


import numpy as np
from src import utils 
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
        BB1 = ss / (sy + epsilon)
        BB2 = sy / (yy + epsilon)
        return BB1, BB2, sy
   

    def optimize(self):
        """Run optimization loop"""
        # Initialize
        num_basis = 50
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

        # Initial cost at first iterate
        V1 = self.V(sol1.x, u1(sol1.x), sol1.y[0,:], sol1.y[1,:])

        # Initial step size (will be adapted)
        alpha = 0.1

        # Optimization loop
        k = 1
        store = True
        alpha_min, alpha_max = 1e-5, 10.0      # bounds for BB step
        alpha_default = 0.1
        ls_beta = 0.5                          # backtracking factor for line search
        ls_max_iter = 500                      # max line-search iterations per outer iter
        ls_tol = 1e-2                          # relative tolerance for cost decrease (allows numerical noise)

        # NOTE: stopping still uses ||G|| as you had; only LS criterion changed
        while utils.L2(grid1, G1) >= self.tol and store:
            # BB step based on coefficients
            alpha_BB1, alpha_BB2, sy = self.BBstep(u0_coeff, u1_coeff, G0_coeff, G1_coeff)

            # Select BB1 or BB2
            if k % 2 == 0:
                alpha_trial = alpha_BB1
            else:
                alpha_trial = alpha_BB2

            # Safeguards: reject bad curvature or nonsensical steps
            if sy <= 0 or alpha_trial <= 0 or np.isnan(alpha_trial) or np.isinf(alpha_trial):
                alpha = alpha_default
            else:
                alpha = np.clip(alpha_trial, alpha_min, alpha_max)

            # ------------------------------------------------------------------
            # Backtracking line search on COST V 
            # ------------------------------------------------------------------
            d_coeff = -G1_coeff          # descent direction in coefficient space
            alpha_ls = alpha
            accepted = False

            # previous cost
            J_prev = V1

            ls_iter = 0
            while ls_iter < ls_max_iter:
                # Trial step in coefficient space
                u_trial_coeff = u1_coeff + alpha_ls * d_coeff
                u_trial = utils.gen_legendre(u_trial_coeff)

                try:
                    # Solve BVP with new control
                    sol_trial = self.sol_BVP(u_trial)
                    grid_trial = sol_trial.x
                    G_trial = self.gradient(u_trial(grid_trial), sol_trial.y[3,:])
                    G_trial_coeff = utils.fit_legendre(grid_trial, G_trial, num_basis)

                    # Compute cost at trial point
                    V_trial = self.V(grid_trial,
                                    u_trial(grid_trial),
                                    sol_trial.y[0,:],
                                    sol_trial.y[1,:])
                except Exception as e_ls:
                    # Treat any BVP failure as a bad step: force alpha shrink
                    V_trial = np.inf

                # Accept if cost decreased (relative decrease requirement handles numerical noise)
                if np.isfinite(V_trial) and V_trial <= J_prev * (1 + ls_tol):
                    accepted = True
                    break

                # Otherwise shrink step
                alpha_ls *= ls_beta
                if alpha_ls < alpha_min:
                    break
                ls_iter += 1

            if not accepted:
                print("Warning: line search failed to find a decreasing-cost step.")
                store = False
                break

            # Accept the line-search result
            u0_coeff, G0_coeff = u1_coeff, G1_coeff   # previous becomes "old"
            u1_coeff, G1_coeff = u_trial_coeff, G_trial_coeff
            u1 = u_trial
            sol1 = sol_trial
            grid1 = grid_trial
            G1 = G_trial
            V1 = V_trial
            alpha = alpha_ls

            # Update iteration counter and report
            k += 1
            G1_norm = utils.L2(grid1, G1)
            print(f" k = {k}, alpha = {alpha}, cost = {V1}, norm G = {G1_norm}")

            # Check for numerical instability
            if np.isnan(G1_norm) or np.isinf(G1_norm) or np.isnan(V1) or np.isinf(V1):
                print("Warning: Numerical instability detected. Terminating.")
                store = False
                break

            # Check max iterations
            if k >= self.max_it:
                store = False
                print("Maximum iterations reached")
                print(f" norm G = {G1_norm}")
                break

        # Final result
        if store == True:
            p = np.array([
                sol1.y[2, 0],
                sol1.y[3, 0]
            ])
            V = V1
            print(f" cost = {V}")
            print(f" norm G = {utils.L2(sol1.x, G1)}")
            return p, V
        else:
            print("maximum iteration reached")
            return None, None

