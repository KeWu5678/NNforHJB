#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 22:42:54 2024

@author: chaoruiz
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# Function definitions
def f_d(x):
    return torch.abs(torch.sin(7 * (1 + torch.abs(x) ** 2) ** (1 / 2)) * torch.exp(-x ** 2 / 2))

def PDAPmultisemidiscrete(p, y_d, alpha, phi, alg_opts):
    # Placeholder implementation of PDAPmultisemidiscrete
    # Replace with your specific algorithm
    u = torch.zeros_like(p['xhat'])  # Example initial solution
    return {'x': u}, None

# Problem setup
def setup_problem_NN_stereo(epsilon, force_upper):
    xhat = torch.linspace(-5, 5, 100)  # Example problem setup
    Phi = lambda p, gamma: lambda u: u  # Placeholder for Phi function
    K = lambda p, xhat, u: xhat + u    # Placeholder for K function
    obj = {'F': lambda x: torch.sum(x ** 2),
           'dF': lambda x: 2 * x}      # Quadratic loss example
    plot_adjoint = lambda p, u, adj, alpha: plt.plot(u['x'].detach().numpy())  # Dummy plot
    plot_forward = lambda p, u, y: plt.plot(y.detach().numpy())               # Dummy plot
    postprocess = lambda p, u, tol: u                                        # Identity function

    return {'xhat': xhat, 'Phi': Phi, 'K': K, 'obj': obj,
            'plot_adjoint': plot_adjoint, 'plot_forward': plot_forward,
            'postprocess': postprocess}

# Initialize problem
epsilon = 0.0001
force_upper = False
p = setup_problem_NN_stereo(epsilon, force_upper)

# Generate data
y_d = f_d(p['xhat']).detach()

alpha = 0.00001
gamma = 0
phi = p['Phi'](p, gamma)

alg_opts = {
    'optimize_x': True,
    'max_step': 15,
    'plot_every': 5,
    'TOL': 1e-6
}

# Solve L1 problem
u_l1, alg_l1 = PDAPmultisemidiscrete(p, y_d, alpha, phi, alg_opts)

# Postprocess and plot
plt.figure(1)
u_l1_pp = p['postprocess'](p, u_l1, 1e-3)
p['plot_adjoint'](p, u_l1_pp, p['obj']['dF'](p['K'](p, p['xhat'], u_l1_pp) - y_d), alpha)
plt.figure(2)
p['plot_forward'](p, u_l1_pp, y_d)
plt.show()

Nnodes_l1 = len(u_l1_pp['x'])
l2_err_l1 = torch.sqrt(2 * p['obj']['F'](p['K'](p, p['xhat'], u_l1_pp) - y_d))
print(f"L1 Nodes: {Nnodes_l1}, L2 Error: {l2_err_l1.item()}")

# Nonconvex problem
gammas = [1e-4, 1e-3, 1e-2, 1e-1, 1]
u_opt = []
alg_out = []
Nnodes_phi = []
l2_err_phi = []

for gamma in gammas:
    phi = p['Phi'](p, gamma)
    u, alg = PDAPmultisemidiscrete(p, y_d, alpha, phi, alg_opts)
    u_opt.append(u)
    alg_out.append(alg)

    plt.figure(3)
    p['plot_adjoint'](p, u, p['obj']['dF'](p['K'](p, p['xhat'], u) - y_d), alpha)
    plt.figure(4)
    p['plot_forward'](p, u, y_d)
    plt.show()

    Nnodes_phi.append(len(u['x']))
    l2_err_phi.append(torch.sqrt(2 * p['obj']['F'](p['K'](p, p['xhat'], u) - y_d)))

print(f"Nonconvex Results:")
for i, gamma in enumerate(gammas):
    print(f"Gamma: {gamma}, Nodes: {Nnodes_phi[i]}, L2 Error: {l2_err_phi[i].item()}")
