#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 22:10:37 2024

@author: chaoruiz
"""

import torch
import numpy as np


class ProblemNN2D:
    def __init__(self, delta, force_upper):
        self.delta = delta
        self.gamma = 0.01
        self.N = 1
        self.dim = 2
        self.L = 1
        self.R = torch.sqrt(torch.tensor(self.dim)) * self.L
        RO = self.R + torch.sqrt(1 + self.R**2)
        rO = -self.R + torch.sqrt(1 + self.R**2)
        self.Omega = [rO.item(), RO.item()]
        self.force_upper = force_upper
        self.u_zero = {'x': torch.zeros(self.dim, 0), 'u': torch.zeros(self.N, 0)}

        # Observation set
        Nobs = 21**2
        Nobs1 = int(torch.floor(Nobs ** (1 / self.dim)))
        x1 = torch.linspace(-self.L, self.L, Nobs1)
        x2 = torch.linspace(-self.L, self.L, Nobs1)
        X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
        self.xhat = torch.stack([X1.flatten(), X2.flatten()], dim=0)

    def kernel(self, xhat, x):
        """
        Compute the kernel and optionally its derivative.
        """
        Nx = x.shape[1]
        Nxh = xhat.shape[1]

        X = xhat.unsqueeze(2).expand(-1, -1, Nx)
        Xhat = x.unsqueeze(2).expand(-1, -1, Nxh).transpose(1, 2)

        x2 = (X**2).sum(dim=0)
        xxhat = (X * Xhat).sum(dim=0)
        y = (2 * xxhat + 1 - x2) / (1 + x2)

        absy = torch.sqrt(self.delta**2 + y**2)
        if not self.force_upper:
            k = 0.5 * (absy + y)
        else:
            k = 0.5 * absy
        return k

    def K(self, xhat, u):
        """
        Compute the kernel result and optionally its derivative.
        """
        k = self.kernel(xhat, u['x'])
        Ku = torch.matmul(k, u['u'].view(-1, 1)).squeeze()
        return Ku

    def Ks(self, x, xhat, y):
        """
        Adjoint kernel computation.
        """
        k = self.kernel(xhat, x)
        Ksy = torch.matmul(k.T, y)
        return Ksy

    def Phi(self, gamma):
        """
        Define Phi and its derivative for optimization.
        """
        phi = {'gamma': gamma}
        if gamma == 0:
            phi['phi'] = lambda t: t
            phi['dphi'] = lambda t: torch.ones_like(t)
            phi['ddphi'] = lambda t: torch.zeros_like(t)
            phi['inv'] = lambda y: y
        else:
            th = 0.5
            gam = gamma / (1 - th)

            phi['phi'] = lambda t: th * t + (1 - th) * torch.log(1 + gam * t) / gam
            phi['dphi'] = lambda t: th + (1 - th) / (1 + gam * t)
            phi['ddphi'] = lambda t: -(1 - th) * gam / (1 + gam * t)**2
            phi['inv'] = lambda y: y / th
        return phi

    def optimize_u(self, y_d, alpha, phi, u):
        """
        Optimize the coefficients u for a fixed set of points x.
        """
        Kred = self.kernel(self.xhat, u['x'])
        ured = u['u'].clone()

        # Define objective function for SSN optimization
        def obj(ured):
            y = Kred @ ured.view(-1, 1)
            loss = 0.5 * torch.norm(y - y_d)**2 + alpha * phi['phi'](ured.abs()).sum()
            return loss

        # Gradient-based optimization using PyTorch's autograd
        ured.requires_grad_(True)
        optimizer = torch.optim.LBFGS([ured], max_iter=50, tolerance_grad=1e-5, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = obj(ured)
            loss.backward()
            return loss

        optimizer.step(closure)
        u_opt = {'x': u['x'], 'u': ured.detach()}
        return u_opt

    def plot_forward(self, u, y_d):
        """
        Plot the forward result.
        """
        import matplotlib.pyplot as plt

        x1 = torch.linspace(-self.L, self.L, 30)
        x2 = torch.linspace(-self.L, self.L, 30)
        X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
        points = torch.stack([X1.flatten(), X2.flatten()], dim=0)

        Y = self.K(points, u)
        Y = Y.view(30, 30)

        fig, ax = plt.subplots()
        c = ax.contourf(X1.numpy(), X2.numpy(), Y.numpy(), levels=50)
        plt.colorbar(c)
        plt.scatter(self.xhat[0, :].numpy(), self.xhat[1, :].numpy(), c=y_d.numpy(), cmap='coolwarm', edgecolor='k')
        plt.show()


# Example usage
delta = 0.1
force_upper = False
problem = ProblemNN2D(delta, force_upper)
u = {'x': torch.rand(2, 10), 'u': torch.rand(10)}
y_d = torch.rand(problem.xhat.shape[1])
alpha = 0.01
phi = problem.Phi(problem.gamma)

# Optimize u
u_opt = problem.optimize_u(y_d, alpha, phi, u)
problem.plot_forward(u_opt, y_d)
