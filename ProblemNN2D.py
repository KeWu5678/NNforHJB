#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 14:27:54 2024

@author: chaoruiz
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple, Callable


@dataclass
class Problem:
    delta: float
    force_upper: bool
    N: int = 1
    dim: int = 1
    R: float = 1.0
    n_points: int = 1000

    def __init__(self, delta: float, force_upper: bool):
        self.delta = delta
        self.force_upper = force_upper
        self.R = 1.0
        self.RO = self.R + torch.sqrt(torch.tensor(1.0 + self.R ** 2))
        self.Omega = [-self.RO.item(), self.RO.item()]
        self.xhat = torch.linspace(-self.R, self.R, self.n_points)
        self.u_zero = {'x': torch.zeros(1, 0), 'u': torch.zeros(1, 0)}

    def kernel(self, xhat: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.numel() == 0:
            return torch.zeros(xhat.numel(), 0), torch.zeros(xhat.numel(), 0)

        # Ensure 1D tensors
        xhat = xhat.reshape(-1)
        x = x.reshape(-1)

        X = x.unsqueeze(0).expand(xhat.numel(), -1)
        Xhat = xhat.unsqueeze(1).expand(-1, x.numel())

        opx2 = 1 + X.pow(2)
        b = (2 - opx2) / opx2
        a = 2 * X / opx2

        dbdx = -2 * a / opx2
        dadx = 2 * b / opx2

        y = a * Xhat + b
        dydx = dadx * Xhat + dbdx

        absy = torch.sqrt(self.delta ** 2 + y.pow(2))

        if not self.force_upper:
            k = 0.5 * (absy + y)
            dk = 0.5 * (y / absy + 1) * dydx
        else:
            k = 0.5 * absy
            dk = 0.5 * (y / absy) * dydx

        return k, dk

    def K(self, xhat: torch.Tensor, u: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        k, dk = self.kernel(xhat, u['x'])
        Ku = k @ u['u'].T
        dKu = dk @ u['u'].T
        return Ku, dKu

    def Ks(self, x: torch.Tensor, xhat: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        k, dk = self.kernel(xhat, x)
        Ksy = k.T @ y
        dKsy = dk.T @ y
        return Ksy, dKsy

    def tracking_objective(self) -> Callable:
        Nx = self.xhat.numel()

        def F(y: torch.Tensor) -> torch.Tensor:
            return 1 / (2 * Nx) * torch.norm(y) ** 2

        def dF(y: torch.Tensor) -> torch.Tensor:
            return 1 / Nx * y

        def ddF(y: torch.Tensor) -> torch.Tensor:
            return 1 / Nx * torch.ones_like(y)

        return {'F': F, 'dF': dF, 'ddF': ddF}

    def postprocess(self, u: dict, pp_radius: float) -> dict:
        x = u['x'].squeeze()
        indices = torch.argsort(x)
        x = x[indices]
        u_vals = u['u'].squeeze()[indices]

        diffs = torch.diff(x) > pp_radius
        cut_indices = torch.cat([torch.tensor([0]),
                                 torch.where(diffs)[0] + 1,
                                 torch.tensor([len(x)])])

        X, U = [], []
        for i in range(len(cut_indices) - 1):
            start, end = cut_indices[i], cut_indices[i + 1]
            cluster_x = x[start:end]
            cluster_u = u_vals[start:end]
            U.append(cluster_u.sum())
            X.append((cluster_x * torch.abs(cluster_u)).sum() / torch.abs(cluster_u).sum())

        return {'x': torch.tensor(X).unsqueeze(1), 'u': torch.tensor(U).unsqueeze(1)}

    def optimize_u(self, y_d: torch.Tensor, alpha: float, u: dict) -> dict:
        k, _ = self.kernel(self.xhat, u['x'])
        u_opt = u['u'].clone().requires_grad_(True)
        optimizer = optim.LBFGS([u_opt])

        def closure():
            optimizer.zero_grad()
            y = k @ u_opt.T
            obj = self.tracking_objective()
            loss = obj['F'](y - y_d) + alpha * torch.abs(u_opt).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        return {'x': u['x'].clone(), 'u': u_opt.detach()}

    def find_max(self, y: torch.Tensor, x0: Optional[torch.Tensor] = None, n_guess: int = 50) -> torch.Tensor:
        y_norm = y / torch.norm(y)

        if x0 is None or x0.size(1) > n_guess // 2:
            if x0 is not None:
                idx = torch.randperm(x0.size(1))[:n_guess // 2]
                x0 = x0[:, idx]
            n_guess = n_guess - (0 if x0 is None else x0.size(1))

            # Generate random points
            randomb = torch.sort(self.xhat.min() + (self.xhat.max() - self.xhat.min()) *
                                 torch.rand(n_guess))[0]
            randomx = 1 / torch.sqrt(1 + randomb.pow(2))
            randomb = randomb / torch.sqrt(1 + randomb.pow(2))
            randomx = randomx / (1 + randomb)
            randomx = torch.sign(torch.rand(randomx.size()) - 0.5) * randomx

            x0 = torch.cat([torch.zeros(1), randomx,
                            x0.flatten() if x0 is not None else torch.tensor([])])

        x = x0.clone().requires_grad_(True)
        optimizer = optim.LBFGS([x], max_iter=500)

        def closure():
            optimizer.zero_grad()
            Ksy, _ = self.Ks(x.unsqueeze(0), self.xhat, y_norm)
            loss = -0.5 * Ksy.pow(2).sum()
            loss.backward()
            return loss

        for _ in range(10):
            optimizer.step(closure)

        xmax = x.detach()
        Ksy, _ = self.Ks(xmax.unsqueeze(0), self.xhat, y)
        mask = torch.abs(Ksy) > 0.25 * torch.abs(Ksy).max()
        xmax = xmax[mask]

        return self.postprocess({'x': xmax.unsqueeze(1),
                                 'u': torch.ones(len(xmax)).unsqueeze(1)},
                                1e-4)['x']


def test():
    problem = Problem(delta=0.01, force_upper=False)
    y_d = torch.sin(2 * torch.pi * problem.xhat)
    u = {'x': torch.randn(1, 5), 'u': torch.randn(1, 5)}
    u_opt = problem.optimize_u(y_d, 0.1, u)
    print("Optimized weights shape:", u_opt['u'].shape)


if __name__ == "__main__":
    test()