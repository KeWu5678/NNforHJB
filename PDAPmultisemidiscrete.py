#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 20:23:33 2024

@author: chaoruiz
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from typing import NamedTuple, Optional, Dict, List
from dataclasses import dataclass

@dataclass
class AlgorithmOptions:
    max_step: int = 1000
    tol: float = 1e-5
    update_m0: bool = True
    sparsification: bool = False
    optimize_x: bool = False
    plot_final: bool = True
    plot_every: int = 0
    u0: Optional[torch.Tensor] = None

@dataclass
class AlgorithmOutput:
    us: List[torch.Tensor]  # List of solutions at each iteration
    js: List[float]  # Objective values
    supps: List[int]  # Support sizes
    tics: List[float]  # Time measurements
    psis: List[float]  # Upper bounds

class Problem:
    """Base class for problem setup"""
    def __init__(self, N: int, xhat: torch.Tensor):
        self.N = N
        self.xhat = xhat
        self.device = xhat.device
        
    def K(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Evaluate operator K at points x with coefficients u"""
        raise NotImplementedError
        
    def Ks(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Evaluate adjoint operator K* at points x with coefficients y"""
        raise NotImplementedError
        
    def kernel(self, x: torch.Tensor) -> torch.Tensor:
        """Compute kernel values and derivatives"""
        raise NotImplementedError

class Phi:
    """Penalty function class"""
    def __init__(self, gamma: float):
        self.gamma = gamma
        
    def phi(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate penalty function"""
        if self.gamma == 0:
            return t
        th = 0.5
        gam = self.gamma/(1-th)
        return th * t + (1-th) * torch.log(1 + gam * t) / gam
        
    def dphi(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate derivative of penalty function"""
        if self.gamma == 0:
            return torch.ones_like(t)
        th = 0.5
        gam = self.gamma/(1-th)
        return th + (1-th) / (1 + gam * t)
        
    def inv(self, y: torch.Tensor) -> torch.Tensor:
        """Evaluate inverse of penalty function"""
        if self.gamma == 0:
            return y
        th = 0.5
        return y / th
        
    def prox(self, sigma: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Evaluate proximal operator"""
        if self.gamma == 0:
            return torch.maximum(g - sigma, torch.zeros_like(g))
        th = 0.5
        gam = self.gamma/(1-th)
        temp = g - sigma*th - 1/gam
        return 0.5 * (temp + torch.sqrt(temp**2 + 4*(g - sigma)/gam)) * (g >= sigma)

def compute_norm(u: torch.Tensor, N: int) -> torch.Tensor:
    """Compute block norms"""
    return torch.norm(u.reshape(-1, N), dim=1)

def pdap_multisemidiscrete(p: Problem, 
                          y_ref: torch.Tensor,
                          alpha: float,
                          phi: Phi,
                          alg_opts: Optional[AlgorithmOptions] = None) -> tuple[torch.Tensor, AlgorithmOutput]:
    """
    Solve the optimization problem:
    min (1/2)*|Ku - y_ref|^2 + alpha*phi(|u|_1,2)
    """
    if alg_opts is None:
        alg_opts = AlgorithmOptions()
        
    # Initialize algorithm output
    alg_out = AlgorithmOutput([], [], [], [], [])
    
    # Initial guess
    uk = alg_opts.u0 if alg_opts.u0 is not None else torch.zeros(0, device=p.device)
    
    # Initial values
    Ku = p.K(uk.x, uk.u)
    norms_u = compute_norm(uk.u, p.N)
    j = 0.5 * torch.norm(Ku - y_ref)**2 + alpha * torch.sum(phi.phi(norms_u))
    suppsize = torch.count_nonzero(norms_u)
    
    # Algorithm parameter M0
    M0 = min(phi.inv(0.5 * torch.norm(-y_ref)**2/alpha).item(), 1e8)
    
    # Save initial state
    alg_out.us.append(uk)
    alg_out.js.append(j.item())
    alg_out.supps.append(suppsize.item())
    alg_out.tics.append(0)
    alg_out.psis.append(0)
    
    start_time = time.time()
    print(f'PDAP: {0:3d}, desc: ({0:1.0e},{0:1.0e}), supp: {suppsize}, j: ({j:1.2e},inf), M0: {M0:1.2e}')
    
    for iter in range(1, alg_opts.max_step + 1):
        # Compute gradient maxima
        yk = Ku - y_ref
        xinit = uk.x if not alg_opts.optimize_x else torch.zeros(0, device=p.device)
        xmax, grad = find_max(p, yk, xinit)
        
        # Dynamic update of bounding parameter
        if alg_opts.update_m0:
            M0 = min(phi.inv(j/alpha).item(), 1e6)
            
        # Compute gradient norms and sort
        norms_grad = compute_norm(grad, p.N)
        perm = torch.argsort(norms_grad, descending=True)
        xmax = xmax[:, perm]
        grad = grad[:, perm]
        norms_grad = norms_grad[perm]
        
        # Select promising points
        buffer = 0.5 * (norms_grad.max() - alpha)
        cutoff = alpha + buffer
        locs = torch.where(norms_grad >= min(alpha + buffer, norms_grad))[0]
        
        # Limit number of new points
        npoint = 15
        locs = locs[:min(npoint, len(locs))]
        
        xmax = xmax[:, locs]
        grad = grad[:, locs]
        norms_grad = norms_grad[locs]
        
        max_grad, loc = torch.max(norms_grad, dim=0)
        
        # Compute descent direction
        coeff = -torch.sqrt(torch.finfo(grad.dtype).eps) * grad / norms_grad.unsqueeze(0)
        coeff[:, loc] = -grad[:, loc] / norms_grad[loc]
        
        newsupp = torch.cat([uk.x, xmax], dim=1)
        vhat_u = torch.cat([torch.zeros_like(uk.u), coeff], dim=1)
        uk_new_u = torch.cat([uk.u, torch.zeros_like(coeff)], dim=1)
        
        # Compute slope and curvature
        Kvhat = p.K(newsupp, vhat_u)
        phat = torch.real(torch.sum(Kvhat * yk))
        what = torch.real(torch.sum(Kvhat * Kvhat))
        
        # Upper bound for global functional error
        phi_u = alpha * torch.sum(phi.phi(norms_u))
        phi_vhat = alpha * phi.phi(M0 * torch.sum(compute_norm(vhat_u, p.N)))
        upperb = phi_u + torch.real(torch.sum(Ku * yk)) - min(phi_vhat + M0*phat, 0.)
        
        # Check termination
        if upperb <= alg_opts.tol * j or iter >= alg_opts.max_step:
            print(f'PDAP: {iter:3d}, desc: ({0:1.0e},{0:1.0e}), supp: {suppsize}, ' +
                  f'j: ({j:1.2e},{upperb:1.0e}), M0: {M0:1.2e}')
            alg_out.psis.append(upperb.item())
            break
            
        # Coordinate descent step
        if phat <= -alpha:
            tau = phi.prox(alpha/what, -phat/what)
        else:
            tau = 0
            
        # Update solution
        uk = torch.cat([uk_new_u, tau * vhat_u], dim=1)
        uk.x = newsupp
        
        # Update values
        Ku = p.K(uk.x, uk.u)
        norms_u = compute_norm(uk.u, p.N)
        newj1 = 0.5 * torch.norm(Ku - y_ref)**2 + alpha * torch.sum(phi.phi(norms_u))
        supp1 = len(norms_u)
        
        # Optional sparsification
        if alg_opts.sparsification:
            uk = sparsify(uk, p, p.N)
            
        # Optional position optimization
        if alg_opts.optimize_x:
            uk = optimize_xu(uk, p, y_ref, alpha, phi)
            
        # Optimize coefficients
        uk = optimize_u(uk, p, y_ref, alpha, phi)
        
        # Clean up solution
        norms_u = compute_norm(uk.u, p.N)
        supp_ind = torch.where(norms_u > 0)[0]
        uk.u = uk.u[:, supp_ind]
        uk.x = uk.x[:, supp_ind]
        
        # Save old objective value
        oldj = j
        
        # Update values
        Ku = p.K(uk.x, uk.u)
        norms_u = compute_norm(uk.u, p.N)
        j = 0.5 * torch.norm(Ku - y_ref)**2 + alpha * torch.sum(phi.phi(norms_u))
        suppsize = torch.count_nonzero(norms_u)
        
        # Output progress
        print(f'PDAP: {iter:3d}, desc: ({oldj-newj1:1.0e},{newj1-j:1.0e}), ' +
              f'supp: {supp1}->{suppsize}, j: ({j:1.2e},{upperb:1.0e}), M0: {M0:1.2e}')
        
        # Save diagnostics
        alg_out.us.append(uk)
        alg_out.js.append(j.item())
        alg_out.supps.append(suppsize.item())
        alg_out.tics.append(time.time() - start_time)
        alg_out.psis.append(upperb.item())
        
        # Optional plotting
        if alg_opts.plot_every > 0 and iter % alg_opts.plot_every == 0:
            plot_solution(p, uk, y_ref, Ku)
            
    # Final plot if requested
    if alg_opts.plot_final:
        plot_solution(p, uk, y_ref, Ku)
        
    return uk, alg_out

def plot_solution(p: Problem, u: torch.Tensor, y_ref: torch.Tensor, Ku: torch.Tensor):
    """Plot current solution"""
    plt.figure(1)
    plt.subplot(2,1,1)
    plt.plot(u.x.cpu().numpy(), u.u.cpu().numpy(), 'ko-')
    plt.grid(True)
    plt.title('Solution nodes and coefficients')
    
    plt.subplot(2,1,2) 
    plt.plot(p.xhat.cpu().numpy(), y_ref.cpu().numpy(), 'b--', label='Reference')
    plt.plot(p.xhat.cpu().numpy(), Ku.cpu().numpy(), 'k-', label='Current')
    plt.grid(True)
    plt.legend()
    plt.title('Function approximation')
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)