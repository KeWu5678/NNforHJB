import ProblemNN2D as p2d
from ProblemNN2D import NN2D, NeuralMeasure, Phi
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class AlgOpts:
    """Algorithm options"""
    max_step: int = 1000
    TOL: float = 1e-5
    update_M0: bool = True
    sparsification: bool = False
    optimize_x: bool = False
    plot_final: bool = True
    plot_every: int = 0
    u0: Optional[Dict] = None


class PDAPOptimizer:
    def __init__(self, problem: NN2D, optimizer_opts: Optional[AlgOpts] = None):
        """
        Args:
            problem: Problem2D instance containing kernel, optimization methods etc.
            optimizer_opts: Optional optimization parameters
        """
        self.problem = problem
        # with AlgOpt() the default instance created
        self.opts = optimizer_opts or AlgOpts()
        

    def compute_norm(self, u: torch.Tensor, N: int) -> torch.Tensor:
        """Compute norm of reshaped tensor"""
        return torch.norm(u.reshape(N, -1), dim=0)
    

    def optimize(self, y_ref: torch.Tensor, alpha: float, phi: Phi) -> Tuple[NeuralMeasure, Dict]:
        N = self.problem.N
        obj = self.problem.obj
        
        # Initial guess
        uk = self.opts.u0 if self.opts.u0 is not None else self.problem.u_zero
        
        # Initial values using Problem2D methods
        Ku = self.problem.K(self.problem.xhat, uk)[0]
        norms_u = self.compute_norm(uk.u, N)
        j = obj.F(Ku - y_ref) + alpha * torch.sum(phi(norms_u))
        suppsize = torch.count_nonzero(norms_u)
        
        # Algorithm parameters
        M0 = min(phi.inv(obj.F(-y_ref)/alpha).item(), 1e8)
        
        alg_out = {
            'us': [uk],
            'js': [j.item()],
            'supps': [suppsize.item()],
            'tics': [0.0],
            'Psis': []
        }
        
        print(f'PDAP: {0:3d}, desc: (0.0e+00,0.0e+00), supp: {suppsize}, ' +
              f'j: ({j:.2e},inf), M0: {M0:.2e}')
        
        iter = 1
        while True:
            # Compute maxima of gradient using Problem2D find_max
            yk = obj.dF(Ku - y_ref)
            xinit = uk.x if not self.opts.optimize_x else self.problem.u_zero.x
            xmax = self.problem.find_max(yk, xinit)
            
            # Use Problem2D's Ks method
            grad = self.problem.Ks(xmax, self.problem.xhat, yk)[0].reshape(self.problem.N, -1)
            
            if self.opts.update_M0:
                M0 = min(phi.inv(j/alpha).item(), 1e6)
            
            norms_grad = self.compute_norm(grad, N)
            
            perm = torch.argsort(norms_grad, descending=True)
            xmax = xmax[:, perm]
            grad = grad[:, perm]
            norms_grad = norms_grad[perm]
            
            buffer = 0.5 * (norms_grad.max() - alpha)
            cutoff = alpha + buffer
            locs = torch.where(norms_grad >= min(alpha + buffer, norms_grad))[0]
            
            Npoint = 15
            locs = locs[:min(Npoint, len(locs))]
            
            xmax = xmax[:, locs]
            grad = grad[:, locs]
            norms_grad = norms_grad[locs]
            
            max_grad, loc = torch.max(norms_grad, 0)
            
            coeff = -torch.sqrt(torch.finfo(torch.float32).eps) * grad / norms_grad
            coeff[:, loc] = -grad[:, loc] / norms_grad[loc]
            
            # Create NeuralMeasure instances
            newsupp = torch.cat([uk.x, xmax], dim=1)
            vhat = NeuralMeasure(
                x=newsupp,
                u=torch.cat([torch.zeros_like(uk.u), coeff], dim=1)
            )
            uk_new = NeuralMeasure(
                x=newsupp,
                u=torch.cat([uk.u, torch.zeros_like(coeff)], dim=1)
            )
            
            # Use Problem2D's kernel
            Kvhat = self.problem.K(self.problem.xhat, vhat)[0]
            phat = torch.real(Kvhat.T @ yk)
            what = torch.real(Kvhat.T @ obj.ddF(Ku - y_ref) @ Kvhat)
            
            phi_u = alpha * torch.sum(phi(norms_u))
            phi_vhat = alpha * phi(torch.sum(M0 * self.compute_norm(vhat.u, N)))
            upperb = phi_u + torch.real(Ku.T @ yk) - min(phi_vhat + M0*phat, torch.tensor(0.0))
            
            if upperb <= self.opts.TOL*j or iter > self.opts.max_step:
                print(f'PDAP: {iter:3d}, desc: (0.0e+00,0.0e+00), supp: {suppsize}, ' +
                      f'j: ({j:.2e},{upperb:.2e}), M0: {M0:.2e}')
                alg_out['Psis'].append(upperb.item())
                break
            
            if phat <= -alpha:
                tau = phi.prox(alpha/what, -phat/what)
            else:
                tau = 0.0
                
            # Update using NeuralMeasure
            uk = NeuralMeasure(
                x=uk_new.x.clone(),
                u=uk_new.u + tau * vhat.u
            )
            
            # Use Problem2D's methods for updates
            Ku = self.problem.K(self.problem.xhat, uk)[0]
            norms_u = self.compute_norm(uk.u, N)
            newj1 = obj.F(Ku - y_ref) + alpha * torch.sum(phi(norms_u))
            supp1 = len(norms_u)
            
            # Use Problem2D's optimize_u
            uk = self.problem.optimize_u(y_ref, alpha, phi, uk)
            
            norms_u = self.compute_norm(uk.u, N)
            supp_ind = torch.where(norms_u > 0)[0]
            uk = NeuralMeasure(
                x=uk.x[:, supp_ind],
                u=uk.u[:, supp_ind]
            )
            
            oldj = j
            
            Ku = self.problem.K(self.problem.xhat, uk)[0]
            norms_u = self.compute_norm(uk.u, N)
            j = obj.F(Ku - y_ref) + alpha * torch.sum(phi(norms_u))
            suppsize = torch.count_nonzero(norms_u)
            
            print(f'PDAP: {iter:3d}, desc: ({oldj-newj1:.1e},{newj1-j:.1e}), ' +
                  f'supp: {supp1}->{suppsize}, j: ({j:.2e},{upperb:.2e}), M0: {M0:.2e}')
            
            iter += 1
            alg_out['us'].append(uk)
            alg_out['js'].append(j.item())
            alg_out['supps'].append(suppsize.item())
            alg_out['Psis'].append(upperb.item())
            
            if self.opts.plot_every > 0 and iter % self.opts.plot_every == 0:
                self.plot_iteration(uk, y_ref, Ku, yk, alpha)
                
        if self.opts.plot_final:
            self.plot_iteration(uk, y_ref, Ku, yk, alpha)
            
        return uk, alg_out
    
    def plot_iteration(self, uk: NeuralMeasure, y_ref: torch.Tensor, 
                      Ku: torch.Tensor, yk: torch.Tensor, alpha: float):
        """Plot current iteration using Problem2D plotting methods"""
        plt.figure(1)
        self.problem.plot_forward(uk, y_ref)
        plt.figure(2)
        self.problem.plot_adjoint(uk, yk, alpha)
        plt.draw()
        plt.pause(0.01)