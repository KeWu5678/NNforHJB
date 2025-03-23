#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 14:27:54 2024

@author: chaoruiz
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Dict




@dataclass
class NeuralMeasure:
    x: torch.Tensor     #The inner weights
    u: torch.Tensor     #The outer weights


class NN2D:
    def __init__(self, epsilon = 2.5, delta = 0.01, force_upper=False):
        """ 
        Initialise the parameters
        Input:
            delta:          float
                            The smoothing parameter taking effect at the corner
            force_upper:    Boolean
                            When ture, taking stereoprojection in upper hemisph
            epsilon:        float(>= 2)
                            Power of the ReLU
        ---------------------------------------------------------------------
        Attribute: 
            N:              integer
                            Dimension of the output
            dim:            integer
                            Dimension of the input
            L:              float
                            Length of the sample rectangle
            Nonbs           integer
                            Number of samples
        ---------------------------------------------------------------------                    
         Induced Parameter:
            Omega(dim, L):  [2] tensor
                            Expressive range of the weights???             
            Nsteps(Nobs):   integer
                            Sampling grid along an axis
            xhat(Nobs, L):  [dim, Nobs1] tensor
                            Sampling grid
        """
        # scalar problem
        self.N = 1
        self.dim = 2  
        self.L = 3.0  
        self.epsilon = epsilon
        self.Nobs = 21 ** 2
        self.delta = delta
        self.force_upper = force_upper
        
        "Define an expressive ring on the projective plane"
        self.R = torch.sqrt(self.dim) * self.L
        RO = self.R + np.sqrt(1 + self.R**2)
        rO = -self.R + np.sqrt(1 + self.R**2)
        self.Omega = torch.tensor([rO, RO])
        
        "Construct the grid for sampling and stack it"
        Nsteps = int(np.floor(self.Nobs**(1.0/self.dim)))
        # To be updated with sample
        x1 = torch.linspace(-self.L, self.L, Nsteps)
        x2 = torch.linspace(-self.L, self.L, Nsteps)
        
        # Stack the matrix to facilitate parallel computation
        X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
        self.xhat = torch.stack([X1.flatten(), X2.flatten()])
        

    def kernel(self, xhat: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The function gives the output of the unweighted neural network and 
        its 1st and 2nd derivative, given the sample xhat and nodes x.
        ---------------------------------------------------------------------
        Input:
            xhat:   shape: [2, Nsteps]
                    The location of samples 
            x:      shape: [2, Nnodes]
                    The stereographic coord of the nodes
        ---------------------------------------------------------------------
        Output:
            k:      Nobs x Nnodes matrix 
                    Output of the network with row as output of a sample, column
                    as output of a neurone
            dk:     Nobs x Nnodes matrix
                    Derivative w.r.t. the inner weights
            ddk:    Nobs x Nnodes matrix
                    Second derivative w.r.t. the inner weights
        """
        Nnodes = x.size(1)
        Nsteps = xhat.size(1)
        epsilon = self.epsilon

        X = torch.zeros(Nsteps, Nnodes, self.dim)
        Xhat = torch.zeros(Nsteps, Nnodes, self.dim)
        
        # Boardcasting.  
        # X: Coordinate matrix of nodes (right duplication)
        # Xhat: Coordinate matrix of samples (down duplication)
        # Reture: 2 Nobs x Nnodes (in each dimemsion)
        for j in range(self.dim):
            X[:,:,j], Xhat[:,:,j] = torch.meshgrid(x[j,:], xhat[j,:], indexing='ij')
        
        # return a Nobs x Nnodes array(the l2 norm of the weights)
        x2 = torch.sum(X**2, dim=2)
        
        # Return Nobs x Nnodes matrix: 
        # with first row the image vector of the first, first column the image of fist neurone.
        xxhat = torch.sum(X * Xhat, dim=2)
        # Compute y = (2*xxhat + 1 - x2) / (1 + x2)
        y = (2 * xxhat + 1 - x2) / (1 + x2)
        absy = torch.sqrt(y ** 2)
        if not self.force_upper:
            k = (0.5 * (absy + y)) ** epsilon
            # dydk: return a Nobs x Nnodes matrix
            dydx = 2 * (Xhat - X - X * y) / (1 + x2)
            # dk: return a Nobs x Nnodes matrix
            dk = epsilon * (y ** (epsilon - 1)) * dydx
            ddk = epsilon * (epsilon - 1) * (y ** (epsilon -2)) * dydx
    
        return k, dk, ddk
    
                            
    def K(self, xhat: torch.Tensor, u: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This function multiplies the output of actfunc with outweights
        ----------------------------------------------------------------------
        INPUT:
        xhat:   2 x Nobs 
                Location of the samples
        u :     dict
                Dictionary of inner and outer weights
        u['x']: 2 x Nnodes
                Inner weights
        u['u']: 1 x Nnodes
                Outer weights
        ----------------------------------------------------------------------
        RETURN:
        Ku : TYPE
            DESCRIPTION.
        dKu : TYPE
            DESCRIPTION.

        """
        k, dk, ddk = self.kernel(xhat, u['x'])
        Ku = k @ u['u'].T
        dKu = dk @ u['u'].T
        return Ku, dKu
    

    def Ks(self, x: torch.Tensor, xhat: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ 
        This function gives 
        ----------------------------------------------------------------------
        INPUT:
            x, xhat:    as in the kernal function
            y:          shape of [Nobs]
                        Used to calculate the gradient p(w) in discrete setting
        ----------------------------------------------------------------------
        OUTPUT:
            
            
        """
        k, dk, ddk = self.kernel(xhat, x)
        Ksy = k.T @ y
        dKsy = dk.T @ y
        ddKsy = ddk.T @ y
        return Ksy, dKsy, ddKsy


""" Define the class of penalty"""
class Phi:
    def __init__(self, gamma: float):
        self.gamma = gamma
        th = 1/2
        gam = gamma/(1-th) if gamma != 0 else 0

        # Define functions using lambda
        self.phi = (lambda t: t) if gamma == 0 else \
                  (lambda t: th * t + (1-th) * torch.log(1 + gam * t) / gam)
        
        self.dphi = (lambda t: torch.ones_like(t)) if gamma == 0 else \
                   (lambda t: th + (1-th) / (1 + gam * t))
        
        self.ddphi = (lambda t: torch.zeros_like(t)) if gamma == 0 else \
                    (lambda t: -(1-th) * gam / (1 + gam * t)**2)
        
        self.prox = (lambda sigma, g: torch.maximum(g - sigma, torch.zeros_like(g))) if gamma == 0 else \
                   (lambda sigma, g: 0.5 * torch.maximum(
                       (g - sigma*th - 1/gam) + torch.sqrt((g - sigma*th - 1/gam)**2 + 4*(g - sigma)/gam), 
                       torch.zeros_like(g)))

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return self.phi(t)
    


class Optimizer2D:
    def __init__(self, NN: NN2D, pen: Phi, gamma: float = 0):
        """
        Initialize the optimization problem
        
        Args:
            model: NeuralNetwork2D instance
            gamma: penalty parameter
        """
        self.NN = NN
        self.pen = pen

    def setup_objective(self):
        """Setup tracking objective function and derivatives"""
        # Take the number of samples
        Nx = self.model.xhat.size(1)
        self.F = lambda y: 0.5/Nx * torch.norm(y)**2
        self.dF = lambda y: 1/Nx * y
        self.ddF = lambda y: 1/Nx * torch.ones_like(y)



    def optimize_u(self, y_d: torch.Tensor, alpha: float, u: NeuralMeasure) -> NeuralMeasure:
        """
        Optimize coefficients for fixed locations(inner weights) using quasi-Newton descent
        
        Input:
            y_d: target values
            u: initial guess dictionary
            
        Returns:
        """
        k, _, _= self.NN.kernel(self.NN.xhat, u.x)
        # Tell Pytorch to track gradient of the tensor
        u_opt = u.u.clone().requires_grad_(True)
        
        optimizer = optim.LBFGS([u_opt])

        def closure():
            optimizer.zero_grad()
            y = k @ u_opt.T
            loss = 0.5 * (y - y_d).pow(2).mean() + alpha * self.pen.phi(u_opt.abs()).sum()
            loss.backward()
            return loss
        
        return NeuralMeasure(x=u.x.clone(), u=u_opt.detach())
        


    def find_max(self, y: torch.Tensor, x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Find maxima of gradient w.r.t. the dual variable(nodes). The sample are 
        given and the outer weights play no role
        ---------------------------------------------------------------------
        Input:
            y:      shape:[Nobs]
                    Meaning see paper
            x0:     shape of [Nnodes]
                    Initial guess of the argmax
        ---------------------------------------------------------------------
        OUTPUT:
            x_max:  shape:[Nnodes]
                    Argmax of the gradient
        """
        y_norm = y / y.norm()
        
        # If no initial guess of nodes given. Generate a random vector of 50 Nodes
        if x0 is None:
            x0 = torch.randn(self.problem.dim, 50)
            x0 = x0 / x0.norm(dim=0, keepdim=True)
            
        x = x0.clone().requires_grad_(True)
        optimizer = optim.LBFGS([x], max_iter=500)
        
        def closure():
            optimizer.zero_grad()
            Ksy, _, _ = self.problem.Ks(x, self.problem.xhat, y_norm)
            loss = -0.5 * Ksy.pow(2).sum()
            loss.backward()
            return loss
        
        for _ in range(10):
            optimizer.step(closure)
            
        x_max = x.detach()
        
        if self.problem.force_upper:
            norm_x2 = x_max.pow(2).sum(0)
            mask = norm_x2 > 1
            x_max[:, mask] = -x_max[:, mask] / norm_x2[mask]
            
        return x_max
        
        
        

