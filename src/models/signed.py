#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:30:52 2024

@author: chaoruiz
"""

from typing import Optional, Tuple
import logging
import torch
from ..SSN.penalty import _phi
from ..eval import data_loss_terms
from .net import ShallowNetwork

logger = logging.getLogger(__name__)


class SignedModel:
    """
    shallow neural networks
    """
    def __init__(
        self,
        alpha: float,
        gamma: float,
        activation: torch.nn.Module = torch.relu,
        power: float = 2.1,
        lr: float = 1.0,
        loss_weights: Tuple[float, float] = (1.0, 1.0),
        th: float = 0.5,
        verbose: bool = True,
        method: Optional[str] = None,
        max_ls_iter: int = 500,
        tolerance_ls: float = 1.0 + 1e-8,
        tolerance_grad: float = 0.0,
        sigmamax: float = 10.0,
        ) -> None:
        """
        Args:
            activation: Callable[[], Tensor]
            power: Power for activation function (default: 1.0)
            loss_weights: Weights for (value_loss, gradient_loss) (default: (1.0, 1.0))
            th: Interpolation parameter between L1 (th=0) and non-convex (th=1) (default: 0.5)
            verbose: Whether to print training progress to terminal (default: True)
        """
        # optimizer parameters
        self.activation = activation
        self.power = power
        self.q = 2.0 / (self.power + 1.0)
        self.alpha = alpha
        self.gamma = gamma
        self.lr = lr
        # loss parameters
        self.loss_weights = loss_weights
        self.th = th
        # verbose
        self.verbose = verbose
        # SSN solver settings (default to today's literals)
        self.method = method
        self.max_ls_iter = max_ls_iter
        self.tolerance_ls = tolerance_ls
        self.tolerance_grad = tolerance_grad
        self.sigmamax = sigmamax
        self.input_dim: Optional[int] = None
        
        # Network is (re)built by set_atoms; the SSN solve lives in the trainer.
        self.net = None
        self.last_fit_summary = {}

        # High-level setup is logged by PDAP after the data has been prepared.
    
    def _create_network(self, inner_weights: torch.Tensor, inner_bias: torch.Tensor, outer_weights: torch.Tensor) -> None:
        """Build the shallow network on a fixed support (frozen inner weights).

        Args:
            inner_weights: hidden weights (frozen — the support is fixed by PDAP)
            inner_bias: hidden bias (frozen)
            outer_weights: output weights (the only trainable parameters)
        """
        input_dim = int(inner_weights.shape[1])
        n = inner_weights.shape[0]
        if self.verbose:
            logger.debug("Network support  atoms=%d", n)

        self.net = ShallowNetwork(
            [input_dim, n, 1],
            self.activation,
            p=self.power,
            inner_weights=inner_weights, inner_bias=inner_bias, outer_weights=outer_weights
        )

        # Freeze hidden layer so only the output weights are trainable.
        self.net.hidden.weight.requires_grad = False
        self.net.hidden.bias.requires_grad = False

    def compute_loss(self, x_input: torch.Tensor, target_v: torch.Tensor, target_dv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the combined MSE loss for value and gradient matching. The loss always use
        the full batch
        Args:
            x_input: Input coordinates
            target_v: Target values
            target_dv: Target gradients
        Returns:
            Tuple of (total_loss, value_loss, grad_loss)
        """
        # Create a fresh tensor that requires gradients for gradient computation
        # Remark: for SSN, the graph actually doesn't need to be created here.
        x = x_input.clone().detach().requires_grad_(True)
        pred_v = self.net(x)
        pred_dv = torch.autograd.grad(
            outputs=pred_v, 
            inputs=x, 
            grad_outputs=torch.ones_like(pred_v), 
            create_graph=True, 
            retain_graph=True
        )[0]
        
        # Data-fidelity term (Nx = N*d normalization) lives in src.eval.
        data_loss, value_loss, grad_loss = data_loss_terms(
            pred_v, pred_dv, target_v, target_dv, self.loss_weights
        )

        # Full objective: data loss + regularization
        # Matches MATLAB SSN.m line 34: obj = @(u) F.F(Sred*u - ref) + alpha*sum(phi.phi(computeNorm(u, NQ)))
        abs_u = torch.abs(self.net.output.weight)
        reg_arg = abs_u ** self.q if self.q != 1.0 else abs_u
        total_loss = data_loss + self.alpha * torch.sum(_phi(reg_arg, self.th, self.gamma))
 
        return total_loss, value_loss, grad_loss

    def predict_tensors(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (V, dV) at x as detached tensors, V:(N,1), dV:(N,d).

        Uniform with SemiconcaveModel.predict_tensors so the PDAP loop and the
        insertion strategy can read residuals from any model the same way.
        """
        if self.net is None:
            raise RuntimeError("network not created yet; call set_atoms() first")
        x_req = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            V = self.net(x_req)
            dV = torch.autograd.grad(V.sum(), x_req, create_graph=False)[0]
        return V.detach(), dV.detach()

    def predict(self, x):
        """Value/gradient as numpy arrays (uniform with SemiconcaveModel.predict)."""
        xt = torch.as_tensor(x, dtype=torch.float64)
        V, dV = self.predict_tensors(xt)
        return V.cpu().numpy(), dV.cpu().numpy()

    # ------------------------------------------------------------------ #
    # Uniform atom interface (matches SemiconcaveModel) for the PDAP loop.
    # Canonical representation: W (n,d), b (n,), c (n,).  The signed network
    # stores c as the (1,n) output weight; set/get reshape across that.
    # ------------------------------------------------------------------ #
    def set_atoms(self, W: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> None:
        """Establish the current support by (re)building the network."""
        W = torch.as_tensor(W, dtype=torch.float64)
        b = torch.as_tensor(b, dtype=torch.float64).reshape(-1)
        c = torch.as_tensor(c, dtype=torch.float64).reshape(1, -1)
        self._create_network(W, b, c)

    def get_atoms(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read the current support as (W (n,d), b (n,), c (n,))."""
        if self.net is None:
            raise RuntimeError("network not created yet; call set_atoms() first")
        W = self.net.hidden.weight.detach().clone()
        b = self.net.hidden.bias.detach().clone()
        c = self.net.output.weight.detach().reshape(-1).clone()
        return W, b, c

    @property
    def n_neurons(self) -> int:
        return 0 if self.net is None else int(self.net.hidden.weight.shape[0])

    # ------------------------------------------------------------------ #
    # Linear-in-theta interface for the trainer SSN solve (src.PDAP.ssn_solve).
    # theta is the output weight; the frozen network's value matrix and gradient
    # kernel are the feature maps. All coords are penalized, none nonnegative.
    # ------------------------------------------------------------------ #
    def jacobians(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Feature maps (Phi_v (N,n), Phi_g (N*d,n)) of V, dV w.r.t. theta."""
        if self.net is None:
            raise RuntimeError("network not created yet; call set_atoms() first")
        x_det = x.detach()
        return self.net.forward_network_matrix(x_det), self.net.forward_gradient_kernel(x_det)

    def get_theta(self) -> torch.Tensor:
        return self.net.output.weight.detach().reshape(-1).clone()

    def set_theta(self, theta: torch.Tensor) -> None:
        with torch.no_grad():
            self.net.output.weight.copy_(theta.reshape(1, -1))

    def penalty_masks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        n = self.n_neurons
        return torch.ones(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)
