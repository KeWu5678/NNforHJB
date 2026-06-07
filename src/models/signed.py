#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Signed shallow-network model: V(x) = sum_i c_i sigma(w_i . x + b_i)^p.

``SignedModel`` *is* the shallow network -- it subclasses :class:`ShallowNetwork`
(an ``nn.Module``) rather than wrapping one, so prediction is the built-in
``forward`` and the weights are ordinary module parameters. It adds only the
PDAP model contract (:class:`src.models.base.PDAPModel`): atom support,
prediction, the linear-in-theta feature maps, and the loss objective. The SSN
solve, warm start, and evaluation live in the trainer.
"""

from typing import Optional, Tuple
import logging

import torch

from .net import ShallowNetwork

logger = logging.getLogger(__name__)


class SignedModel(ShallowNetwork):
    def __init__(
        self,
        activation: torch.nn.Module = torch.relu,
        power: float = 2.1,
        verbose: bool = True,
    ) -> None:
        """Store the forward-defining parameters; layers are built by ``set_atoms``.

        The objective and SSN-solver hyperparameters are the trainer's, not the
        model's (see :mod:`src.PDAP.ssn_solve`).  ``power`` defines the activation
        ``sigma^p`` and induces the penalty exponent ``q = 2/(power+1)``.
        """
        # Bare nn.Module init -- no atoms/layers yet (built by set_atoms).
        torch.nn.Module.__init__(self)
        self.activation = activation
        self.power = power
        self.q = 2.0 / (power + 1.0)
        self.verbose = verbose
        self.input_dim: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Atom support: (re)build the shallow net on the current support.
    # Canonical representation W (n,d), b (n,), c (n,); c is stored as the
    # output layer's (1,n) weight, the only trainable parameter.
    # ------------------------------------------------------------------ #
    def set_atoms(self, W: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> None:
        """Establish the current support by rebuilding the network.

        ``n`` changes every PDAP iteration, so the layers are rebuilt by
        re-running ``ShallowNetwork``'s constructor (which owns weight and RNG
        handling -- preserving the random stream the PDAP loop is tuned against).
        """
        W = torch.as_tensor(W, dtype=torch.float64)
        b = torch.as_tensor(b, dtype=torch.float64).reshape(-1)
        c = torch.as_tensor(c, dtype=torch.float64).reshape(1, -1)
        n, d = int(W.shape[0]), int(W.shape[1])
        ShallowNetwork.__init__(
            self, [d, n, 1], self.activation, p=self.power,
            inner_weights=W, inner_bias=b, outer_weights=c,
        )
        # Freeze the hidden layer: only the output weights are trainable.
        self.hidden.weight.requires_grad_(False)
        self.hidden.bias.requires_grad_(False)
        self.input_dim = d
        if self.verbose:
            logger.debug("Network support  atoms=%d", n)

    def get_atoms(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read the current support as (W (n,d), b (n,), c (n,))."""
        if "hidden" not in self._modules:
            raise RuntimeError("no support yet; call set_atoms() first")
        return (
            self.hidden.weight.detach().clone(),
            self.hidden.bias.detach().clone(),
            self.output.weight.detach().reshape(-1).clone(),
        )

    @property
    def n_neurons(self) -> int:
        return int(self.hidden.weight.shape[0]) if "hidden" in self._modules else 0

    # ------------------------------------------------------------------ #
    # Prediction (forward is inherited from ShallowNetwork).
    # ------------------------------------------------------------------ #
    def predict_tensors(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (V (N,1), dV (N,d)) as detached tensors.

        With no support yet this is the empty (zero) network: V = 0, dV = 0 -- so
        the trainer can read a residual before the first atoms are inserted.
        """
        if "hidden" not in self._modules:
            return x.new_zeros(x.shape[0], 1), x.new_zeros(x.shape)
        x_req = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            V = self(x_req)
            dV = torch.autograd.grad(V.sum(), x_req, create_graph=False)[0]
        return V.detach(), dV.detach()

    def predict(self, x):
        """Value/gradient as numpy arrays (uniform with SemiconcaveModel.predict)."""
        xt = torch.as_tensor(x, dtype=torch.float64)
        V, dV = self.predict_tensors(xt)
        return V.cpu().numpy(), dV.cpu().numpy()

    # ------------------------------------------------------------------ #
    # Linear-in-theta interface for the trainer SSN solve.  theta is the output
    # weight (read/written via torch's parameters_to_vector built-ins); the
    # feature maps are the network's value matrix and gradient kernel. All coords
    # are penalized, none are constrained nonnegative.
    # ------------------------------------------------------------------ #
    def jacobians(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Feature maps (Phi_v (N,n), Phi_g (N*d,n)) of V, dV w.r.t. theta."""
        if "hidden" not in self._modules:
            raise RuntimeError("no support yet; call set_atoms() first")
        x_det = x.detach()
        return self.forward_network_matrix(x_det), self.forward_gradient_kernel(x_det)

    def penalty_masks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        n = self.n_neurons
        return torch.ones(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)
