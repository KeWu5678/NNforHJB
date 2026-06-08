"""Semiconcave parametric model: a composed value network (nn.Module).

Model:  V(x) = 0.5 * C * ||x||^2 - g(x),
        g(x) = sum_i c_i * sigma(w_i . x + b_i)^p + a . x + b0,
with c_i >= 0 (convex g => semiconcave V), C >= 0, and a, b0 free.  V is the sum
of three submaps (quadratic head, shallow network, affine), each linear in its
own parameters, so V is linear in theta = [c (n) | C (1) | a (d) | b0 (1)] with
the inner weights (w_i, b_i) frozen.

This is an ``nn.Module`` parametrization: ``c, C, a, b0`` are parameters (in that
order, so ``parameters()`` matches theta and the penalty masks), ``W, b`` are
buffers (the fixed support).  It owns the composed forward, the prediction, and
the feature maps (``jacobians`` = Jacobians of V, dV w.r.t. theta).
The SSN outer solve lives in the trainer (:mod:`src.PDAP.ssn_solve`): only ``c``
is penalised; ``c`` and ``C`` are nonnegative; ``a, b0`` are free.
"""

from __future__ import annotations

import logging
from typing import Callable, Tuple

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector

logger = logging.getLogger(__name__)


TensorLike = torch.Tensor | np.ndarray


class SemiconcaveModel(torch.nn.Module):
    def __init__(
        self,
        power: float = 1.0,
        activation: Callable[[torch.Tensor], torch.Tensor] | None = None,
        c_init: float = 1.0,
        verbose: bool = True,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """Store the forward-defining parameters; the support is set by ``set_atoms``.

        Objective and SSN-solver hyperparameters are the trainer's (see
        :mod:`src.PDAP.ssn_solve`); ``power`` defines the atom ``sigma^p`` and the
        penalty exponent ``q = 2/(power+1)``.
        """
        if power < 1.0:
            raise ValueError("semiconcave model requires a convex atom: power >= 1")
        super().__init__()
        self.power = float(power)
        self.q = 2.0 / (self.power + 1.0)
        self.activation = activation if activation is not None else torch.relu
        self.verbose = verbose
        self.dtype = dtype

        # Trainable parameters in theta-order [c | C | a | b0].  c and a are
        # resized once the support / input dimension are known (set_atoms /
        # _ensure_affine); C and b0 are scalars.
        self.c = torch.nn.Parameter(torch.zeros(0, dtype=dtype))
        self.C = torch.nn.Parameter(torch.tensor(float(c_init), dtype=dtype))
        self.affine_w = torch.nn.Parameter(torch.zeros(0, dtype=dtype))
        self.affine_b = torch.nn.Parameter(torch.zeros((), dtype=dtype))
        # Fixed support (frozen inner weights), not optimized -> buffers.
        self.register_buffer("W", torch.zeros(0, 0, dtype=dtype))
        self.register_buffer("b", torch.zeros(0, dtype=dtype))

        self.input_dim: int | None = None

    # ------------------------------------------------------------------ #
    # State management
    # ------------------------------------------------------------------ #
    def _ensure_affine(self, d: int) -> None:
        if self.affine_w.numel() != d:
            self.affine_w = torch.nn.Parameter(torch.zeros(d, dtype=self.dtype))
        self.input_dim = d

    def set_atoms(self, W: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> None:
        W = torch.as_tensor(W, dtype=self.dtype)
        b = torch.as_tensor(b, dtype=self.dtype).reshape(-1)
        c = torch.as_tensor(c, dtype=self.dtype).reshape(-1)
        self.W = W
        self.b = b
        self.c = torch.nn.Parameter(c)
        self._ensure_affine(W.shape[1])

    def get_atoms(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read the current support as (W (n,d), b (n,), c (n,)); empty if none."""
        if self.n_neurons == 0:
            d = self.input_dim or 0
            z = torch.zeros(0, dtype=self.dtype)
            return torch.zeros(0, d, dtype=self.dtype), z, z.clone()
        return self.W.detach().clone(), self.b.detach().clone(), self.c.detach().clone()

    @property
    def n_neurons(self) -> int:
        return int(self.W.shape[0])

    # ------------------------------------------------------------------ #
    # Composed value network and prediction.
    # V(x) = 0.5 C||x||^2 - (sum_i c_i sigma(w_i·x+b_i)^p + a·x + b0), the sum of
    # three submaps (quadratic head, shallow network, affine), linear in
    # theta = [c | C | a | b0].  _value is the single forward, parameterized by a
    # flat theta so the same map serves prediction and the SSN feature maps.
    # ------------------------------------------------------------------ #
    def _theta(self) -> torch.Tensor:
        """Current parameters packed as the flat vector [c | C | a | b0]."""
        return parameters_to_vector([self.c, self.C, self.affine_w, self.affine_b]).detach()

    def _value(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Value V(x) (N, 1) for a parameter vector theta = [c | C | a | b0]."""
        n, d = self.n_neurons, int(self.input_dim)
        C = theta[n]
        a = theta[n + 1:n + 1 + d]
        b0 = theta[n + 1 + d]
        quad = 0.5 * C * (x * x).sum(dim=1, keepdim=True)
        g = (x @ a + b0).reshape(-1, 1)
        if n > 0:
            S = self.activation(x @ self.W.T + self.b) ** self.power   # (N, n)
            g = g + (S @ theta[:n]).reshape(-1, 1)
        return quad - g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._value(self._theta(), x)

    def predict_tensors(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_req = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            V = self(x_req)
            dV = torch.autograd.grad(V.sum(), x_req, create_graph=False)[0]
        return V.detach(), dV.detach()

    def predict(self, x: TensorLike) -> Tuple[np.ndarray, np.ndarray]:
        xt = torch.as_tensor(x, dtype=self.dtype)
        V, dV = self.predict_tensors(xt)
        return V.cpu().numpy(), dV.cpu().numpy()

    # ------------------------------------------------------------------ #
    # Linear-in-theta interface for the trainer SSN solve (src.PDAP.ssn_solve).
    # The feature maps are the Jacobians of (V, dV) w.r.t. theta. They are built
    # directly from the chain rule; full higher-order functional Jacobians
    # materialize a large tape on pendulum-sized batches.
    # ------------------------------------------------------------------ #
    def jacobians(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Feature maps (Phi_v (N,P), Phi_g (N*d,P)) of V, dV w.r.t. theta."""
        self._ensure_affine(x.shape[1])
        x_det = x.detach()
        N, d = x_det.shape
        n = self.n_neurons

        if n > 0:
            z = x_det @ self.W.T + self.b
            z = z.detach().requires_grad_(True)
            with torch.enable_grad():
                S = self.activation(z) ** self.power
                dS_dz = torch.autograd.grad(
                    S, z, grad_outputs=torch.ones_like(S), create_graph=False,
                )[0]
            S = S.detach()
            dS_dz = dS_dz.detach()
            dS_dx = dS_dz.unsqueeze(2) * self.W.detach().unsqueeze(0)  # (N, n, d)
            Phi_g_c = -dS_dx.permute(0, 2, 1).reshape(N * d, n)
        else:
            S = x_det.new_zeros(N, 0)
            Phi_g_c = x_det.new_zeros(N * d, 0)

        Phi_v = torch.cat(
            [
                -S,
                0.5 * (x_det * x_det).sum(dim=1, keepdim=True),
                -x_det,
                -x_det.new_ones(N, 1),
            ],
            dim=1,
        )

        eye = torch.eye(d, dtype=x_det.dtype, device=x_det.device)
        Phi_g = torch.cat(
            [
                Phi_g_c,
                x_det.reshape(N * d, 1),
                -eye.expand(N, d, d).reshape(N * d, d),
                x_det.new_zeros(N * d, 1),
            ],
            dim=1,
        )
        return Phi_v, Phi_g

    def penalty_masks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bool masks over theta: penalized (c block) and nonnegative (c, C)."""
        n, d = self.n_neurons, int(self.input_dim)
        P = n + d + 2
        penalized = torch.zeros(P, dtype=torch.bool)
        nonneg = torch.zeros(P, dtype=torch.bool)
        if n > 0:
            penalized[:n] = True
            nonneg[:n] = True
        nonneg[n] = True  # C >= 0, unpenalised
        return penalized, nonneg
