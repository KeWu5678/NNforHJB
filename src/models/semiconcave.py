"""Semiconcave parametric model driven by the SSN pipeline.

Model:  V(x) = 0.5 * C * ||x||^2 - g(x),
        g(x) = sum_i c_i * sigma(w_i . x + b_i)^p + a . x + b0,
with c_i >= 0 (convex g => semiconcave V), C >= 0, and a, b0 free.

The outer solve optimises theta = [c (n) | C (1) | a (d) | b0 (1)] with the inner
weights (w_i, b_i) frozen, using :class:`src.SSN.SSN` with masks: only ``c`` is
penalised; ``c`` and ``C`` are nonnegative; ``a, b0`` are free.

Both the value and the gradient predictions are *linear* in theta, so the data
loss is a quadratic whose Hessian is assembled in closed form
(``_build_data_hessian``) and handed to SSN -- matching the contract of the
pure-network ``src.model.model`` (which sets ``optimizer.data_hessian``).
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import torch
from loguru import logger

from ..SSN import SSN
from ..utils import _phi


TensorLike = torch.Tensor | np.ndarray


class SemiconcaveModel:
    def __init__(
        self,
        alpha: float,
        gamma: float,
        power: float = 1.0,
        th: float = 0.5,
        activation: Callable[[torch.Tensor], torch.Tensor] | None = None,
        loss_weights: str | Tuple[float, float] = "h1",
        lr: float = 1.0,
        c_init: float = 1.0,
        verbose: bool = True,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if power < 1.0:
            raise ValueError("semiconcave model requires a convex atom: power >= 1")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.th = float(th)
        self.power = float(power)
        self.q = 2.0 / (self.power + 1.0)
        self.activation = activation if activation is not None else torch.relu
        self.lr = float(lr)
        self.verbose = verbose
        self.dtype = dtype
        self.optimizer_type = "SSN_semiconcave"

        if isinstance(loss_weights, str):
            mapping = {"l2": (1.0, 0.0), "h1": (1.0, 1.0)}
            loss_weights = mapping[loss_weights.lower()]
        self.loss_weights = (float(loss_weights[0]), float(loss_weights[1]))

        # Atom state (frozen inner weights + nonneg outer weights).
        self.W: torch.Tensor | None = None   # (n, d)
        self.b: torch.Tensor | None = None   # (n,)
        self.c: torch.Tensor | None = None   # (n,)
        # Unpenalised structural parameters.
        self.C: float = float(c_init)
        self.affine_w: torch.Tensor | None = None  # (d,)
        self.affine_b: float = 0.0
        self.input_dim: int | None = None

    def _prepare_data(self, data: dict):
        """Split a data dict into (train, valid) tensor tuples (x, V, dV).

        Mirrors src.model.model._prepare_data: all-but-one point for training,
        the last for validation (the external eval set is held out separately).
        """
        ob_x, ob_v, ob_dv = data["x"], data["v"], data["dv"]
        x = torch.as_tensor(ob_x, dtype=self.dtype)
        v = torch.as_tensor(ob_v, dtype=self.dtype).reshape(-1, 1)
        dv = torch.as_tensor(ob_dv, dtype=self.dtype)
        self.input_dim = int(x.shape[1])
        n = x.shape[0]
        split = max(1, n - 1)
        train = (x[:split], v[:split], dv[:split])
        valid = (x[split:], v[split:], dv[split:])
        return train, valid

    # ------------------------------------------------------------------ #
    # State management
    # ------------------------------------------------------------------ #
    def _ensure_affine(self, d: int) -> None:
        if self.affine_w is None:
            self.affine_w = torch.zeros(d, dtype=self.dtype)
        self.input_dim = d

    def set_atoms(self, W: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> None:
        W = torch.as_tensor(W, dtype=self.dtype)
        b = torch.as_tensor(b, dtype=self.dtype).reshape(-1)
        c = torch.as_tensor(c, dtype=self.dtype).reshape(-1)
        self.W, self.b, self.c = W, b, c
        self._ensure_affine(W.shape[1])

    @property
    def n_neurons(self) -> int:
        return 0 if self.W is None else int(self.W.shape[0])

    # ------------------------------------------------------------------ #
    # Feature maps (linear in theta = [c | C | a | b0])
    # ------------------------------------------------------------------ #
    def _atom_value(self, x: torch.Tensor) -> torch.Tensor:
        """sigma(x w + b)^p, shape (N, n)."""
        pre = x @ self.W.T + self.b
        return self.activation(pre) ** self.power

    def _atom_grad_kernel(self, x: torch.Tensor) -> torch.Tensor:
        """d/dx [sigma^p], stacked to (N*d, n) with row index k*d + l."""
        z = (x @ self.W.T + self.b).detach().requires_grad_(True)
        with torch.enable_grad():
            S = self.activation(z) ** self.power
            dS_dz = torch.autograd.grad(S, z, grad_outputs=torch.ones_like(S))[0]
        dS_dz = dS_dz.detach()                       # (N, n)
        dS_dx = dS_dz.unsqueeze(2) * self.W.unsqueeze(0)  # (N, n, d)
        return dS_dx.permute(0, 2, 1).reshape(-1, self.W.shape[0])  # (N*d, n)

    def _build_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Return (Phi_v (N,P), Phi_g (N*d,P), n) for theta=[c|C|a|b0]."""
        N, d = x.shape
        n = self.n_neurons
        # value features
        cols_v = []
        if n > 0:
            cols_v.append(-self._atom_value(x))                       # c: (N,n)
        cols_v.append(0.5 * (x * x).sum(dim=1, keepdim=True))         # C: (N,1)
        cols_v.append(-x)                                             # a: (N,d)
        cols_v.append(-torch.ones(N, 1, dtype=self.dtype))           # b0:(N,1)
        Phi_v = torch.cat(cols_v, dim=1)
        # gradient features (rows: k*d + l)
        cols_g = []
        if n > 0:
            cols_g.append(-self._atom_grad_kernel(x))                 # c: (N*d,n)
        cols_g.append(x.reshape(-1, 1))                               # C: (N*d,1)
        # a block: -I_d stacked N times  => (N*d, d)
        neg_I = -torch.eye(d, dtype=self.dtype).repeat(N, 1)          # (N*d, d)
        cols_g.append(neg_I)
        cols_g.append(torch.zeros(N * d, 1, dtype=self.dtype))        # b0:(N*d,1)
        Phi_g = torch.cat(cols_g, dim=1)
        return Phi_v, Phi_g, n

    def _theta_vector(self, n: int, d: int) -> torch.Tensor:
        parts = []
        if n > 0:
            parts.append(self.c.reshape(-1))
        parts.append(torch.tensor([self.C], dtype=self.dtype))
        parts.append(self.affine_w.reshape(-1))
        parts.append(torch.tensor([self.affine_b], dtype=self.dtype))
        return torch.cat(parts)

    def _unpack_theta(self, theta: torch.Tensor, n: int, d: int) -> None:
        idx = 0
        if n > 0:
            self.c = theta[:n].detach().clone()
            idx = n
        self.C = float(theta[idx].item()); idx += 1
        self.affine_w = theta[idx:idx + d].detach().clone(); idx += d
        self.affine_b = float(theta[idx].item())

    def _masks(self, n: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        P = n + d + 2
        penalized = torch.zeros(P, dtype=torch.bool)
        nonneg = torch.zeros(P, dtype=torch.bool)
        if n > 0:
            penalized[:n] = True
            nonneg[:n] = True
        nonneg[n] = True  # C >= 0, unpenalised
        return penalized, nonneg

    # ------------------------------------------------------------------ #
    # Prediction / losses (analytic, for eval & logging)
    # ------------------------------------------------------------------ #
    def predict_tensors(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.enable_grad():
            x_req = x.detach().clone().requires_grad_(True)
            quad = 0.5 * self.C * (x_req * x_req).sum(dim=1, keepdim=True)
            g = x_req @ self.affine_w + self.affine_b
            if self.n_neurons > 0:
                g = g.reshape(-1, 1) + (self._atom_value(x_req) @ self.c).reshape(-1, 1)
            else:
                g = g.reshape(-1, 1)
            V = quad - g
            dV = torch.autograd.grad(V.sum(), x_req, create_graph=False)[0]
        return V.detach(), dV.detach()

    def predict(self, x: TensorLike) -> Tuple[np.ndarray, np.ndarray]:
        xt = torch.as_tensor(x, dtype=self.dtype)
        V, dV = self.predict_tensors(xt)
        return V.cpu().numpy(), dV.cpu().numpy()

    def _penalty(self) -> torch.Tensor:
        if self.n_neurons == 0:
            return torch.zeros((), dtype=self.dtype)
        c = self.c.clamp_min(0.0)
        arg = c if self.q == 1.0 else c.clamp_min(1e-30) ** self.q
        return self.alpha * torch.sum(_phi(arg, self.th, self.gamma))

    def _compute_loss(self, x, V, dV):
        Vp, dVp = self.predict_tensors(x)
        N, d = x.shape
        Nx = N * d
        w1, w2 = self.loss_weights
        value_loss = torch.sum((Vp - V) ** 2) / (2 * Nx)
        grad_loss = torch.sum((dVp - dV) ** 2) / (2 * Nx)
        data = w1 * value_loss + w2 * grad_loss
        return data + self._penalty(), value_loss, grad_loss

    @torch.no_grad()
    def _compute_relative_errors(self, x, V, dV):
        Vp, dVp = self.predict_tensors(x)
        v_diff = torch.sum((Vp - V) ** 2)
        dv_diff = torch.sum((dVp - dV) ** 2)
        v_ref = torch.sum(V ** 2).clamp_min(1e-30)
        dv_ref = torch.sum(dV ** 2).clamp_min(1e-30)
        err_l2 = torch.sqrt(v_diff / v_ref)
        err_grad = torch.sqrt(dv_diff / dv_ref)
        err_h1 = torch.sqrt((v_diff + dv_diff) / (v_ref + dv_ref))
        return float(err_l2), float(err_grad), float(err_h1)

    # ------------------------------------------------------------------ #
    # SSN outer solve
    # ------------------------------------------------------------------ #
    def train_ssn(self, x: torch.Tensor, V: torch.Tensor, dV: torch.Tensor,
                  iterations: int = 20) -> bool:
        N, d = x.shape
        self._ensure_affine(d)
        n = self.n_neurons

        Phi_v, Phi_g, _ = self._build_features(x)
        Phi_v = Phi_v.detach()
        Phi_g = Phi_g.detach()
        Vt = V.reshape(-1).detach()
        dVt = dV.reshape(-1).detach()
        Nx = N * d
        w1, w2 = self.loss_weights

        H = (w1 / Nx) * (Phi_v.T @ Phi_v) + (w2 / Nx) * (Phi_g.T @ Phi_g)

        theta = torch.nn.Parameter(self._theta_vector(n, d))
        penalized, nonneg = self._masks(n, d)

        optimizer = SSN(
            [theta], alpha=self.alpha, gamma=self.gamma,
            penalized_mask=penalized, nonneg_mask=nonneg,
            th=self.th, lr=self.lr, power=self.power,
        )
        optimizer.data_hessian = H

        def closure():
            optimizer.zero_grad()
            rv = Phi_v @ theta - Vt
            rg = Phi_g @ theta - dVt
            data = (w1 / (2 * Nx)) * (rv @ rv) + (w2 / (2 * Nx)) * (rg @ rg)
            # penalty on the c block (theta[:n] >= 0), matching alpha_vec/_penalty_grad
            if n > 0:
                cblk = theta[:n].clamp_min(0.0)
                arg = cblk if self.q == 1.0 else cblk.clamp_min(1e-30) ** self.q
                penalty = self.alpha * torch.sum(_phi(arg, self.th, self.gamma))
            else:
                penalty = torch.zeros((), dtype=self.dtype)
            return data + penalty

        made_progress = False
        prev = float(closure().detach())
        for _ in range(iterations):
            loss = float(optimizer.step(closure).detach())
            if loss < prev - 1e-15:
                made_progress = True
            prev = loss

        self._unpack_theta(theta.detach(), n, d)
        if n > 0:
            self.c = self.c.clamp_min(0.0)
        self.C = max(self.C, 0.0)
        if self.verbose:
            logger.info(f"semiconcave SSN: C={self.C:.4e}, |c| sum={float(self.c.abs().sum()) if n>0 else 0:.4e}")
        return made_progress
