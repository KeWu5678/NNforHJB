"""Coordinate-descent warm start for newly inserted atoms.

One 1-D proximal step (MATLAB PDAPmultisemidiscrete.m:104-147): pick the new
candidate whose descent profile against the current residual is largest, take the
nonconvex prox step along that combined direction, and return initial outer
weights for all new atoms.

The signed and semiconcave models share this computation exactly. They differ
only in how an atom enters the value function: the signed network *adds* it
(two-sided weights, descend along -profile), while the semiconcave model
*subtracts* a convex ``g`` (nonnegative weights, ascend along +profile then clamp
at zero). That single difference is the ``nonneg`` flag.
"""

from __future__ import annotations

import logging
from typing import Callable, Tuple

import torch

from ..SSN.prox import _phi_prox

logger = logging.getLogger(__name__)


def warm_start(
    W_new: torch.Tensor,
    b_new: torch.Tensor,
    residual: Tuple[torch.Tensor, torch.Tensor],
    X: torch.Tensor,
    *,
    activation: Callable[[torch.Tensor], torch.Tensor],
    power: float,
    loss_weights: Tuple[float, float],
    alpha: float,
    th: float,
    gamma: float,
    use_sphere: bool = True,
    nonneg: bool = False,
    verbose: bool = False,
) -> torch.Tensor:
    """Return initial outer weights ``(n_new,)`` for the new atoms.

    ``residual = (res_v, res_dv)`` is ``prediction - target`` from the current
    model (the trainer owns the no-atoms case, where it is ``-target``).
    ``nonneg`` selects the semiconcave convention: nonnegative weights for an atom
    that enters the value function with a minus sign.
    """
    n_new = W_new.shape[0]
    if n_new == 0:
        return torch.zeros(0, dtype=torch.float64)

    res_v, res_dv = residual
    res_v = res_v.detach().reshape(-1)
    res_dv = res_dv.detach().reshape(-1)
    X_det = X.detach()
    N, d = X_det.shape[0], X_det.shape[1]
    Nx = N * d
    w1, w2 = loss_weights
    p = power

    with torch.no_grad():
        pre = X_det @ W_new.T + b_new
        act = activation(pre)
        S_val = act ** p
        if use_sphere:
            act_deriv = (pre > 0).double()
        else:
            pre_tmp = pre.detach().requires_grad_(True)
            with torch.enable_grad():
                act_tmp = activation(pre_tmp)
                act_deriv = torch.autograd.grad(act_tmp.sum(), pre_tmp, create_graph=False)[0].detach()
        dS_dz = act_deriv if p == 1.0 else p * act ** (p - 1) * act_deriv
        S_grad = (dS_dz.unsqueeze(2) * W_new.unsqueeze(0)).permute(0, 2, 1).reshape(-1, n_new)

        profiles = (w1 / Nx) * (S_val.T @ res_v) + (w2 / Nx) * (S_grad.T @ res_dv)
        abs_profiles = profiles.abs()
        best = int(abs_profiles.argmax().item())
        eps_sqrt = float(torch.finfo(torch.float64).eps) ** 0.5
        safe = abs_profiles.clamp_min(1e-30)
        # Largest profile gets a unit step; the rest a tiny eps step (same sign).
        base = eps_sqrt * profiles / safe
        base[best] = profiles[best] / safe[best]
        coeff = base if nonneg else -base

        Kv = S_val @ coeff
        Kg = S_grad @ coeff
        phat = float((w1 / Nx) * Kv.dot(res_v) + (w2 / Nx) * Kg.dot(res_dv))
        if nonneg:
            phat = -phat
        what = float((w1 / Nx) * Kv.dot(Kv) + (w2 / Nx) * Kg.dot(Kg))

        if phat <= -alpha and what > 1e-30:
            tau = _phi_prox(alpha / what, -phat / what, th, gamma, q=2.0 / (p + 1.0))
        else:
            tau = 0.0
        out = tau * coeff
        if nonneg:
            out = out.clamp_min(0.0)
        if verbose:
            logger.debug(
                "Warm start        initialized %d new output weights  max |weight|=%.2e",
                n_new, float(out.abs().max().item()) if out.numel() else 0.0,
            )
    return out.reshape(-1)
