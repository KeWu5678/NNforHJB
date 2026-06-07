"""Trainer-side SSN outer-weight solve, shared by every model.

A model is linear in its outer parameters ``theta`` for this solve. ``theta`` is
just the model's trainable ``nn.Module`` parameters, read and written with torch's
``parameters_to_vector`` / ``vector_to_parameters``; the model only has to supply
the feature maps (``jacobians`` -> ``(Phi_v, Phi_g)``) and the penalized /
nonnegative coordinate masks (``penalty_masks``). The data Hessian is the
Gauss-Newton form ``(1/Nx)(w1 Phi_v'Phi_v + w2 Phi_g'Phi_g)``; the closure is the
data loss on ``Phi @ theta`` plus the nonconvex penalty on the penalized block.
:class:`src.SSN.SSN` owns the semismooth-Newton step.

SSN hyperparameters (alpha, gamma, th, power, lr, method, line-search/trust-region
tolerances) are read from the model, where the config places them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from ..SSN import SSN
from ..SSN.penalty import _phi

if TYPE_CHECKING:
    from ..models.base import PDAPModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Objective:
    """What is minimized: data fidelity (loss_weights) + the nonconvex penalty.

    The penalty exponent q is *not* here -- it is q = 2/(power+1), derived from
    the model's activation power (the prox closed-forms depend on it), so it lives
    on the model.
    """

    alpha: float = 1e-5
    gamma: float = 0.0
    th: float = 0.5
    loss_weights: Tuple[float, float] = (1.0, 1.0)


@dataclass(frozen=True)
class SolverConfig:
    """How the SSN outer solve is run (globalization + line-search tolerances)."""

    lr: float = 1.0
    method: str = "levenberg_marquardt"
    max_ls_iter: int = 500
    tolerance_ls: float = 1.0 + 1e-8
    tolerance_grad: float = 0.0
    sigmamax: float = 10.0


def nonconvex_penalty(
    theta: torch.Tensor, penalized: torch.Tensor, nonneg: torch.Tensor,
    *, alpha: float, th: float, gamma: float, q: float,
) -> torch.Tensor:
    """The regularizer alpha * sum_i phi(arg_i) over the penalized coordinates.

    arg = base^q with base = |theta| on free-sign coords and clamp(theta, 0) on
    nonnegative ones.  This is the trainer's objective term -- shared by the SSN
    closure and the loss recording so they cannot drift apart.
    """
    pen = theta[penalized]
    if pen.numel() == 0:
        return theta.new_zeros(())
    base = torch.where(nonneg[penalized], pen.clamp_min(0.0), pen.abs())
    arg = base if q == 1.0 else base.clamp_min(1e-30) ** q
    return alpha * torch.sum(_phi(arg, th, gamma))


def ssn_solve(
    model: "PDAPModel", data_train, objective: Objective, solver: SolverConfig,
    *, iterations: int, verbose: bool = False,
) -> float:
    """Solve for the model's outer weights in place; return the final train loss."""
    X, V, dV = data_train
    Phi_v, Phi_g = model.jacobians(X)
    Phi_v = Phi_v.detach()
    Phi_g = Phi_g.detach()
    Vt = V.reshape(-1).detach()
    dVt = dV.reshape(-1).detach()
    N, d = X.shape[0], X.shape[1]
    Nx = N * d
    w1, w2 = objective.loss_weights
    alpha, gamma, th, q = objective.alpha, objective.gamma, objective.th, model.q

    H = (w1 / Nx) * (Phi_v.T @ Phi_v) + (w2 / Nx) * (Phi_g.T @ Phi_g)

    # theta is the model's trainable parameters flattened (output weights for the
    # signed net; [c | C | a | b0] for the semiconcave model).  SSN solves the
    # linear-in-theta subproblem on a standalone copy, then writes it back.
    params = [p for p in model.parameters() if p.requires_grad]
    theta = torch.nn.Parameter(parameters_to_vector(params).detach().clone())
    penalized, nonneg = model.penalty_masks()

    optimizer = SSN(
        [theta], alpha=alpha, gamma=gamma,
        penalized_mask=penalized, nonneg_mask=nonneg,
        th=th, lr=solver.lr, power=model.power, method=solver.method,
        max_ls_iter=solver.max_ls_iter, tolerance_ls=solver.tolerance_ls,
        tolerance_grad=solver.tolerance_grad, sigmamax=solver.sigmamax,
    )
    optimizer.data_hessian = H

    def closure():
        optimizer.zero_grad()
        rv = Phi_v @ theta - Vt
        rg = Phi_g @ theta - dVt
        data = (w1 / (2 * Nx)) * (rv @ rv) + (w2 / (2 * Nx)) * (rg @ rg)
        penalty = nonconvex_penalty(theta, penalized, nonneg, alpha=alpha, th=th, gamma=gamma, q=q)
        return data + penalty

    prev = float(closure().detach())
    for _ in range(iterations):
        prev = float(optimizer.step(closure).detach())

    vector_to_parameters(theta.detach(), params)
    if verbose:
        logger.debug("Output-weight solve complete  train_loss=%.6e", prev)
    return prev
