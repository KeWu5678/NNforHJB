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
from typing import TYPE_CHECKING, Dict

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from ..SSN import SSN
from ..SSN.penalty import _phi

if TYPE_CHECKING:
    from ..models.base import PDAPModel

logger = logging.getLogger(__name__)


def ssn_solve(model: "PDAPModel", data_train, *, iterations: int, verbose: bool = False) -> Dict:
    """Solve for the model's outer weights in place; return a fit summary."""
    X, V, dV = data_train
    Phi_v, Phi_g = model.jacobians(X)
    Phi_v = Phi_v.detach()
    Phi_g = Phi_g.detach()
    Vt = V.reshape(-1).detach()
    dVt = dV.reshape(-1).detach()
    N, d = X.shape[0], X.shape[1]
    Nx = N * d
    w1, w2 = model.loss_weights
    alpha, gamma, th, q = model.alpha, model.gamma, model.th, model.q

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
        th=th, lr=model.lr, power=model.power, method=model.method,
        max_ls_iter=model.max_ls_iter, tolerance_ls=model.tolerance_ls,
        tolerance_grad=model.tolerance_grad, sigmamax=model.sigmamax,
    )
    optimizer.data_hessian = H

    def closure():
        optimizer.zero_grad()
        rv = Phi_v @ theta - Vt
        rg = Phi_g @ theta - dVt
        data = (w1 / (2 * Nx)) * (rv @ rv) + (w2 / (2 * Nx)) * (rg @ rg)
        pen = theta[penalized]
        if pen.numel():
            base = torch.where(nonneg[penalized], pen.clamp_min(0.0), pen.abs())
            arg = base if q == 1.0 else base.clamp_min(1e-30) ** q
            penalty = alpha * torch.sum(_phi(arg, th, gamma))
        else:
            penalty = theta.new_zeros(())
        return data + penalty

    made_progress = False
    prev = float(closure().detach())
    for _ in range(iterations):
        loss = float(optimizer.step(closure).detach())
        if loss < prev - 1e-15:
            made_progress = True
        prev = loss

    vector_to_parameters(theta.detach(), params)
    summary = {
        "best_step": iterations - 1,
        "best_train_loss": prev,
        "successful_steps": iterations if made_progress else 0,
    }
    model.last_fit_summary = summary
    if verbose:
        logger.debug("Output-weight solve complete  train_loss=%.6e", prev)
    return summary
