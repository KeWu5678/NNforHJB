"""Performance evaluation of a model's predictions against value samples.

Pure functions of (prediction, target) tensors with no model state: a model
produces ``(V, dV)`` via ``predict_tensors`` and the trainer scores them here.
This is the single home for the data-fidelity numbers (relative errors and the
value/gradient loss split) that were previously duplicated verbatim inside each
model's ``_compute_relative_errors`` / ``compute_loss``.

The regularizer is *not* here: the nonconvex penalty depends on which parameters
a given model penalizes, so it stays the model's responsibility and is added to
``data_loss`` to form the full training objective.
"""

from __future__ import annotations

from typing import Tuple

import torch


def data_loss_terms(
    v_pred: torch.Tensor,
    dv_pred: torch.Tensor,
    v_true: torch.Tensor,
    dv_true: torch.Tensor,
    loss_weights: Tuple[float, float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(data_loss, value_loss, grad_loss)`` for the data-fidelity term.

    MSE is normalized by ``Nx = N * d`` (matching the MATLAB reference, where
    ``Nx = numel(xhat) = d * N``) and halved.  ``data_loss`` weights the two by
    ``loss_weights = (w1, w2)``.  Differentiable (plain torch ops), so it can sit
    inside an SSN closure as well as serve evaluation.
    """
    nx = v_true.shape[0] * dv_true.shape[1]
    value_loss = torch.sum((v_pred - v_true) ** 2) / (2 * nx)
    grad_loss = torch.sum((dv_pred - dv_true) ** 2) / (2 * nx)
    w1, w2 = loss_weights
    data_loss = w1 * value_loss + w2 * grad_loss
    return data_loss, value_loss, grad_loss


@torch.no_grad()
def relative_errors(
    v_pred: torch.Tensor,
    dv_pred: torch.Tensor,
    v_true: torch.Tensor,
    dv_true: torch.Tensor,
) -> Tuple[float, float, float]:
    """Return relative ``(L2, gradient, H1)`` errors as plain floats.

    Each is ``||pred - true|| / ||true||`` in the relevant norm; denominators are
    clamped to ``1e-30`` to stay finite on all-zero targets.
    """
    v_diff = torch.sum((v_pred - v_true) ** 2)
    dv_diff = torch.sum((dv_pred - dv_true) ** 2)
    v_ref = torch.sum(v_true ** 2)
    dv_ref = torch.sum(dv_true ** 2)

    err_l2 = torch.sqrt(v_diff / v_ref.clamp_min(1e-30))
    err_grad = torch.sqrt(dv_diff / dv_ref.clamp_min(1e-30))
    err_h1 = torch.sqrt((v_diff + dv_diff) / (v_ref + dv_ref).clamp_min(1e-30))
    return float(err_l2.item()), float(err_grad.item()), float(err_h1.item())
