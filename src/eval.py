"""Performance evaluation of a model's predictions against value samples.

Pure functions of (prediction, target) tensors with no model state: a model
produces ``(V, dV)`` via ``predict_tensors`` and the trainer scores them here.
This is the single home for the data-fidelity numbers: the relative errors and
the value/gradient loss split (the data term of the training objective).

The regularizer is *not* here: the nonconvex penalty lives with the trainer
(``src.PDAP.ssn_solve.nonconvex_penalty``), which adds it to ``data_loss`` to
form the full objective.
"""

from __future__ import annotations

from typing import Dict, Tuple

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


@torch.no_grad()
def region_split_errors(
    v_pred: torch.Tensor,
    dv_pred: torch.Tensor,
    v_true: torch.Tensor,
    dv_true: torch.Tensor,
    near_mask: torch.Tensor,
) -> Dict[str, float]:
    """Relative ``(L2, grad, H1)`` errors split by a near/far boolean mask.

    ``near_mask`` is a length-``N`` boolean tensor over the same samples as the
    predictions: ``True`` = near the switching set, ``False`` = far.

    Two families of per-region numbers are returned (see
    ``experiments/region_split_pendulum/README.md`` for the rationale):

      * ``{near,far}_{l2,grad,h1}`` — region-local *relative* errors (each region
        normalized by its own ‖true‖). Kept for continuity, but confounded here:
        the *far* (interior) region contains the upright equilibrium where V→0 and
        ∇V→0, so its small denominator inflates the relative error — making models
        look spuriously better near the switching set.
      * ``{near,far}_l1_{value,grad,h1}`` — region **mean** per-sample absolute
        (L1) error, normalized by the **global mean** ‖true‖ over *all* samples.
        L1 is the norm of choice near discontinuities (L2 worst-case bounds do not
        generalize across a shock); using the per-sample *mean* (not the sum)
        keeps near/far count-fair (``near`` is only ~10% of samples, so a summed
        L1 would trivially favour it); the shared global denominator removes the
        per-region V→0 confound while keeping near/far on one comparable scale.

    Region counts are included so an empty/degenerate band is visible.
    """
    near = near_mask.to(torch.bool)
    far = ~near
    metrics: Dict[str, float] = {
        "near_count": float(int(near.sum().item())),
        "far_count": float(int(far.sum().item())),
    }
    for tag, sel in (("near", near), ("far", far)):
        if not bool(sel.any()):
            for suffix in ("l2", "grad", "h1", "l1_value", "l1_grad", "l1_h1"):
                metrics[f"{tag}_{suffix}"] = float("nan")
            continue
        l2, grad, h1 = relative_errors(
            v_pred[sel], dv_pred[sel], v_true[sel], dv_true[sel]
        )
        metrics[f"{tag}_l2"] = l2
        metrics[f"{tag}_grad"] = grad
        metrics[f"{tag}_h1"] = h1

    # Mean per-sample L1 (absolute), normalized by global mean ‖true‖ — count-fair
    # across regions and robust to V→0.
    v_err = (v_pred - v_true).abs().reshape(-1)
    g_err = (dv_pred - dv_true).norm(dim=1)
    v_den = v_true.abs().mean().clamp_min(1e-30)
    g_den = dv_true.norm(dim=1).mean().clamp_min(1e-30)
    for tag, sel in (("near", near), ("far", far)):
        if not bool(sel.any()):
            continue
        nv = v_err[sel].mean()
        ng = g_err[sel].mean()
        metrics[f"{tag}_l1_value"] = float((nv / v_den).item())
        metrics[f"{tag}_l1_grad"] = float((ng / g_den).item())
        metrics[f"{tag}_l1_h1"] = float(((nv + ng) / (v_den + g_den)).item())
    return metrics


@torch.no_grad()
def distance_binned_error(
    v_pred: torch.Tensor,
    dv_pred: torch.Tensor,
    v_true: torch.Tensor,
    dv_true: torch.Tensor,
    distance: torch.Tensor,
    n_bins: int = 8,
) -> Dict[str, float]:
    """Per-sample absolute H1 error in quantile bins of distance-to-switching-set.

    The literature-standard diagnostic for behaviour near a discontinuity (Gibbs
    error decays with distance from the kink). Each bin holds an equal share of
    samples (distance quantiles); the reported value is the bin's mean per-sample
    absolute error ``|ΔV| + ‖Δ∇V‖`` divided by that mean over *all* samples — a
    scale-free ratio, so ``bin_ratio > 1`` means worse-than-average there, and the
    curve is comparable across models. Bins are indexed from nearest the switching
    set (``bin1``) outward; bin edges are fixed by the dataset's distance cache.
    """
    per_sample = (v_pred - v_true).abs().reshape(-1) + (dv_pred - dv_true).norm(dim=1)
    overall = per_sample.mean().clamp_min(1e-30)
    d = distance.reshape(-1)
    quantiles = torch.linspace(0.0, 1.0, n_bins + 1, dtype=d.dtype)
    edges = torch.quantile(d, quantiles)
    metrics: Dict[str, float] = {}
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (d >= lo) & (d <= hi) if i == n_bins - 1 else (d >= lo) & (d < hi)
        ratio = (per_sample[in_bin].mean() / overall) if bool(in_bin.any()) else torch.tensor(float("nan"))
        metrics[f"distbin{i + 1}_ratio"] = float(ratio.item())
    return metrics
