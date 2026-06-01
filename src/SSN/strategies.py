"""Globalization strategies for the SSN step.

Both members of the trust-region family (scipy/Ceres classify Levenberg-Marquardt
as a trust-region method); they differ only in how the step subproblem is solved:

  * ``solve_levenberg_marquardt`` — adaptive diagonal damping ``(DG + (1/theta) I)``
    with backtracking on ``theta`` (the historical ``SSN.step``).
  * ``solve_steihaug_cg`` — a truncated/Steihaug CG trust-region solve via ``mpcg``
    with an explicit radius ``sigma`` (the historical ``SSN_TR.step``).

Each is a free function taking the optimizer instance ``opt`` (for its ``_prox`` /
``_trial`` primitives, config and ``self.state``) plus the already-built system
``(loss, params, q, Gq, DG)``.  They set ``opt.last_step_success`` and return the
post-step loss, restoring the params snapshot on rejection.
"""

import logging
from typing import Callable

import torch
from torch import Tensor

from .mpcg import mpcg

logger = logging.getLogger(__name__)

__all__ = ["solve_levenberg_marquardt", "solve_steihaug_cg"]


def solve_levenberg_marquardt(
    opt, closure: Callable[[], Tensor], loss: Tensor,
    params: Tensor, q: Tensor, Gq: Tensor, DG: Tensor, c: float, lr: float,
) -> Tensor:
    """Damped-Newton step with backtracking on the damping ``theta``."""
    group = opt.param_groups[0]
    tolerance_ls: float = group["tolerance_ls"]
    max_ls_iter: int = group["max_ls_iter"]
    iter_ls: int = 0

    I: Tensor = torch.eye(params.shape[0], device=params.device, dtype=params.dtype)
    theta0: float = 1.0 / (1e-12 * torch.norm(DG, p=float("inf")).item())

    try:
        system_matrix: Tensor = DG + (1 / theta0) * I
        dq: Tensor = -torch.linalg.solve(system_matrix, Gq)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Initial linear solve failed: {e}")
        logger.error(f"Matrix condition was: {torch.linalg.cond(system_matrix).item():.2e}")
        opt.last_step_success = False
        return loss

    qnew: Tensor = q + dq
    _, loss_new = opt._trial(closure, qnew, c)

    # Initialize the damping parameter
    dq_norm: float = torch.norm(dq).item()
    Gq_norm: float = max(torch.norm(Gq).item(), 1e-20)  # clamp to avoid div-by-zero
    theta: float = max(dq_norm / Gq_norm, theta0)

    min_theta: float = 1e-12  # prevent theta from underflowing to zero
    while (torch.isnan(loss_new) or loss_new > tolerance_ls * loss) and iter_ls < max_ls_iter:
        try:
            theta_safe = max(theta, min_theta)
            qnew = q - lr * torch.linalg.solve(DG + (1 / theta_safe) * I, Gq)
            _, loss_new = opt._trial(closure, qnew, c)
            theta = max(theta / 4.0, min_theta)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Damped linear solve failed: {e}")
            opt._set_flat(params)
            opt.last_step_success = False
            return loss
        iter_ls += 1

    if iter_ls >= max_ls_iter or torch.isnan(loss_new):
        logger.debug(f"Line search failed after {iter_ls} iterations")
        logger.debug(
            f"loss={loss.item():.6e}, "
            f"loss_new={loss_new.item() if not torch.isnan(loss_new) else 'NaN'}, "
            f"theta={theta:.2e}"
        )
        opt._set_flat(params)
        opt.last_step_success = False
        return loss
    opt.last_step_success = True
    return loss_new


def solve_steihaug_cg(
    opt, closure: Callable[[], Tensor], loss: Tensor,
    params: Tensor, q: Tensor, Gq: Tensor, DG: Tensor, c: float, lr: float,
) -> Tensor:
    """Trust-region step via a truncated/Steihaug CG solve (mpcg) with radius sigma."""
    group = opt.param_groups[0]
    sigmamax: float = group["sigmamax"]
    st = opt.state[opt._params[0]]
    sigma: float = st.setdefault("sigma", 0.01 * sigmamax)

    DP: Tensor = opt._dprox(q, c, prox_result=params)
    I_active: Tensor = torch.diag(DP) != 0
    kmaxit: int = max(1, int(2 * I_active.sum().item()))

    dq, tr_flag, pred, _, _ = mpcg(DG, -Gq, 1e-3, kmaxit, sigma, DP)

    qnew: Tensor = q + lr * dq
    _, loss_new = opt._trial(closure, qnew, c)

    if not torch.isfinite(loss_new) or (loss_new > loss + 1e-10 * torch.abs(loss)):
        # reject: shrink the trust region and restore params
        st["sigma"] = 0.2 * sigma
        logger.debug(f"TR reject: dloss={(loss_new - loss).item():.3e} {sigma:.2e}->{st['sigma']:.2e}")
        opt._set_flat(params)
        opt.last_step_success = False
        return loss

    # Accept.  ``pred`` from mpcg is a model decrease for the Newton system, not a
    # reliable loss-decrease prediction, so adapt sigma on acceptance / boundary hit
    # rather than via a TR ratio (which tends to collapse sigma to ~0 and stall).
    if tr_flag == "radius":
        st["sigma"] = min(2.0 * sigma, sigmamax)
    else:
        st["sigma"] = min(1.2 * sigma, sigmamax)
    opt.last_step_success = True
    return loss_new
