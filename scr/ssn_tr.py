from typing import Callable, Iterable

import torch
from torch import Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from loguru import logger
from .mpcg import mpcg
from .ssn import SSN
from .utils import _compute_prox, _compute_dprox


class SSN_TR(SSN):
    """Trust-Region Semismooth Newton optimizer.

    Mirrors the MATLAB SSN_TR.m flow while keeping compatibility with SSN API.
    Uses the same proximal and penalty helpers from SSN and a TR-Krylov inner
    solve via mpcg.
    
    Args:
        params (iterable): iterable of parameters to optimize (should be outer weights only)
        alpha (float): regularization parameter for the penalty function
        gamma (float): parameter for the non-convex penalty function
        th (float): interpolation parameter between L1 (th=0) and non-convex (th=1) (default: 0.5)
        lr (float): mixing parameter between old and new parameters (default: 1.0)
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        alpha: float,
        gamma: float,
        th: float = 0.5,
        lr: float = 1.0,
        sigmamax: float = 10.0
    ) -> None:
        super().__init__(params=params, alpha=alpha, gamma=gamma, th=th, lr=lr)
        # Trust-region parameters
        self.sigmamax: float = sigmamax
        self.sigma: float = 0.01 * self.sigmamax

    def step(self, closure: Callable[[], Tensor]) -> Tensor:
        """
        One trust-region SSN step using Krylov/TR solve (mpcg).
        
        This is the SSN_TR counterpart of SSN.step(), but replaces the direct
        linear solve with a (preconditioned) TR-Krylov solve.
        """
        alpha, th, gamma, lr = (self.param_groups[0][k] for k in ("alpha", "th", "gamma", "lr"))
        # Keep c consistent with SSN.step() (stable scaling)
        c: float = 1.0 + alpha * gamma
        lr = float(self.param_groups[0].get("lr", 1.0))

        loss: Tensor = closure()
        params: Tensor = parameters_to_vector(self.param_groups[0]["params"])

        # Prepare quantities (reuse SSN helpers)
        q: Tensor = self._initialize_q(alpha, gamma, c, th, params, loss)
        Gq: Tensor = self._initilize_G(alpha, gamma, c, th, q, params, loss)
        DP: Tensor = _compute_dprox(q, alpha / c)
        DG: Tensor = self._DG(alpha, gamma, c, th, q, params)

        # Krylov/TR step: 
        I_active: Tensor = torch.diag(DP) != 0
        kmaxit: int = max(1, int(2 * I_active.sum().item()))
        
        dq: Tensor
        tr_flag: str
        pred: float
        dq, tr_flag, pred, _, _ = mpcg(DG, -Gq, 1e-3, kmaxit, self.sigma, DP)

        # Tentative update and trust-region ratio
        qnew: Tensor = q + lr * dq
        unew: Tensor = _compute_prox(qnew, alpha / c)
        vector_to_parameters(unew, self.param_groups[0]["params"])
        loss_new: Tensor = closure()

        sigmaold: float = self.sigma

        if not torch.isfinite(loss_new) or (loss_new > loss + 1e-10 * torch.abs(loss)):
            # reject
            self.sigma = 0.2 * self.sigma
            logger.debug(f"TR reject: Δloss={(loss_new - loss).item():.3e} {sigmaold:.2e}->{self.sigma:.2e}")
            # restore original params
            vector_to_parameters(params, self.param_groups[0]["params"])
            return loss
        else:
            # Accept step. IMPORTANT:
            # `pred` returned by mpcg is a decrease in the quadratic model for the
            # *Newton system*, not a reliable prediction of the *loss* decrease here.
            # Using it to compute a TR ratio (rho) tends to shrink sigma to ~0 and
            # makes training stall. Instead, adapt sigma based on acceptance and
            # whether the step hit the TR boundary.
            if tr_flag == "radius":
                self.sigma = min(2.0 * self.sigma, self.sigmamax)
            else:
                # mild expansion to avoid sigma collapsing
                self.sigma = min(1.2 * self.sigma, self.sigmamax)
            # logger.debug(f"TR accept: flag={tr_flag} pred={pred:.3e} {sigmaold:.2e}->{self.sigma:.2e}")

            # keep updated params (already set)
            return loss_new

__all__ = ["SSN_TR"]

