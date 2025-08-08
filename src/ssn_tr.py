import torch
from loguru import logger
from .mpcg import mpcg
from .ssn import SSN


class SSN_TR(SSN):
    """Trust-Region Semismooth Newton optimizer.

    Mirrors the MATLAB SSN_TR.m flow while keeping compatibility with SSN API.
    Uses the same proximal and penalty helpers from SSN and a TR-Krylov inner
    solve via mpcg.
    """

    def __init__(self, params, alpha: float, gamma: float, th: float = 0.5, lr: float = 1.0):
        super().__init__(params=params, alpha=alpha, gamma=gamma, th=th, lr=lr)
        # Trust-region parameters
        self.sigmamax = 100.0
        self.sigma = 0.01 * self.sigmamax
        self.maxiter_tr = 1  # one TR step per optimizer.step()

    def _flatten_params(self) -> torch.Tensor:
        return torch.cat([p.view(-1) for p in self.param_groups[0]["params"]])

    def _set_params_from_flat(self, flat: torch.Tensor) -> None:
        self._update_parameters(flat)

    def step(self, closure):
        loss = closure()
        params = self._flatten_params()

        # Prepare quantities
        q = self._transform_param2q(params, loss)
        Gq = self._Gradient(q, params, loss)
        DP = self._compute_dprox(q, self.alpha / self.c)

        # Build DG using the same Hessian routine as SSN for consistency
        DG = self._Hessian(q, params, loss)

        # Krylov/TR step
        I_active = torch.diag(DP) != 0
        kmaxit = max(1, int(2 * I_active.sum().item()))
        
        dq, flag, pred, relres, iter_cg = mpcg(DG, -Gq, 1e-3, kmaxit, self.sigma, DP)

        # Tentative update and trust-region ratio
        qnew = q + dq
        unew = self._compute_prox(qnew, self.alpha / self.c)
        self._set_params_from_flat(unew)
        loss_new = closure()

        sigmaold = self.sigma

        if not torch.isfinite(loss_new) or (loss_new > loss + 1e-10 * torch.abs(loss)):
            # reject
            self.sigma = 0.2 * self.sigma
            logger.debug(f"TR reject: Δloss={(loss_new - loss).item():.3e} {sigmaold:.2e}->{self.sigma:.2e}")
            # restore original params
            self._set_params_from_flat(params)
            return loss
        else:
            model = pred
            rho = ((loss_new - loss) / model) if model != 0 else torch.tensor(0.0, device=params.device, dtype=params.dtype)
            rho_val = float(rho)

            if abs(rho_val - 1) < 0.2 or abs((loss - loss_new).item()) / (abs(loss.item()) + 1e-30) < 1e-10:
                self.sigma = min(2 * self.sigma, self.sigmamax)
            elif abs(rho_val - 1) > 0.6:
                self.sigma = 0.4 * self.sigma
            logger.debug(f"TR accept: rho={rho_val:.3f} {sigmaold:.2e}->{self.sigma:.2e}")

            # keep updated params (already set)
            self.n_iters += 1
            self.consecutive_failures = 0
            return loss

__all__ = ["SSN_TR"]

