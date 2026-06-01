from typing import Callable, Iterable, Optional

import torch
from torch import Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import Optimizer
from loguru import logger
from .prox import _compute_prox, _compute_dprox
from .penalty import _penalty_grad, _nonconvex_correction, _nonconvex_correction_dd
from .strategies import solve_levenberg_marquardt, solve_steihaug_cg

__all__ = ["SSN"]

# Globalization strategies, selected via the ``method`` argument.
_STRATEGIES = {
    "levenberg_marquardt": solve_levenberg_marquardt,
    "steihaug_cg": solve_steihaug_cg,
}

class SSN(Optimizer):
    """Semismooth Newton optimizer for non-convex regularized problems.

    Solves   min_u  (1/2)||S u - ref||^2 + alpha * phi(|u|^q)   on the outer
    weights, with a semismooth Newton reformulation.  Based on the MATLAB
    NonConvexSparseNN reference.  One class now covers what used to be three:

      * signed network        -> default masks (penalise all, no nonneg)
      * semiconcave model      -> ``penalized_mask`` / ``nonneg_mask``
      * trust-region variant   -> ``method="steihaug_cg"``

    Symbol table (kept verbatim from the paper / MATLAB for traceability):
      q       proximal preimage (the SSN working variable)
      Gq      residual G(q) of the reformulated optimality system
      DG      generalized (semismooth) Jacobian of G
      dq      Newton step,  solved from (DG + damping) dq = -Gq
      theta   damping parameter of the Levenberg-Marquardt strategy
      sigma   trust-region radius of the Steihaug-CG strategy
      c       1 + alpha*gamma  (stable scaling of the prox parameter)
      alpha_vec  per-coordinate alpha (alpha on penalised coords, 0 on free)

    Args:
        params:        parameters to optimize (outer weights only).
        alpha, gamma:  regularization / non-convexity parameters.
        th:            L1 (th=1) <-> non-convex (th=0) interpolation (default 0.5).
        lr:            step mixing factor (default 1.0).
        max_ls_iter:   max line-search iterations (levenberg_marquardt).
        tolerance_ls:  accept step if loss_new <= tolerance_ls * loss.
        power:         activation power; sets q = 2/(power+1).
        method:        globalization strategy, ``"levenberg_marquardt"`` (damped
                       Newton) or ``"steihaug_cg"`` (trust region).
        sigmamax:      max trust-region radius (steihaug_cg).
        tolerance_grad: early-exit tol on ||Gq||_inf; 0.0 disables it.
        penalized_mask: bool mask of penalised coords (default: all True).
        nonneg_mask:    bool mask of coords clamped >= 0 (default: all False).

    Note:
        ``data_hessian`` (the Gauss-Newton Hessian of the data term) must be set
        by the caller on the instance before each ``step`` — an API wart inherited
        from the original design: it really is a per-step input that would be
        cleaner to pass through the closure, but callers (model.py,
        semiconcave_model.py) currently poke it as an attribute.
    """
    
    def __init__(
        self,
        params: Iterable[Tensor],
        alpha: float,
        gamma: float,
        th: float = 0.5,
        lr: float = 1.0,
        max_ls_iter: int = 500,
        tolerance_ls: float = 1.0 + 1e-8,
        power: float = 1.0,
        method: str = "levenberg_marquardt",
        sigmamax: float = 10.0,
        tolerance_grad: float = 0.0,
        penalized_mask: Optional[Tensor] = None,
        nonneg_mask: Optional[Tensor] = None,
    ) -> None:
        if method not in _STRATEGIES:
            raise ValueError(
                f"unknown method {method!r}; expected one of {sorted(_STRATEGIES)}"
            )
        # State placement (LBFGS convention): tunable hyperparameters live in
        # ``defaults``/``param_groups[0]``; problem structure (masks, alpha_vec,
        # power/q, data_hessian) lives in instance attributes; transient
        # cross-step state (trust-region ``sigma``) lives in ``self.state``.
        defaults = {
            "lr": lr,
            "alpha": alpha,
            "gamma": gamma,
            "th": th,
            "max_ls_iter": max_ls_iter,
            "tolerance_ls": tolerance_ls,
            "method": method,
            "sigmamax": sigmamax,
            # First-order optimality tolerance for early exit on ||Gq||_inf.
            # 0.0 disables it (preserves the historical always-step behaviour the
            # PDAP loops are tuned against); set >0 to enable early stopping.
            "tolerance_grad": tolerance_grad,
            }
        super().__init__(params, defaults)
        if len(self.param_groups) > 1:
            raise ValueError("SSN doesn't support per-parameter options")

        self._params = self.param_groups[0]["params"]
        self.q: float = 2.0 / (power + 1.0)  # power-transformed exponent
        self.data_hessian: Optional[Tensor] = None
        self.last_step_success: bool = True

        # --- proximal structure (problem-fixed, not tunable) ---
        # Defaults recover the signed network exactly: every coordinate is
        # penalised with scalar ``alpha`` and there is no nonnegativity
        # constraint.  A semiconcave model passes masks to penalise only the
        # convex ``c`` block and to clamp it nonnegative.
        flat = parameters_to_vector(self.param_groups[0]["params"])
        P = flat.shape[0]
        if penalized_mask is None:
            self.penalized_mask = torch.ones(P, dtype=torch.bool, device=flat.device)
        else:
            self.penalized_mask = penalized_mask.to(dtype=torch.bool, device=flat.device)
            if self.penalized_mask.shape[0] != P:
                raise ValueError(
                    f"penalized_mask length {self.penalized_mask.shape[0]} != param vector length {P}"
                )
        if nonneg_mask is None:
            self.nonneg_mask = torch.zeros(P, dtype=torch.bool, device=flat.device)
        else:
            self.nonneg_mask = nonneg_mask.to(dtype=torch.bool, device=flat.device)
            if self.nonneg_mask.shape[0] != P:
                raise ValueError("nonneg_mask length must match param vector length")
        # per-coordinate alpha: ``alpha`` on penalised coords, 0 on free coords.
        self.alpha_vec: Tensor = torch.where(
            self.penalized_mask, torch.full_like(flat, float(alpha)), torch.zeros_like(flat)
        )

    # ------------------------------------------------------------------ #
    # Masked nonnegative proximal and its Jacobian.
    # Defaults (penalised everywhere, no nonneg coords) reduce these to the
    # plain signed soft-threshold prox and its diagonal Jacobian.
    # ------------------------------------------------------------------ #
    def _mu(self, c: float) -> Tensor:
        """Per-coordinate proximal parameter alpha_vec / c (0 => identity prox)."""
        return self.alpha_vec / c

    def _prox(self, v: Tensor, c: float) -> Tensor:
        out = _compute_prox(v, self._mu(c), q=self.q)
        kill = self.nonneg_mask & (v < 0)
        return torch.where(kill, torch.zeros_like(out), out)

    def _dprox(self, v: Tensor, c: float, prox_result: Tensor) -> Tensor:
        DP = _compute_dprox(v, self._mu(c), q=self.q, prox_result=prox_result)
        kill = self.nonneg_mask & (v < 0)
        diag = torch.where(kill, torch.zeros_like(prox_result), torch.diagonal(DP))
        return torch.diag(diag)

    # ------------------------------------------------------------------ #
    # Flat parameter <-> vector primitives and the trial-evaluation step,
    # shared by both globalization strategies (cf. torch.optim.LBFGS).
    # ------------------------------------------------------------------ #
    def _clone_flat(self) -> Tensor:
        """Snapshot the parameters as a flat vector (the line-search restore point)."""
        return parameters_to_vector(self._params)

    def _set_flat(self, vec: Tensor) -> None:
        vector_to_parameters(vec, self._params)

    def _trial(self, closure: Callable[[], Tensor], qnew: Tensor, c: float) -> tuple[Tensor, Tensor]:
        """prox(qnew) -> set params -> evaluate.  Returns (unew, loss_new)."""
        unew = self._prox(qnew, c)
        self._set_flat(unew)
        return unew, closure()

    def _initialize_q(
        self,
        alpha: float,
        gamma: float,
        c: float,
        th: float,
        params: Tensor,
        loss: Tensor
    ) -> Tensor:
        """
        Following MATLAB SSN.m (lines 47-53). Assumes the NOC is fulfilled
        and back-calculates the proximal preimage q_var.
        """
        qq = self.q
        grad_loss = torch.autograd.grad(
            loss, self.param_groups[0]["params"], create_graph=False, retain_graph=True
        )
        grad_flat = torch.cat([g.view(-1) for g in grad_loss])

        abs_u = torch.abs(params)
        sign_u = torch.sign(params)

        # Subtract reg gradient from autograd (which includes it) to get data-only.
        # alpha_vec is ``alpha`` on penalised coords and 0 on free coords, so the
        # penalty terms vanish on free coords automatically.
        reg_grad = _penalty_grad(abs_u, sign_u, self.alpha_vec, th, gamma, q=qq)
        grad_data = grad_flat - reg_grad

        # gf_u = grad_data + alpha * D_nonconvex
        gf_u = grad_data + self.alpha_vec * _nonconvex_correction(abs_u, sign_u, th, gamma, q=qq)

        # Override on nonzero *penalised* entries (NOC condition).
        # For q=1: -alpha * sign(u). For general q: -alpha * q * |u|^{q-1} * sign(u).
        active = (abs_u > 0) & self.penalized_mask
        if qq == 1.0:
            gf_u = torch.where(active, -self.alpha_vec * sign_u, gf_u)
        else:
            s = abs_u.clamp(min=1e-30)
            gf_u = torch.where(active, -self.alpha_vec * qq * s ** (qq - 1) * sign_u, gf_u)

        q_var = params - (1.0 / c) * gf_u
        # On free (unpenalised) coords the proximal is the identity, whose preimage
        # is the parameter itself; this makes G reduce to a plain Newton step there.
        q_var = torch.where(self.penalized_mask, q_var, params)
        return q_var

    def _initialize_G(
        self,
        alpha: float,
        gamma: float,
        c: float,
        th: float,
        q_var: Tensor,
        params: Tensor,
        loss: Tensor
    ) -> Tensor:
        """
        Compute the gradient G(q) of the reformulated objective.
        """
        qq = self.q
        abs_u = torch.abs(params)
        sign_u = torch.sign(params)

        D_nc = _nonconvex_correction(abs_u, sign_u, th, gamma, q=qq)

        grad_loss = torch.autograd.grad(loss, self.param_groups[0]["params"], retain_graph=True)
        grad_flat = torch.cat([g.view(-1) for g in grad_loss])

        reg_grad = _penalty_grad(abs_u, sign_u, self.alpha_vec, th, gamma, q=qq)
        grad_data = grad_flat - reg_grad

        return c * (q_var - params) + self.alpha_vec * D_nc + grad_data

    def _DG(
        self,
        alpha: float,
        gamma: float,
        c: float,
        th: float,
        q_var: Tensor,
        params: Tensor
    ) -> Tensor:
        """
        Compute the generalized Jacobian DG of the semismooth Newton system.
        DG * dq = -G(q).
        """
        qq = self.q
        DPc: Tensor = self._dprox(q_var, c, prox_result=params)
        I: Tensor = torch.eye(params.shape[0], device=params.device, dtype=params.dtype)

        corr_dd = _nonconvex_correction_dd(torch.abs(params), th, gamma, q=qq)

        return (
            c * (I - DPc)
            + torch.diag(self.alpha_vec * corr_dd) @ DPc
            + self.data_hessian @ DPc  # type: ignore[union-attr]
        )

    # data_hessian is set by callers before step():
    # H = (1/N) * (w1 * S'S + w2 * S_grad'S_grad)   cf. paper eq.(3), ddF = 1/N
    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor]) -> Tensor:
        """Perform one semismooth-Newton step.

        Builds the SSN system (preimage ``q``, residual ``Gq``, Jacobian ``DG``)
        then hands it to the globalization strategy selected by ``method``
        (``levenberg_marquardt`` damped Newton, or ``steihaug_cg`` trust region).

        Args:
            closure (callable): re-evaluates the model and returns the loss.
        Returns:
            Tensor: the loss after the step (or the original loss if it failed).
        """
        # Run the closure with grad enabled even though step() is no_grad, so the
        # SSN-system builders can call autograd.grad on the loss (cf. LBFGS).
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        alpha, th, gamma = group["alpha"], group["th"], group["gamma"]
        lr = float(group.get("lr", 1.0))  # mixing factor for step size
        self.last_step_success = True

        # c = alpha/gamma would make 1/c huge for small alpha and blow up
        # q = u - (1/c)*g; the stable choice (matching the MATLAB reference) is:
        c: float = 1.0 + alpha * gamma

        loss: Tensor = closure()
        params: Tensor = self._clone_flat()

        # Build the SSN system: proximal preimage, residual G(q), Jacobian DG.
        q: Tensor = self._initialize_q(alpha, gamma, c, th, params, loss)
        Gq: Tensor = self._initialize_G(alpha, gamma, c, th, q, params, loss)

        # Optional first-order optimality early exit (disabled when tolerance_grad=0).
        tolerance_grad: float = group["tolerance_grad"]
        if tolerance_grad > 0.0 and torch.norm(Gq, p=float("inf")).item() <= tolerance_grad:
            return loss

        DG: Tensor = self._DG(alpha, gamma, c, th, q, params)

        strategy = _STRATEGIES[group["method"]]
        return strategy(self, closure, loss, params, q, Gq, DG, c, lr)
