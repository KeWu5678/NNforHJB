from typing import Callable, Iterable, Optional

import torch
from torch import Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import Optimizer
from loguru import logger
from .utils import (_compute_prox, _compute_dprox,
                     _penalty_grad, _nonconvex_correction, _nonconvex_correction_dd)

__all__ = ["SSN"]

class SSN(Optimizer):
    """Semismooth Newton optimizer for non-convex regularized problems.
    
    This optimizer solves problems of the form:
        min_u (1/2)*||Sred*u - ref||^2 + alpha*phi(||u||_1,2)
    
    Based on the MATLAB implementation from NonConvexSparseNN.
    
    Args:
        params (iterable):  iterable of parameters to optimize (should be outer weights only)
        alpha (float):      regularization parameter for the penalty function
        gamma (float):      parameter for the non-convex penalty function
        th (float):         interpolation parameter between L1 (th=0) and non-convex (th=1) (default: 0.5)
        max_ls_iter (int):  maximum number of line search iterations (default: 100)
        tolerance_ls (float):  line search tolerance, accepts step if loss_new <= tolerance * loss (default: 1.0)
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
    ) -> None:
        defaults = {
            "lr": lr,
            "alpha": alpha,
            "gamma": gamma,
            "th": th,
            "max_ls_iter": max_ls_iter,
            "tolerance_ls": tolerance_ls,
            }
        super().__init__(params, defaults)
        if len(self.param_groups) > 1:
            raise ValueError("SSN doesn't support per-parameter options")

        self.q: float = 2.0 / (power + 1.0)  # power-transformed exponent
        self.data_hessian: Optional[Tensor] = None
        self.last_step_success: bool = True

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

        # Subtract reg gradient from autograd (which includes it) to get data-only
        reg_grad = _penalty_grad(abs_u, sign_u, alpha, th, gamma, q=qq)
        grad_data = grad_flat - reg_grad

        # gf_u = grad_data + alpha * D_nonconvex
        gf_u = grad_data + alpha * _nonconvex_correction(abs_u, sign_u, th, gamma, q=qq)

        # Override on nonzero entries (NOC condition)
        # For q=1: -alpha * sign(u). For general q: -alpha * q * |u|^{q-1} * sign(u).
        active = abs_u > 0
        if qq == 1.0:
            gf_u = torch.where(active, -alpha * sign_u, gf_u)
        else:
            s = abs_u.clamp(min=1e-30)
            gf_u = torch.where(active, -alpha * qq * s ** (qq - 1) * sign_u, gf_u)

        q_var = params - (1.0 / c) * gf_u
        return q_var

    def _initilize_G(
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

        reg_grad = _penalty_grad(abs_u, sign_u, alpha, th, gamma, q=qq)
        grad_data = grad_flat - reg_grad

        return c * (q_var - params) + alpha * D_nc + grad_data

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
        DPc: Tensor = _compute_dprox(q_var, alpha / c, q=qq, prox_result=params)
        I: Tensor = torch.eye(params.shape[0], device=params.device, dtype=params.dtype)

        corr_dd = _nonconvex_correction_dd(torch.abs(params), th, gamma, q=qq)

        return (
            c * (I - DPc)
            + alpha * torch.diag(corr_dd) @ DPc
            + self.data_hessian @ DPc  # type: ignore[union-attr]
        )

    # data_hessian is set by callers before step():
    # H = (1/N) * (w1 * S'S + w2 * S_grad'S_grad)   cf. paper eq.(3), ddF = 1/N
    def step(self, closure: Callable[[], Tensor]) -> Tensor:
        """
        Perform a single optimization step using semismooth Newton method with damping.
        
        The algorithm follows these steps:
        1. Compute gradient G(q) and generalized Jacobian DG at current q
        2. Solve the damped Newton system: (DG + (1/theta)*I) * dq = -G(q)
        3. Apply proximal operator: u_new = prox(q + dq)
        4. Line search: reduce damping theta until loss decreases
        
        Args:
            closure (callable): A closure that reevaluates the model and returns the loss.   
        Returns:
            Tensor: The loss value after the step (either improved or original if failed).
        """
        # Get current loss and parameters
        alpha, th, gamma, lr = (self.param_groups[0][k] for k in ("alpha", "th", "gamma", "lr"))
        self.last_step_success = True
        
        # IMPORTANT:
        # Using c = alpha/gamma makes 1/c = gamma/alpha huge for small alpha,
        # which can blow up q = u - (1/c)*g and cause line search to fail with massive loss.
        # A stable choice (used in the other SSN implementation in this repo) is:
        c: float = 1.0 + alpha * gamma
        lr = float(self.param_groups[0].get("lr", 1.0))  # mixing factor for step size
        iter_ls: int = 0
        
        loss: Tensor = closure()      
        params: Tensor = parameters_to_vector(self.param_groups[0]["params"])
        # logger.debug(f"Initial loss: {loss.item():.6e}, penalty: {(alpha * torch.sum(self._phi(torch.abs(params)))).item():.6e}")
        
        # Initialize proximal projection, approximative Gradient and Hessian
        q: Tensor = self._initialize_q(alpha, gamma, c, th, params, loss)
        Gq: Tensor = self._initilize_G(alpha, gamma, c, th, q, params, loss)    
        DG: Tensor = self._DG(alpha, gamma, c, th, q, params)
        I: Tensor = torch.eye(params.shape[0], device=params.device, dtype=params.dtype)
        theta0: float = 1.0 / (1e-12 * torch.norm(DG, p=float('inf')).item())
        
        # Initilize outerweights and loss
        try:
            system_matrix: Tensor = DG + (1/theta0) * I
            dq: Tensor = -torch.linalg.solve(system_matrix, Gq)
        except Exception as e:
            logger.error(f"Initial linear solve failed: {e}")
            logger.error(f"Matrix condition was: {torch.linalg.cond(system_matrix).item():.2e}")
            self.last_step_success = False
            return loss
        
        qnew: Tensor = q + dq
        unew: Tensor = _compute_prox(qnew, alpha / c, q=self.q)
        
        # Evaluate loss at full Newton step
        vector_to_parameters(unew, self.param_groups[0]["params"])
        loss_new: Tensor = closure()
        
        # Initialize the damping parameter
        dq_norm: float = torch.norm(dq).item()
        Gq_norm: float = max(torch.norm(Gq).item(), 1e-20)  # clamp to avoid div-by-zero
        theta: float = max(dq_norm / Gq_norm, theta0)
        
        # IMPORTANT: The original code only varied damping in the solve, but always FULL step.
        # As theta→0, dq→0, so qnew→q, unew→prox(q) the line search fails
        # FIX: Use a proper step-size (alpha_ls) to interpolate between params and unew:
        #      utrial = params + alpha_ls * (unew - params)
        
        # Initialize the line search with backtracking (matching MATLAB exactly)
        tolerance_ls: float = self.param_groups[0]["tolerance_ls"]
        max_ls_iter: int = self.param_groups[0]["max_ls_iter"]
        
        min_theta: float = 1e-12  # prevent theta from underflowing to zero
        # check (loss_new - tolerance_ls * loss)
        # check if DG is invertible. and condition number.  
        # check the inner weights （if they are linear independent)
        while (torch.isnan(loss_new) or loss_new > tolerance_ls * loss) and iter_ls < max_ls_iter:
            try:
                theta_safe = max(theta, min_theta)
                qnew = q - lr * torch.linalg.solve(DG + (1/theta_safe) * I, Gq)
                unew = _compute_prox(qnew, alpha / c, q=self.q)
                vector_to_parameters(unew, self.param_groups[0]["params"])
                loss_new = closure()
                theta = max(theta / 4.0, min_theta)
            except Exception as e:
                logger.error(f"Damped linear solve failed: {e}")
                vector_to_parameters(params, self.param_groups[0]["params"])
                self.last_step_success = False
                return loss

            iter_ls += 1
        
        if iter_ls >= max_ls_iter or torch.isnan(loss_new):
            logger.debug(f"Line search failed after {iter_ls} iterations")
            logger.debug(f"loss={loss.item():.6e}, loss_new={loss_new.item() if not torch.isnan(loss_new) else 'NaN'}, theta={theta:.2e}")
            vector_to_parameters(params, self.param_groups[0]["params"])
            self.last_step_success = False
            return loss
        self.last_step_success = True
        return loss_new
