from tabnanny import verbose
from typing import Callable, Iterable, Optional

import torch
from torch import Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import Optimizer
from loguru import logger
from .utils import _dphi, _ddphi, _compute_prox, _compute_dprox

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
        max_ls_iter: int = 30000,
        tolerance_ls: float = 1.0 + 1e-10,
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

        
        self.hidden_activations: Optional[Tensor] = None  # Store S matrix for Hessian
        # Expose whether the last SSN step was accepted.
        # This is useful for training-loop early stopping / fallbacks.
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
        Following MATLAB SSN.m (lines 47-53). It assume the NOC it fufilled and back calculate the proxy.
        """
        grad_loss = torch.autograd.grad(
            loss, self.param_groups[0]["params"], create_graph=False, retain_graph=True
        )
        grad_flat = torch.cat([g.view(-1) for g in grad_loss])

        # gradient of F w.r.t the outerweight u (cf.p27 Kontantin)
        gf_u = (
            grad_flat +
            alpha * torch.sign(params) * (_dphi(torch.abs(params), th, gamma) - 1.0) 
        )

        # Override on nonzero entries: gf0_i = -alpha * sign(u_i)
        nonzero_mask = torch.abs(params) > 0
        gf_u = torch.where(nonzero_mask, -alpha * torch.sign(params), gf_u)

        q = params - (1.0 / c) * gf_u
        return q

    def _initilize_G(
        self, 
        alpha: float, 
        gamma: float, 
        c: float, 
        th: float, 
        q: Tensor, 
        params: Tensor, 
        loss: Tensor
    ) -> Tensor:
        """
        Compute the gradient G(q) of the reformulated objective.
        
        This computes the gradient of the semismooth Newton system with respect
        to the proximal preimage q.
        
        Args:
            alpha: Regularization parameter for the penalty function.
            gamma: Parameter for the non-convex penalty function.
            c: Scaling constant (typically 1 + alpha*gamma).
            th: Interpolation parameter between L1 (th=0) and non-convex (th=1).
            q: Proximal preimage tensor.
            params: Current outer weights as a flattened tensor.
            loss: Current loss tensor (used for gradient computation).
            
        Returns:
            The gradient G(q) of the semismooth Newton system.
        """
        # D_nonconvex compute d/du (phi(|u|) - |u|)
        D_nonconvex = torch.sign(params) * (_dphi(torch.abs(params), th, gamma) - 1)
        grad_loss = torch.autograd.grad(loss, self.param_groups[0]["params"], retain_graph=True)
        grad_flat = torch.cat([g.view(-1) for g in grad_loss])
        return c * (q - params) + alpha * D_nonconvex + grad_flat

    def _DG(
        self, 
        alpha: float, 
        gamma: float, 
        c: float, 
        th: float, 
        q: Tensor, 
        params: Tensor
    ) -> Tensor:
        """
        Compute the generalized Jacobian DG of the semismooth Newton system.
        
        This computes the (generalized) derivative of G(q) which forms the 
        linear system for the Newton step: DG * dq = -G(q).
        
        Args:
            alpha: Regularization parameter for the penalty function.
            gamma: Parameter for the non-convex penalty function.
            c: Scaling constant (typically 1 + alpha*gamma).
            th: Interpolation parameter between L1 (th=0) and non-convex (th=1).
            q: Proximal preimage tensor.
            params: Current outer weights as a flattened tensor.
            
        Returns:
            Tensor: The generalized Jacobian matrix DG (n x n).
        """
        DPc: Tensor = _compute_dprox(q, alpha / c)
        I: Tensor = torch.eye(params.shape[0], device=params.device, dtype=params.dtype)
        S: Tensor = self.hidden_activations  # type: ignore[assignment]
        
        return (
            c * (I - DPc) 
            + alpha * torch.diag(_ddphi(torch.abs(params), th, gamma)) @ DPc 
            + (S.T @ S / S.shape[0]) @ DPc
        )
        
    # Hidden activations are set directly by callers: optimizer.hidden_activations = S.detach()
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
        unew: Tensor = _compute_prox(qnew, alpha / c)
        
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
                unew = _compute_prox(qnew, alpha / c)
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
            if verbose:
                logger.warning(f"Line search failed after {iter_ls} iterations")
                logger.warning(f"Final lossective values: loss={loss.item():.6e}, loss_new={loss_new.item() if not torch.isnan(loss_new) else 'NaN'}")
                logger.warning(f"Final theta: {theta:.2e}")
            vector_to_parameters(params, self.param_groups[0]["params"])
            self.last_step_success = False
            return loss
        self.last_step_success = True
        return loss_new
