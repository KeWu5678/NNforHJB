import torch
from torch._inductor.config import autotune_remote_cache_default
from torch.optim import Optimizer
import numpy as np
from loguru import logger
from .utils import _phi, _dphi, _ddphi, _compute_prox, _compute_dprox


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
        lr (float):         mixing parameter between old and new parameters (default: 0.01)
        max_ls_iter (int):  maximum number of line search iterations (default: 100)
        tolerance_ls (float):  line search tolerance, accepts step if loss_new <= tolerance * loss (default: 1.0)
    """
    
    def __init__(
        self,
        params,
        alpha,
        gamma,
        th=0.5,
        lr=0.01,
        max_ls_iter = 30000,
        tolerance_ls=1.0 + 1e-10,
    ):
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

        
        self.hidden_activations = None  # Store S matrix for Hessian

    def _initialize_q(self, alpha, gamma, th, params, loss):
        """
        Following MATLAB SSN.m (lines 47-53).
        """
        grad_loss = torch.autograd.grad(
            loss, self.param_groups[0]["params"], create_graph=True, retain_graph=True
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

        q = params - (1.0 / self.c) * gf_u
        return q

    def _initilize_G(self, alpha, gamma,th, c, q, params, loss):
        """
        compute the gradient of the loss function. 
        args: 
        - params: the outer weights
        - q: preimage of u w.r.t. the proximal operator
        - c, alpha: hyperparameter in the algorithm
        """
        # D_nonconvex compute d/du (phi(|u|) - |u|)
        D_nonconvex = torch.sign(params) * (_dphi(torch.abs(params), th, gamma) - 1)
        # 
        grad_loss = torch.autograd.grad(loss, self.param_groups[0]["params"], create_graph=True, retain_graph=True)
        grad_flat = torch.cat([g.view(-1) for g in grad_loss])
        return c * (q - params) + alpha * D_nonconvex + grad_flat

    def _DG(self, gamma, th, q, params):
        DPc = _compute_dprox(q, self.alpha / self.c)
        I = torch.eye(params.shape[0], device=params.device, dtype=params.dtype)
        S = self.hidden_activations
        
        return (
            self.c * (I - DPc) 
            + self.alpha * torch.diag(_ddphi(torch.abs(params), th, gamma)) @ DPc 
            + (S.T @ S / S.shape[0]) @ DPc
        )
    
    def _update_parameters(self, u_flat):
        """Helper function to update model parameters from flattened tensor."""
        start = 0
        for p in self.param_groups[0]["params"]:
            numel = p.numel()
            p.data.copy_(u_flat[start:start + numel].view(p.shape))
            start += numel
    
    # Hidden activations are set directly by callers: optimizer.hidden_activations = S.detach()
    def step(self, closure):
        """
        Perform a single optimization step using semismooth Newton method with damping.
        Args:
            closure (callable): A closure that reevaluates the model and returns the loss.   
        Returns:
            loss: The loss value.
        """
        # logger.info(f"SSN Step {self.n_iters}")
        # Get current loss and parameters
        alpha, th, gamma = (self.param_groups[0][k] for k in ("alpha", "th", "gamma"))
        c = alpha / gamma  # SSN constant, lambda in the proximal operator
        iter_ls = 0
        
        loss = closure()      
        params = torch.cat([p.view(-1) for p in self.param_groups[0]["params"]])
        # logger.debug(f"Initial loss: {loss.item():.6e}, penalty: {(self.alpha * torch.sum(self._phi(torch.abs(params)))).item():.6e}")
        
        # Initialize proximal projection, approximative Gradient and Hessian
        q = self._initialize_q(alpha, gamma, th, params, loss)  # Pass loss instead of loss for gradient computation
        Gq = self._initilize_G(q, alpha, gamma, c, q, params, loss)    
        DG = self._DG(gamma, th, q, params)
        I = torch.eye(params.shape[0], device=params.device, dtype=params.dtype)
        theta0 = 1.0 / (1e-12 * torch.norm(DG, p=float('inf')).item())
        
        try:
            system_matrix = DG + (1/theta0) * I
            dq = -torch.linalg.solve(system_matrix, Gq)
            logger.debug(f"Solution (dq) norm: {torch.norm(dq).item():.6e}")
        except Exception as e:
            logger.error(f"Linear solve failed: {e}")
            logger.error(f"Matrix condition was: {torch.linalg.cond(system_matrix).item():.2e}")
            return loss
        
        # Initilize outerweights and loss
        qnew = q + dq
        unew = _compute_prox(qnew, alpha / c)
        self._update_parameters(unew)
        loss_new = closure()
        
        # Initialize the damping parameter
        dq_norm, Gq_norm = torch.norm(dq).item(), max(torch.norm(Gq).item(), 1e-20)  # clamp to avoid div-by-zero
        theta = max(dq_norm / Gq_norm, theta0)
        logger.debug(f"Initial theta0: {theta0:.6e}, step norm: {dq_norm:.6e}")

        # Initialize the Line search with backtracking (matching MATLAB exactly)
        tolerance_ls = self.param_groups[0]["tolerance_ls"]
        max_ls_iter = self.param_groups[0]["max_ls_iter"]
        
        while (torch.isnan(loss_new) or loss_new > tolerance_ls * loss) and iter_ls < max_ls_iter:
            # ------------------------------------------------------------------
            # SAFETY GUARD 1: stop damping if theta is already tiny. Line Search stops
            # ------------------------------------------------------------------
            # if theta < 1e-20:
            #     logger.warning(f"Theta reached {theta:.1e}. Breaking line search loop.")
            #     break
            # ------------------------------------------------------------------
            # if iter_ls < 10 or iter_ls % 20 == 0:  # Only log occasionally
            #     logger.debug(f"Damping step {iter_ls}: theta={theta:.2e}, loss_new={loss_new.item():.10e}")
            # ------------------------------------------------------------------
            # SAFETY GUARD 2: if the tentative step exploded, back-track harder
            # ------------------------------------------------------------------
            # if torch.isnan(unew).any() or torch.isinf(unew).any():
            #     logger.warning("unew contains Inf/NaN – increasing damping (theta *= 4) and retrying")
            #     theta = theta * 4.0  # undo last division so next loop will divide again
            #     iter_ls += 1
            #     break

            # if torch.isnan(loss_new) and iter_ls > 50:
            #     logger.warning("Persistent NaN in line search, stopping")
            #     break

            try:
                qnew = q - torch.linalg.solve(DG + (1/theta) * I, Gq)
                unew = _compute_prox(qnew, self.alpha / self.c)
                self._update_parameters(unew)
                loss_new = closure()
                theta = theta / 4.0 
            except Exception as e:
                logger.error(f"Damped linear solve failed: {e}")
                # Restore original parameters
                self._update_parameters(params)
                return loss 
            
            iter_ls += 1
        
        if iter_ls >= max_ls_iter or torch.isnan(loss_new):
            logger.warning(f"Line search failed after {iter_ls} iterations")
            logger.warning(f"Final lossective values: loss={loss.item():.6e}, loss_new={loss_new.item() if not torch.isnan(loss_new) else 'NaN'}")
            logger.warning(f"Final theta: {theta:.2e}")
            
            # Restore original parameters
            self._update_parameters(params)
            return loss

        return loss
    


        
            





