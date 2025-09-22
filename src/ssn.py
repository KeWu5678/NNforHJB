import torch
from torch.optim import Optimizer
import numpy as np
from loguru import logger
from .utils import _phi, _dphi, _ddphi, _compute_prox, _compute_dprox

# This is a standalone SSN optimizer that doesn't depend on deepxde


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
        lr (float):         mixing parameter between old and new parameters (default: 1.0)
    """
    
    def __init__(
        self,
        params,
        alpha,
        gamma,
        th=0.5,
        lr=0.01,
    ):
        # Learning rate should be passed to the superclass optimizer
        defaults = {"lr": lr}
        super().__init__(params, defaults)
        
        # param_groups is a list of dictionaries, each containing a 'params' key, defaulted in the PyTorch optimizer class.
        if len(self.param_groups) > 1:
            raise ValueError("SSN doesn't support per-parameter options")
        
        self.alpha = alpha
        self.gamma = gamma
        self.th = th
        
        # SSN constant, lambda in the proximal operator
        self.c = 1 + alpha * gamma
        
        self.q = None  # preimage of outer weights w.r.t. the proximal operator
        self.n_iters = 0    # Number of SSN iterations
        self.consecutive_failures = 0
        self.hidden_activations = None  # Store S matrix for Hessian
    
    def _Gradient(self, q, params, loss):
        # - params: the outer weights
        # - q: preimage of u w.r.t. the proximal operator
        D_nonconvex = torch.sign(params) * (_ddphi(torch.abs(params), self.th, self.gamma) - 1)
        try:
            grad_loss = torch.autograd.grad(loss, self.param_groups[0]["params"], create_graph=True, retain_graph=True)
            grad_flat = torch.cat([g.view(-1) for g in grad_loss])
        except Exception as e:
            logger.error(f"Failed to compute gradients in _Gradient: {e}")
            grad_flat = torch.zeros_like(params)
        return self.c * (q - params) + self.alpha * D_nonconvex + grad_flat

    def _Hessian(self, q, params, loss):
        DD_nonconvex = _ddphi(torch.abs(params), self.th, self.gamma)
        DPc = _compute_dprox(q, self.alpha / self.c)
        I = torch.eye(params.shape[0], device=params.device, dtype=params.dtype)
        
        # Use correct Gauss-Newton Hessian: S^T * S * DPc
        # where S is the hidden activations matrix
        if self.hidden_activations is not None:
            S = self.hidden_activations  # shape: (batch_size, n_neurons)
            STS = S.T @ S / S.shape[0]    # shape: (n_neurons, n_neurons), normalized by batch size
            hessian_data_term = STS @ DPc
            # logger.debug(f"Using correct Gauss-Newton Hessian: S^T*S*DPc, S shape: {S.shape}")
        else:
            # Fallback to identity if hidden activations not available
            hessian_data_term = I @ DPc
            logger.warning("Hidden activations not available, using identity approximation")
        
        return self.c * (I - DPc) + self.alpha * torch.diag(DD_nonconvex) @ DPc + hessian_data_term
    
    def _transform_param2q(self, params, loss):
        """Compute q from current params following MATLAB SSN.m (lines 47-53).

        gf0 = alpha * Dphima(u) + grad_loss(u), where
        Dphima(u) = (dphi(|u|) - 1) * sign(u).
        For nonzero entries of u, set gf0_i = -alpha * sign(u_i).
        Then q = u - (1/c) * gf0.
        """
        # Compute gradient of data term w.r.t. output weights
        try:
            grad_loss = torch.autograd.grad(
                loss, self.param_groups[0]["params"], create_graph=True, retain_graph=True
            )
            grad_flat = torch.cat([g.view(-1) for g in grad_loss])
        except Exception as e:
            logger.error(f"Failed to compute gradients in _transform_param2q: {e}")
            grad_flat = torch.zeros_like(params)

        # Dphima(u) = (dphi(|u|) - 1) * sign(u)
        dphi_term = _dphi(torch.abs(params), self.th, self.gamma) - 1.0
        gf0 = self.alpha * torch.sign(params) * dphi_term + grad_flat

        # Override on nonzero entries: gf0_i = -alpha * sign(u_i)
        nonzero_mask = torch.abs(params) > 0
        gf0 = torch.where(nonzero_mask, -self.alpha * torch.sign(params), gf0)

        # q = u - (1/c) * gf0
        q = params - (1.0 / self.c) * gf0
        return q

    # def _transform_param2q0(self, params, loss):
    #     """Transform parameters to auxiliary variable q.
        
    #     Args:
    #         params: the outer weights
    #         loss: the loss value (should be a scalar tensor with grad_fn)
    #     Returns:
    #         q: preimage of u w.r.t. the proximal operator
    #     """
    #     # Check if loss requires gradients
    #     try:
    #         grad_loss = torch.autograd.grad(loss, self.param_groups[0]["params"], create_graph=True, retain_graph=True)
    #         grad_flat = torch.cat([g.view(-1) for g in grad_loss])
    #         logger.debug(f"Gradient computed successfully, norm: {torch.norm(grad_flat).item():.6e}")
    #     except Exception as e:
    #         logger.error(f"Failed to compute gradients: {e}")
    #         return params.clone()
            
    #     return params - (1 / self.c) * (grad_flat + self.alpha * torch.sign(params) * (_ddphi(torch.abs(params), self.th, self.gamma)))
    
    def _update_parameters(self, u_flat):
        """Helper function to update model parameters from flattened tensor."""
        start = 0
        for p in self.param_groups[0]["params"]:
            numel = p.numel()
            p.data.copy_(u_flat[start:start + numel].view(p.shape))
            start += numel
    
    # Hidden activations are set directly by callers: optimizer.hidden_activations = S.detach()

    def step(self, closure):
        """Perform a single optimization step using semismooth Newton method with damping.
        
        Args:
            closure (callable): A closure that reevaluates the model and returns the loss.
                
        Returns:
            loss: The loss value.
        """
        # logger.info(f"SSN Step {self.n_iters}")
        
        # Get current loss and parameters
        loss = closure()      
        params = torch.cat([p.view(-1) for p in self.param_groups[0]["params"]])
        # logger.debug(f"Initial loss: {loss.item():.6e}, penalty: {(self.alpha * torch.sum(self._phi(torch.abs(params)))).item():.6e}")
        
        # Initialize q
        q = self._transform_param2q(params, loss)  # Pass loss instead of loss for gradient computation
        # Compute gradient and Hessian
        Gq = self._Gradient(q, params, loss)    
        DG = self._Hessian(q, params, loss)
        grad_norm = torch.norm(Gq).item()
        I = torch.eye(params.shape[0], device=params.device, dtype=params.dtype)
        theta0_base = 1.0 / (1e-12 * torch.norm(DG, p=float('inf')).item())
        
        # Calculte the initial gradient decent and update the parameter q and u. 
        try:
            # Log the linear system we're trying to solve
            system_matrix = DG + (1/theta0_base) * I
            # logger.debug(f"Linear system matrix condition: {torch.linalg.cond(system_matrix).item():.2e}")
            # logger.debug(f"RHS (Gq) norm: {torch.norm(Gq).item():.6e}")
            dq = -torch.linalg.solve(system_matrix, Gq)
            logger.debug(f"Solution (dq) norm: {torch.norm(dq).item():.6e}")
        except Exception as e:
            logger.error(f"Linear solve failed: {e}")
            logger.error(f"Matrix condition was: {torch.linalg.cond(system_matrix).item():.2e}")
            return loss
        
        # Update outer weights and evaluate new loss
        qnew = q + dq
        unew = _compute_prox(qnew, self.alpha / self.c)
        self._update_parameters(unew)
        loss_new = closure()
        
        # update the damping parameter
        dq_norm = torch.norm(dq).item()
        Gq_norm = torch.norm(Gq).item()
        if Gq_norm < 1e-20:  # avoid division by zero
            Gq_norm = 1e-20
        theta0 = max(dq_norm / Gq_norm, theta0_base)
        logger.debug(f"Initial theta0: {theta0:.6e}, step norm: {dq_norm:.6e}")
        theta = theta0

        # Initialize the Line search with backtracking (matching MATLAB exactly)
        iter_ls = 0
        eps_machine = torch.finfo(loss.dtype).eps
        tolerance = 1 + 1000 * eps_machine
        if not torch.isnan(loss_new) and loss_new <= loss.item():
            self.n_iters += 1
            return loss
        
        # Limit line search iterations to prevent infinite loops
        max_ls_iter = 100
        
        while (torch.isnan(loss_new) or loss_new > tolerance * loss) and iter_ls < max_ls_iter:
            # ------------------------------------------------------------------
            # SAFETY GUARD 1: stop damping if theta is already tiny. Line Search stops
            # ------------------------------------------------------------------
            if theta < 1e-15:
                logger.warning(f"Theta reached {theta:.1e}. Breaking line search loop.")
                break
            # ------------------------------------------------------------------
            if iter_ls < 10 or iter_ls % 20 == 0:  # Only log occasionally
                logger.debug(f"Damping step {iter_ls}: theta={theta:.2e}, loss_new={loss_new.item():.10e}")
            # ------------------------------------------------------------------
            # SAFETY GUARD 2: if the tentative step exploded, back-track harder
            # ------------------------------------------------------------------
            if torch.isnan(unew).any() or torch.isinf(unew).any():
                logger.warning("unew contains Inf/NaN – increasing damping (theta *= 4) and retrying")
                theta = theta * 4.0  # undo last division so next loop will divide again
                iter_ls += 1
                break
            # ------------------------------------------------------------------   

            # Check for NaN early and break
            if torch.isnan(loss_new) and iter_ls > 50:
                logger.warning("Persistent NaN in line search, stopping")
                break

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
            
            # Track consecutive failures to prevent infinite loops
            self.consecutive_failures += 1
            if self.consecutive_failures >= 5:
                logger.error(f"SSN optimizer has failed {self.consecutive_failures} consecutive times. This may indicate:")
                logger.error("  1. The problem is not suitable for SSN optimization")
                logger.error("  2. The regularization parameters (alpha, gamma) are inappropriate")
                logger.error("  3. The gradient computation is incorrect")
                logger.error("  Consider switching to Adam optimizer or adjusting parameters.")
                raise RuntimeError("SSN optimizer failed repeatedly - stopping to prevent infinite loop")
            
            # Restore original parameters
            self._update_parameters(params)
            return loss

        self.consecutive_failures = 0
        self.n_iters += 1
        return loss
    


        
            





