import torch
from torch.optim import Optimizer
import numpy as np
from loguru import logger

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
        lr=1.0,
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
        
        # SSN constant
        self.c = 1 + alpha * gamma
        
        self.q = None  # preimage of outer weights w.r.t. the proximal operator
        self.n_iters = 0    # Number of SSN iterations
        self.consecutive_failures = 0
        self.hidden_activations = None  # Store S matrix for Hessian
        
    def _phi(self, t):
        """
        Non-convex penalty function phi(t) for gamma > 0.
        th = 0: the full nonconvex penalty
        th = 1: the L1 penalty
        """
        th = self.th
        if th == 1:
            return t
        else:
            gam = self.gamma / (1 - th)  # = 2*gamma
            return th * t + (1 - th) * torch.log(1 + gam * t) / gam
    
    def _dphi(self, t):
        """Derivative of penalty function."""
        th = self.th
        if th == 1:
            return torch.ones_like(t)
        else:
            gam = self.gamma / (1 - th)
            return th + (1 - th) / (1 + gam * t)
    
    def _ddphi(self, t):
        """Second derivative of penalty function."""
        th = self.th
        if th == 1:
            return torch.zeros_like(t)
        else:
            # Safeguard against th very close to 1
            if abs(1 - th) < 1e-8:
                return torch.zeros_like(t)
            gam = self.gamma / (1 - th)
            return -(1 - th) * gam / ((1 + gam * t) ** 2)
    
    def _compute_prox(self, v, mu):
        """Compute the soft thresholding operator for scalar sparsity.
        Args:
            v: input vector
            mu: regularization parameter
        Returns:
            vprox: proximal operator result
        """
        normsv = torch.abs(v)

        # Safeguard against division by zero
        eps = torch.finfo(v.dtype).eps
        normsv_safe = torch.clamp(normsv, min=(mu + eps) * eps)
        
        # Apply scalar soft shrinkage operator
        # vprox = max(0, 1 - mu / |v|) * v for each element
        shrinkage_factor = torch.clamp(1 - mu / normsv_safe, min=0)
        vprox = shrinkage_factor * v
        
        return vprox
    
    def _compute_dprox(self, v, mu):
        """Compute the derivative of the proximal operator for scalar sparsity.
        
        Args:
            v: the vector-valued parameter
            mu: proximal parameter
            
        Returns:
            DP: derivative matrix of the proximal operator (diagonal)
        """
        # Ensure input is real
        assert torch.is_floating_point(v), "Input must be real-valued"
        
        # For scalar sparsity (N=1), each element is its own group
        # normsv = abs(v) for each element
        normsv = torch.abs(v)
        
        # Safeguard against division by zero
        eps = torch.finfo(v.dtype).eps
        normsv_safe = torch.clamp(normsv, min=(mu + eps) * eps)
        
        # First term: max(0, 1 - mu / normsv_safe)
        diagonal_term = torch.clamp(1 - mu / normsv_safe, min=0)
        
        # Second term: (normsv >= mu) * mu / (normsv_safe^3) * v^2
        mask = normsv >= mu
        outer_product_term = mask.float() * mu / (normsv_safe ** 3) * (v ** 2)

        # Create diagonal matrix
        DP = torch.diag(diagonal_term + outer_product_term)
        
        return DP
    
    def _Gradient(self, q, params, loss):
        # - params: the outer weights
        # - q: preimage of u w.r.t. the proximal operator
        D_nonconvex = torch.sign(params) * (self._ddphi(torch.abs(params)) - 1)
        try:
            grad_loss = torch.autograd.grad(loss, self.param_groups[0]["params"], create_graph=True, retain_graph=True)
            grad_flat = torch.cat([g.view(-1) for g in grad_loss])
        except Exception as e:
            logger.error(f"Failed to compute gradients in _Gradient: {e}")
            grad_flat = torch.zeros_like(params)
        return self.c * (q - params) + self.alpha * D_nonconvex + grad_flat

    def _Hessian(self, q, params, loss):
        DD_nonconvex = self._ddphi(torch.abs(params))
        DPc = self._compute_dprox(q, self.alpha / self.c)
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
        """Transform parameters to auxiliary variable q.
        
        Args:
            params: the outer weights
            loss: the loss value (should be a scalar tensor with grad_fn)
        Returns:
            q: preimage of u w.r.t. the proximal operator
        """
        # Check if loss requires gradients
        if not loss.requires_grad:
            logger.warning("Loss tensor does not require gradients - SSN may not work properly")
            return params.clone()
            
        try:
            grad_loss = torch.autograd.grad(loss, self.param_groups[0]["params"], create_graph=True, retain_graph=True)
            grad_flat = torch.cat([g.view(-1) for g in grad_loss])
            logger.debug(f"Gradient computed successfully, norm: {torch.norm(grad_flat).item():.6e}")
        except Exception as e:
            logger.error(f"Failed to compute gradients: {e}")
            return params.clone()
            
        return params - (1 / self.c) * (self.alpha * torch.sign(params) * (self._ddphi(torch.abs(params)) - 1) + grad_flat)
    
    def _update_parameters(self, u_flat):
        """Helper function to update model parameters from flattened tensor."""
        start = 0
        for p in self.param_groups[0]["params"]:
            numel = p.numel()
            p.data.copy_(u_flat[start:start + numel].view(p.shape))
            start += numel
    
    def set_hidden_activations(self, hidden_activations):
        """Set the hidden activations matrix S for Gauss-Newton Hessian computation."""
        self.hidden_activations = hidden_activations.detach()  # Don't need gradients for S

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
        logger.debug(f"Initial loss: {loss.item():.6e}, penalty: {(self.alpha * torch.sum(self._phi(torch.abs(params)))).item():.6e}")
        
        # Initialize q
        q = self._transform_param2q(params, loss)  # Pass loss instead of loss for gradient computation
        # Compute gradient and Hessian
        Gq = self._Gradient(q, params, loss)    
        DG = self._Hessian(q, params, loss)
        grad_norm = torch.norm(Gq).item()
        # logger.info(f"Gradient norm: {grad_norm:.6e}")
        
        # if grad_norm < 1e-12:
        #     logger.warning("Gradient norm is extremely small - may indicate convergence or gradient computation issues")
        #     return loss
        # # Log detailed Hessian information
        # hessian_min = DG.min().item()
        # hessian_max = DG.max().item()
        # hessian_det = torch.det(DG).item() if DG.shape[0] <= 10 else "N/A (too large)"
        # hessian_cond = torch.linalg.cond(DG).item()
        
        # logger.info(f"Hessian analysis:")
        # logger.info(f"  Shape: {DG.shape}")
        # logger.info(f"  Min eigenvalue: {hessian_min:.6e}")
        # logger.info(f"  Max eigenvalue: {hessian_max:.6e}")
        # logger.info(f"  Condition number: {hessian_cond:.2e}")
        # logger.info(f"  Determinant: {hessian_det}")
        # logger.info(f"  Contains NaN: {torch.isnan(DG).any().item()}")
        # logger.info(f"  Contains Inf: {torch.isinf(DG).any().item()}")

        # if hessian_cond > 1e12:
        #     logger.warning(f"Hessian is ill-conditioned: {hessian_cond:.2e}, adding regularization")
        #     # Add stronger regularization for ill-conditioned case
        #     DG = DG + 1e-6 * torch.eye(DG.shape[0], device=DG.device, dtype=DG.dtype)
        #     hessian_cond_new = torch.linalg.cond(DG).item()
        #     logger.info(f"After regularization, condition number: {hessian_cond_new:.2e}")
        
        # if torch.isnan(DG).any() or torch.isinf(DG).any():
        #     logger.error("Hessian contains NaN or Inf values - cannot proceed")
        #     return loss
        
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
        unew = self._compute_prox(qnew, self.alpha / self.c)
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
        # logger.info(f"Line search start: loss={loss.item():.6e}, loss_new={loss_new.item():.6e}")
        # logger.info(f"Tolerance factor: {tolerance:.6e}")
        # logger.info(f"Line search condition: loss_new > tolerance * loss = {tolerance * loss.item():.6e}")
        # Add early termination if we already have a good step
        if not torch.isnan(loss_new) and loss_new <= loss.item():
            # logger.info("Step already provides descent, no line search needed")
            # descent = loss.item() - loss_new.item()
            # support = torch.sum(torch.abs(unew) > 1e-8).item()
            # logger.info(f"Descent: {descent:.6e}, Support: {support}")
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
                unew = self._compute_prox(qnew, self.alpha / self.c)
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
        
        # Successful step - parameters are already updated
        # descent = loss.item() - loss_new.item()
        # support = torch.sum(torch.abs(unew) > 1e-8).item()
        
        # logger.info(f"Line search successful: {iter_ls} iterations, descent: {descent:.6e}")
        # logger.info(f"Support: {support}, damping ratio: {theta/theta0:.2e}")
        
        # Reset failure counter on success
        self.consecutive_failures = 0
        self.n_iters += 1
        return loss
    


        
            





