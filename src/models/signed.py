#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:30:52 2024

@author: chaoruiz
"""

from typing import Optional, Tuple
import logging
import torch
from ..SSN import SSN
from ..SSN.penalty import _phi
from ..SSN.prox import _phi_prox
from ..eval import data_loss_terms
from .net import ShallowNetwork

logger = logging.getLogger(__name__)


class SignedModel:
    """
    shallow neural networks
    """
    def __init__(
        self,
        alpha: float,
        gamma: float,
        activation: torch.nn.Module = torch.relu,
        power: float = 2.1,
        lr: float = 1.0,
        loss_weights: Tuple[float, float] = (1.0, 1.0),
        th: float = 0.5,
        verbose: bool = True,
        method: Optional[str] = None,
        max_ls_iter: int = 500,
        tolerance_ls: float = 1.0 + 1e-8,
        tolerance_grad: float = 0.0,
        sigmamax: float = 10.0,
        ) -> None:
        """
        Args:
            activation: Callable[[], Tensor]
            power: Power for activation function (default: 1.0)
            loss_weights: Weights for (value_loss, gradient_loss) (default: (1.0, 1.0))
            th: Interpolation parameter between L1 (th=0) and non-convex (th=1) (default: 0.5)
            verbose: Whether to print training progress to terminal (default: True)
        """
        # optimizer parameters
        self.activation = activation
        self.power = power
        self.q = 2.0 / (self.power + 1.0)
        self.alpha = alpha
        self.gamma = gamma
        self.lr = lr
        # loss parameters
        self.loss_weights = loss_weights
        self.th = th
        # verbose
        self.verbose = verbose
        # SSN solver settings (default to today's literals)
        self.method = method
        self.max_ls_iter = max_ls_iter
        self.tolerance_ls = tolerance_ls
        self.tolerance_grad = tolerance_grad
        self.sigmamax = sigmamax
        self.input_dim: Optional[int] = None
        
        # Initialize training components
        self.net = None
        self.optimizer = None  # Will store the actual optimizer instance (standard PyTorch convention)
        self.loss_history = {'train_loss': [], 'val_loss': [], 'value_loss': [], 'grad_loss': []}
        self.last_fit_summary = {}

        # High-level setup is logged by PDAP after the data has been prepared.
    
    def _create_network(self, inner_weights: torch.Tensor, inner_bias: torch.Tensor, outer_weights: torch.Tensor) -> None:
        """Build the shallow network on a fixed support (frozen inner weights).

        Args:
            inner_weights: hidden weights (frozen — the support is fixed by PDAP)
            inner_bias: hidden bias (frozen)
            outer_weights: output weights (the only trainable parameters)
        """
        input_dim = int(inner_weights.shape[1])
        n = inner_weights.shape[0]
        if self.verbose:
            logger.debug("Network support  atoms=%d", n)

        self.net = ShallowNetwork(
            [input_dim, n, 1],
            self.activation,
            p=self.power,
            inner_weights=inner_weights, inner_bias=inner_bias, outer_weights=outer_weights
        )

        # Freeze hidden layer so only the output weights are trainable.
        self.net.hidden.weight.requires_grad = False
        self.net.hidden.bias.requires_grad = False

    def _setup_optimizer(self) -> None:
        """Build the SSN solver over the output weights (the only free params)."""
        method = self.method or "levenberg_marquardt"
        self.optimizer = SSN(
            [self.net.output.weight], alpha=self.alpha, gamma=self.gamma, th=self.th,
            lr=self.lr, power=self.power, method=method,
            max_ls_iter=self.max_ls_iter, tolerance_ls=self.tolerance_ls,
            tolerance_grad=self.tolerance_grad, sigmamax=self.sigmamax,
        )
        if self.verbose:
            logger.debug(
                "Output-weight solver  method=%s  alpha=%.2e  gamma=%.2e  penalty_mix=%.2f  lr=%.2g",
                method, self.alpha, self.gamma, self.th, self.lr,
            )
    
    def _compute_loss(self, x_input: torch.Tensor, target_v: torch.Tensor, target_dv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the combined MSE loss for value and gradient matching. The loss always use
        the full batch
        Args:
            x_input: Input coordinates
            target_v: Target values
            target_dv: Target gradients
        Returns:
            Tuple of (total_loss, value_loss, grad_loss)
        """
        # Create a fresh tensor that requires gradients for gradient computation
        # Remark: for SSN, the graph actually doesn't need to be created here.
        x = x_input.clone().detach().requires_grad_(True)
        pred_v = self.net(x)
        pred_dv = torch.autograd.grad(
            outputs=pred_v, 
            inputs=x, 
            grad_outputs=torch.ones_like(pred_v), 
            create_graph=True, 
            retain_graph=True
        )[0]
        
        # Data-fidelity term (Nx = N*d normalization) lives in src.eval.
        data_loss, value_loss, grad_loss = data_loss_terms(
            pred_v, pred_dv, target_v, target_dv, self.loss_weights
        )

        # Full objective: data loss + regularization
        # Matches MATLAB SSN.m line 34: obj = @(u) F.F(Sred*u - ref) + alpha*sum(phi.phi(computeNorm(u, NQ)))
        abs_u = torch.abs(self.net.output.weight)
        reg_arg = abs_u ** self.q if self.q != 1.0 else abs_u
        total_loss = data_loss + self.alpha * torch.sum(_phi(reg_arg, self.th, self.gamma))
 
        return total_loss, value_loss, grad_loss

    def predict_tensors(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (V, dV) at x as detached tensors, V:(N,1), dV:(N,d).

        Uniform with SemiconcaveModel.predict_tensors so the PDAP loop and the
        insertion strategy can read residuals from any model the same way.
        """
        if self.net is None:
            raise RuntimeError("network not created yet; call set_atoms() first")
        x_req = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            V = self.net(x_req)
            dV = torch.autograd.grad(V.sum(), x_req, create_graph=False)[0]
        return V.detach(), dV.detach()

    # ------------------------------------------------------------------ #
    # Uniform atom interface (matches SemiconcaveModel) for the PDAP loop.
    # Canonical representation: W (n,d), b (n,), c (n,).  The signed network
    # stores c as the (1,n) output weight; set/get reshape across that.
    # ------------------------------------------------------------------ #
    def set_atoms(self, W: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> None:
        """Establish the current support by (re)building the network."""
        W = torch.as_tensor(W, dtype=torch.float64)
        b = torch.as_tensor(b, dtype=torch.float64).reshape(-1)
        c = torch.as_tensor(c, dtype=torch.float64).reshape(1, -1)
        self._create_network(W, b, c)

    def get_atoms(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read the current support as (W (n,d), b (n,), c (n,))."""
        if self.net is None:
            raise RuntimeError("network not created yet; call set_atoms() first")
        W = self.net.hidden.weight.detach().clone()
        b = self.net.hidden.bias.detach().clone()
        c = self.net.output.weight.detach().reshape(-1).clone()
        return W, b, c

    @property
    def n_neurons(self) -> int:
        return 0 if self.net is None else int(self.net.hidden.weight.shape[0])

    def warm_start(
        self,
        W_new: torch.Tensor,
        b_new: torch.Tensor,
        data_train: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        use_sphere: bool = True,
        verbose: bool = True,
    ) -> torch.Tensor:
        """Coordinate-descent initial outer weights for new neurons, shape (n_new,).

        1D proximal step along the combined new-atom direction against the current
        model residual (MATLAB PDAPmultisemidiscrete.m:104-147).  Residual is taken
        from the current network (or -target when no atoms exist yet).
        """
        X_train, V_train, dV_train = data_train
        N, d = X_train.shape[0], X_train.shape[1]
        Nx = N * d
        p = self.power
        w1, w2 = self.loss_weights
        alpha, th, gamma = self.alpha, self.th, self.gamma
        n_new = W_new.shape[0]
        if n_new == 0:
            return torch.zeros(0, dtype=torch.float64)

        X_det = X_train.detach()
        if self.net is not None:
            pred_v, pred_dv = self.predict_tensors(X_det)
            res_val = (pred_v - V_train).detach().reshape(-1)
            res_grad = (pred_dv - dV_train).detach().reshape(-1)
        else:
            res_val = -V_train.detach().reshape(-1)
            res_grad = -dV_train.detach().reshape(-1)

        with torch.no_grad():
            pre = X_det @ W_new.T + b_new
            act = self.activation(pre)
            S_val_new = act ** p
            if use_sphere:
                act_deriv = (pre > 0).double()
            else:
                pre_tmp = pre.detach().requires_grad_(True)
                with torch.enable_grad():
                    act_tmp = self.activation(pre_tmp)
                    act_deriv = torch.autograd.grad(act_tmp.sum(), pre_tmp, create_graph=False)[0].detach()
            dS_dz = act_deriv if p == 1.0 else p * act ** (p - 1) * act_deriv
            dS_dx = dS_dz.unsqueeze(2) * W_new.unsqueeze(0)
            S_grad_new = dS_dx.permute(0, 2, 1).reshape(-1, n_new)

            profiles = (w1 / Nx) * (S_val_new.T @ res_val) + (w2 / Nx) * (S_grad_new.T @ res_grad)
            abs_profiles = profiles.abs()
            best_idx = int(abs_profiles.argmax().item())
            eps_sqrt = float(torch.finfo(torch.float64).eps) ** 0.5
            safe_abs = abs_profiles.clamp_min(1e-30)
            coeff = -eps_sqrt * profiles / safe_abs
            coeff[best_idx] = -profiles[best_idx] / safe_abs[best_idx]

            Kvhat_val = S_val_new @ coeff
            Kvhat_grad = S_grad_new @ coeff
            phat = float((w1 / Nx) * Kvhat_val.dot(res_val) + (w2 / Nx) * Kvhat_grad.dot(res_grad))
            what = float((w1 / Nx) * Kvhat_val.dot(Kvhat_val) + (w2 / Nx) * Kvhat_grad.dot(Kvhat_grad))

            if phat <= -alpha and what > 1e-30:
                tau = _phi_prox(alpha / what, -phat / what, th, gamma, q=2.0 / (p + 1.0))
            else:
                tau = 0.0
            W_outer_new = tau * coeff
            if verbose:
                logger.debug(
                    "Warm start        initialized %d new output weights  max |weight|=%.2e",
                    n_new, float(W_outer_new.abs().max().item()),
                )
                logger.debug(
                    "Warm-start details: descent_score=%.2e curvature=%.2e step=%.4e min_abs_weight=%.2e",
                    -phat, what, tau, float(W_outer_new.abs().min().item()),
                )
        return W_outer_new.reshape(-1)

    def fit_outer_weights(
        self,
        data_train: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        data_valid: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        iterations: int = 20,
        display_every: int = 2,
    ) -> bool:
        """Run the SSN outer-weight solve on the already-set atoms."""
        return self._fit_loop(data_train, data_valid, iterations, display_every)

    def _fit_loop(
        self,
        data_train: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        data_valid: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        iterations: int,
        display_every: int,
    ) -> bool:
        """SSN outer-weight solve on the already-built network (the body of train)."""
        # Initialize
        train_x_tensor, train_v_tensor, train_dv_tensor = data_train
        valid_x_tensor, valid_v_tensor, valid_dv_tensor = data_valid
        self._setup_optimizer()
        if self.verbose:
            logger.debug("Output-weight training")
            logger.debug("  %-6s %-14s %-14s", "step", "train loss", "val loss")
        
        # Reset loss history
        self.loss_history = {'train_loss': [], 'val_loss': [], 'value_loss': [], 'grad_loss': []}
        # Track and save the running best model by training loss.
        best_train_loss = float('inf')
        best_epoch = -1
        # Ensure best_state is always defined (e.g. iterations == 0)
        best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}

        # Define closure() for SSN
        def closure():
            total_loss, _, _ = self._compute_loss(
                train_x_tensor, train_v_tensor, train_dv_tensor
            )
            return total_loss

        # Training loop
        consecutive_failed_ssn_steps = 0
        max_consecutive_failed_ssn_steps = 3
        successful_steps = 0
        for epoch in range(iterations):
            self.optimizer.zero_grad()
            if isinstance(self.optimizer, SSN):
                x_detach = train_x_tensor.detach()
                S = self.net.forward_network_matrix(x_detach).detach()       # (N, n)
                S_grad = self.net.forward_gradient_kernel(x_detach).detach() # (N*d, n)
                N = S.shape[0]
                d = train_x_tensor.shape[1]
                Nx = N * d
                # Data-loss Hessian: d²l/du² = (1/Nx)*(w1*S'S + w2*S_grad'S_grad)
                # Matches MATLAB: ddF = 1/Nx where Nx = numel(xhat) = d*N
                self.optimizer.data_hessian = (1.0 / Nx) * (
                    self.loss_weights[0] * (S.T @ S)
                    + self.loss_weights[1] * (S_grad.T @ S_grad)
                )
                loss = self.optimizer.step(closure)
            else:
                total_loss, _, _ = self._compute_loss(train_x_tensor, train_v_tensor, train_dv_tensor)
                total_loss.backward()
                self.optimizer.step()
                loss = total_loss
                successful_steps += 1

            # Save running best model
            # self.net.eval() # set to evaluation mode (not be needed since always full patch)
            train_loss, _, _ = self._compute_loss(train_x_tensor, train_v_tensor, train_dv_tensor)
            if train_loss.item() < best_train_loss:
                best_train_loss = train_loss.item()
                best_epoch = epoch
                # Snapshot a real checkpoint (not just a view into current params)
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}

            # Early-stop SSN training if line search fails repeatedly
            if isinstance(self.optimizer, SSN):
                step_success = bool(getattr(self.optimizer, "last_step_success", True))
                if not step_success:
                    consecutive_failed_ssn_steps += 1
                    if self.verbose:
                        max_ls_iter = self.optimizer.param_groups[0]['max_ls_iter']
                        logger.debug(
                            f"SSN step rejected (consecutive={consecutive_failed_ssn_steps}/{max_consecutive_failed_ssn_steps}, max_ls_iter={max_ls_iter})"
                        )
                else:
                    consecutive_failed_ssn_steps = 0
                    successful_steps += 1

                if consecutive_failed_ssn_steps >= max_consecutive_failed_ssn_steps:
                    if self.verbose:
                        logger.info(
                            f"SSN early stop: line search stalled {max_consecutive_failed_ssn_steps} epochs; "
                            f"restored best epoch {best_epoch} (train={best_train_loss:.3e})."
                        )
                    break
            
            # log the loss
            if epoch % display_every == 0:
                val_loss, val_value_loss, val_grad_loss = self._compute_loss(
                    valid_x_tensor, valid_v_tensor, valid_dv_tensor
                )
                if self.verbose:
                    logger.debug("  %6d %-14.6e %-14.6e", epoch, loss.item(), val_loss.item())
                self.loss_history['train_loss'].append(loss.item())
                self.loss_history['val_loss'].append(val_loss.item())
                self.loss_history['value_loss'].append(val_value_loss.item())
                self.loss_history['grad_loss'].append(val_grad_loss.item())

        
        # Restore the best model before returning and report best loss
        self.net.load_state_dict(best_state)
        self.last_fit_summary = {
            "best_step": best_epoch,
            "best_train_loss": best_train_loss,
            "successful_steps": successful_steps,
        }
        if self.verbose:
            logger.debug(
                "Output-weight solve complete  best_inner_step=%d  best_train_loss=%.6e",
                best_epoch, best_train_loss,
            )

        return successful_steps > 0
