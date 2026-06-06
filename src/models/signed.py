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
        optimizer: str = "SGD", 
        activation: torch.nn.Module = torch.relu, 
        power: float = 2.1, 
        lr: float = 1.0,
        loss_weights: Tuple[float, float] = (1.0, 1.0), 
        th: float = 0.5,
        training_percentage: float = 0.9,
        verbose: bool = True,
        train_outerweights: bool = False,
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
            training_percentage: Fraction of data for training (default: 0.9)
            th: Interpolation parameter between L1 (th=0) and non-convex (th=1) (default: 0.5)
            verbose: Whether to print training progress to terminal (default: True)
        """
        # data processing parameters
        self.training_percentage = training_percentage
        # optimizer parameters
        self.optimizer_type = optimizer
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
        # network parameters
        self.train_outerweights = train_outerweights
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
    
    def _prepare_data(self, data: dict) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Prepare and split data into training and validation sets.
        
        Args:
            data: Dictionary with keys 'x', 'v', 'dv'
            
        Returns:
            Tuple of (train_tensors, valid_tensors) where each is (x, v, dv)
        """
        # Handle different data formats
        if isinstance(data, dict):
            # Already in dictionary format
            ob_x, ob_v, ob_dv = data["x"], data["v"], data["dv"]
        else:   
            raise ValueError(
                "Data must be provided as a dictionary with keys 'x', 'v', 'dv'. "
                "Please convert your structured numpy array to dictionary format:\n"
                "  data = {'x': x_array, 'v': v_array, 'dv': dv_array}"
            )
        
        # Record input dimension for network creation
        if hasattr(ob_x, "ndim") and ob_x.ndim < 2:
            raise ValueError(f"Expected 'x' to be 2D with shape (N, d); got shape {getattr(ob_x, 'shape', None)}")
        self.input_dim = int(ob_x.shape[1])

        # Split data into training and validation sets
        split_idx = int(len(ob_x) * self.training_percentage)
        train_x, valid_x = ob_x[:split_idx], ob_x[split_idx:]
        train_v, valid_v = ob_v[:split_idx], ob_v[split_idx:]
        train_dv, valid_dv = ob_dv[:split_idx], ob_dv[split_idx:]
        
        # Convert to tensors
        train_tensors = (
            torch.tensor(train_x, dtype=torch.float64, requires_grad=True),
            torch.tensor(train_v.reshape(-1, 1), dtype=torch.float64),
            torch.tensor(train_dv, dtype=torch.float64)
        )
        
        valid_tensors = (
            torch.tensor(valid_x, dtype=torch.float64, requires_grad=True),
            torch.tensor(valid_v.reshape(-1, 1), dtype=torch.float64),
            torch.tensor(valid_dv, dtype=torch.float64)
        )
        
        return train_tensors, valid_tensors
    
    def _create_network(self, inner_weights: Optional[torch.Tensor] = None, inner_bias: Optional[torch.Tensor] = None, outer_weights: Optional[torch.Tensor] = None) -> None:
        """
        Create the shallow network.
        
        Args:
        For full training:
            inner_weights: Pre-defined inner weights (optional)
            inner_bias: Pre-defined inner bias (optional)
            outer_weights: Pre-defined outer weights (optional)
        for outer weights training:
            inner_weights: Pre-defined inner weights (frozen)
            inner_bias: Pre-defined inner bias (frozen)
            outer_weights: Pre-defined outer weights (trainable)
        """
        if inner_weights is not None:
            input_dim = int(inner_weights.shape[1])
        else:
            if self.input_dim is None:
                raise ValueError("input_dim is not set. Call _prepare_data() before training or pass inner_weights.")
            input_dim = int(self.input_dim)

        if inner_weights is None:
            # Default case - no atoms provided, create a network with 30 neurons
            n = 30
        else:
            # Number of neurons is the first dimension for PyTorch
            n = inner_weights.shape[0]
            if self.verbose:
                logger.debug("Network support  atoms=%d", n)

        # Create the shallow network
        self.net = ShallowNetwork(
            [input_dim, n, 1],
            self.activation,
            p=self.power,
            inner_weights=inner_weights, inner_bias=inner_bias, outer_weights=outer_weights
        )

        if self.train_outerweights:
            # Freeze hidden layer so only output weights are trainable
            self.net.hidden.weight.requires_grad = False
            self.net.hidden.bias.requires_grad = False

    
    def _setup_optimizer(self) -> None:
        """Setup the optimizer based on optimizer type."""
        if self.optimizer_type in ["SSN", "SSN_TR"]:
            if self.train_outerweights:
                output_params = [self.net.output.weight]
                # SSN_TR folded into SSN as the trust-region (Steihaug-CG) method;
                # an explicit configured method overrides the optimizer_type shorthand.
                method = self.method or (
                    "steihaug_cg" if self.optimizer_type == "SSN_TR" else "levenberg_marquardt"
                )
                self.optimizer = SSN(
                    output_params, alpha=self.alpha, gamma=self.gamma, th=self.th,
                    lr=self.lr, power=self.power, method=method,
                    max_ls_iter=self.max_ls_iter, tolerance_ls=self.tolerance_ls,
                    tolerance_grad=self.tolerance_grad, sigmamax=self.sigmamax,
                )
                if self.verbose:
                    logger.debug(
                        "Output-weight solver  method=%s  alpha=%.2e  gamma=%.2e  penalty_mix=%.2f  lr=%.2g",
                        method, self.alpha, self.gamma, self.th, self.lr,
                    )
            else:
                # SSN optimizer is for outerweights only
                logger.debug("SSN optimizer is for outerweights only")
        else:
            output_params = [self.net.output.weight] if self.train_outerweights else self.net.parameters()
            optimizer_class = getattr(torch.optim, self.optimizer_type, None)

            if optimizer_class is None:
                logger.warning("Optimizer %r is not available; using SGD instead", self.optimizer_type)
                optimizer_class = torch.optim.SGD
            # Always instantiate the optimizer (including fallback SGD)
            self.optimizer = optimizer_class(output_params, lr=self.lr)
    
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
        
        # Match MATLAB: Nx = numel(p.xhat) = d * N_points
        # (setup_problem_NN_2d_from_xhat.m Tracking function, line ~196)
        N = x_input.shape[0]
        d = x_input.shape[1]
        Nx = N * d
        value_loss = torch.sum((pred_v - target_v) ** 2) / (2 * Nx)
        grad_loss = torch.sum((pred_dv - target_dv) ** 2) / (2 * Nx)
        data_loss = self.loss_weights[0] * value_loss + self.loss_weights[1] * grad_loss

        # Full objective: data loss + regularization
        # Matches MATLAB SSN.m line 34: obj = @(u) F.F(Sred*u - ref) + alpha*sum(phi.phi(computeNorm(u, NQ)))
        abs_u = torch.abs(self.net.output.weight)
        reg_arg = abs_u ** self.q if self.q != 1.0 else abs_u
        total_loss = data_loss + self.alpha * torch.sum(_phi(reg_arg, self.th, self.gamma))
 
        return total_loss, value_loss, grad_loss

    @torch.no_grad()
    def _compute_relative_errors(
        self, x_input: torch.Tensor, target_v: torch.Tensor, target_dv: torch.Tensor
    ) -> Tuple[float, float, float]:
        """Compute relative L2, gradient, and H1 errors.

        Returns:
            (err_l2, err_grad, err_h1) as plain floats.
        """
        x = x_input.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            pred_v = self.net(x)
            pred_dv = torch.autograd.grad(
                outputs=pred_v, inputs=x,
                grad_outputs=torch.ones_like(pred_v),
                create_graph=False, retain_graph=False,
            )[0]

        v_diff_sq = torch.sum((pred_v - target_v) ** 2)
        v_true_sq = torch.sum(target_v ** 2)
        dv_diff_sq = torch.sum((pred_dv - target_dv) ** 2)
        dv_true_sq = torch.sum(target_dv ** 2)

        err_l2 = torch.sqrt(v_diff_sq / v_true_sq.clamp_min(1e-30))
        err_grad = torch.sqrt(dv_diff_sq / dv_true_sq.clamp_min(1e-30))
        err_h1 = torch.sqrt((v_diff_sq + dv_diff_sq) / (v_true_sq + dv_true_sq).clamp_min(1e-30))
        return float(err_l2.item()), float(err_grad.item()), float(err_h1.item())

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
        """Run the SSN outer-weight solve on the already-set atoms (see train())."""
        return self._fit_loop(data_train, data_valid, iterations, display_every)

    def train(
        self,
        data_train: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        data_valid: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
        inner_weights: Optional[torch.Tensor] = None, 
        inner_bias: Optional[torch.Tensor] = None, 
        outer_weights: Optional[torch.Tensor] = None,
        iterations: int = 5000, 
        display_every: int = 1000
    ) -> bool:
        """
        Train the model on the provided data.

        Args:
            data_train: Training tensors (x, v, dv)
            data_valid: Validation tensors (x, v, dv)
            inner_weights: Pre-defined inner weights (optional)
            inner_bias: Pre-defined inner bias (optional)
            outer_weights: Pre-defined outer weights (optional)
            iterations: Number of training iterations (default: 5000)
            display_every: Display frequency (default: 1000)

        Returns:
            True if training made progress (at least one successful SSN step),
            False if SSN line search failed on every step.
        """

        self._create_network(inner_weights, inner_bias, outer_weights)
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
            train_loss, train_value_loss, train_grad_loss = self._compute_loss(train_x_tensor, train_v_tensor, train_dv_tensor)
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
