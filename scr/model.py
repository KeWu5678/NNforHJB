#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:30:52 2024

@author: chaoruiz
"""

from typing import Optional, Tuple
import torch
import os
from loguru import logger
from .ssn import SSN
from .ssn_tr import SSN_TR
from .net import ShallowNetwork
from .utils import _phi


class model:
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
        th: float = 0.01,
        training_percentage: float = 0.9,  
        verbose: bool = True,
        train_outerweights: bool = False
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
        self.input_dim: Optional[int] = None
        
        # Initialize training components
        self.net = None
        self.optimizer = None  # Will store the actual optimizer instance (standard PyTorch convention)
        self.loss_history = {'train_loss': [], 'val_loss': [], 'value_loss': [], 'grad_loss': []}
        self.config = None
        
        # Log initialization (logger should be configured at application level via setup_logging())
        if self.verbose:
            logger.info("Model initialized")
    
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
        
        if self.verbose:
            logger.info(f"Training set: {len(train_x)} samples, Validation set: {len(valid_x)} samples")
        
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

        if self.train_outerweights == False:
            if inner_weights is None:
                # Default case - will be handled by ShallowNetwork, create a network with 30 neurons
                n = 30
            else:
                # Number of neurons is the first dimension for PyTorch
                n = inner_weights.shape[0]
                if self.verbose:
                    logger.info(f"Creating network with {n} neurons")
        
        # Create the shallow network
            self.net = ShallowNetwork(
                [input_dim, n, 1], 
                self.activation,
                p=self.power, 
                inner_weights=inner_weights, inner_bias=inner_bias, outer_weights=outer_weights
            )
        else:
            if inner_weights is None:
                n = 30 # same as above
            else:
                n = inner_weights.shape[0]
            self.net = ShallowNetwork(
                [input_dim, n, 1], 
                self.activation, 
                p=self.power, 
                inner_weights=inner_weights, inner_bias=inner_bias, outer_weights=outer_weights
            )
            # Freeze hidden layer so only output weights are trainable
            self.net.hidden.weight.requires_grad = False
            self.net.hidden.bias.requires_grad = False
        
    
    def _setup_optimizer(self) -> None:
        """Setup the optimizer based on optimizer type."""
        if self.optimizer_type in ["SSN", "SSN_TR"]:
            if self.train_outerweights == True:
                output_params = [self.net.output.weight]
                optimizer_class = SSN if self.optimizer_type == "SSN" else SSN_TR
                self.optimizer = optimizer_class(output_params, alpha=self.alpha, gamma=self.gamma, th=self.th, lr=self.lr)
                if self.verbose:
                    logger.info(f"Using {self.optimizer_type} optimizer with alpha={self.alpha}, gamma={self.gamma}, th={self.th}, lr ={self.lr}")
            else:
                # SSN optimizer is for outerweights only
                logger.debug(f"SSN optimizer is for outerweights only")
        else:
            output_params = [self.net.output.weight] if self.train_outerweights else self.net.parameters()
            optimizer_class = getattr(torch.optim, self.optimizer_type, None)

            if optimizer_class is None:
                logger.info(f"Warning: {self.optimizer_type} not found, using SGD")
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
        
        value_loss = torch.mean((pred_v - target_v) ** 2)   
        grad_loss = torch.mean((pred_dv - target_dv) ** 2) 
        data_loss = self.loss_weights[0] * value_loss + self.loss_weights[1] * grad_loss

        if isinstance(self.optimizer, (SSN, SSN_TR)):
            total_loss = data_loss # SSN uses line search hence need true loss.
        else:
            # For other method without line search, normal loss (gradient graph) is calculated
            total_loss = data_loss + self.alpha * torch.sum(_phi(torch.abs(self.net.output.weight), self.th, self.gamma))
 
        return total_loss, value_loss, grad_loss

    def train(
        self, 
        data_train: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        data_valid: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
        inner_weights: Optional[torch.Tensor] = None, 
        inner_bias: Optional[torch.Tensor] = None, 
        outer_weights: Optional[torch.Tensor] = None,
        iterations: int = 5000, 
        display_every: int = 1000
    ) -> None:
        """
        Train the model on the provided data.
        
        Args:
            data: Dictionary with keys 'x', 'v', 'dv'
            inner_weights: Pre-defined inner weights (optional)
            inner_bias: Pre-defined inner bias (optional)
            iterations: Number of training iterations (default: 5000)
            batch_size: Batch size (default: 1620)
            display_every: Display frequency (default: 1000)
            
        Returns:
            Tuple of (model_wrapper, weight, bias, output_weight)
        """

        # Initialize
        train_x_tensor, train_v_tensor, train_dv_tensor = data_train
        valid_x_tensor, valid_v_tensor, valid_dv_tensor = data_valid
        self._create_network(inner_weights, inner_bias, outer_weights)
        self._setup_optimizer()
        logger.info("Starting network training session") if self.verbose else None
        
        # Reset loss history
        self.loss_history = {'train_loss': [], 'val_loss': [], 'value_loss': [], 'grad_loss': []}
        # Track and save running best model by validation loss
        best_val_loss = float('inf')
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
        for epoch in range(iterations):
            self.optimizer.zero_grad()
            if isinstance(self.optimizer, (SSN, SSN_TR)):
                with torch.no_grad():
                    self.optimizer.hidden_activations = self.net.forward_network_matrix(train_x_tensor.detach()).detach()
                loss = self.optimizer.step(closure)
            else:
                total_loss, _, _ = self._compute_loss(train_x_tensor, train_v_tensor, train_dv_tensor)
                total_loss.backward()
                self.optimizer.step()
                loss = total_loss

            # Save running best model
            # self.net.eval() # set to evaluation mode (not be needed since always full patch)
            train_loss, train_value_loss, train_grad_loss = self._compute_loss(train_x_tensor, train_v_tensor, train_dv_tensor)
            if train_loss.item() < best_train_loss:
                best_train_loss = train_loss.item()
                best_epoch = epoch
                # Snapshot a real checkpoint (not just a view into current params)
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}

            val_loss, val_value_loss, val_grad_loss = self._compute_loss(valid_x_tensor, valid_v_tensor, valid_dv_tensor)
            # if val_loss.item() < best_val_loss:
            #     best_val_loss = val_loss.item()
            #     best_epoch = epoch
            #     # Snapshot a real checkpoint (not just a view into current params)
            #     best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}

            # Early-stop SSN training if line search fails repeatedly
            if isinstance(self.optimizer, (SSN, SSN_TR)):
                step_success = bool(getattr(self.optimizer, "last_step_success", True))
                if not step_success:
                    consecutive_failed_ssn_steps += 1
                    if self.verbose:
                        max_ls_iter = self.optimizer.param_groups[0]['max_ls_iter']
                        logger.warning(
                            f"SSN step rejected (consecutive={consecutive_failed_ssn_steps}/{max_consecutive_failed_ssn_steps}, max_ls_iter={max_ls_iter})"
                        )
                else:
                    consecutive_failed_ssn_steps = 0

                if consecutive_failed_ssn_steps >= max_consecutive_failed_ssn_steps:
                    if self.verbose:
                        logger.warning(
                            f"Early stopping: SSN line search failed for {max_consecutive_failed_ssn_steps} consecutive epochs. "
                            f"Restoring best model at epoch {best_epoch} (val={best_val_loss:.6f})."
                        )
                    break
            
            # log the loss
            if epoch % display_every == 0:
                if self.verbose:
                    logger.info(f"Epoch {epoch}: Train Loss = {loss.item():.6f}, "f"Val Loss = {val_loss.item():.6f}")
                self.loss_history['train_loss'].append(loss.item())
                self.loss_history['val_loss'].append(val_loss.item())
                self.loss_history['value_loss'].append(val_value_loss.item())
                self.loss_history['grad_loss'].append(val_grad_loss.item())

        
        # Restore the best model before returning and report best loss
        self.net.load_state_dict(best_state)
        if self.verbose:
            logger.info(f"Best validation loss: {best_val_loss:.6f} at iteration {best_epoch}")

        
