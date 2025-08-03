#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:30:52 2024

@author: chaoruiz
"""

import os

import numpy as np
import torch
import sys
import os
from loguru import logger
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssn import SSN
from net import ShallowNetwork


class model:
    """
    Van der Pol model trainer using shallow neural networks.
    
    This class handles the complete training pipeline including:
    - Data preparation and splitting
    - Network initialization
    - Training with various optimizers (Adam, SSN)
    - Logging and checkpointing
    """
    
    def __init__(self, data, activation=torch.tanh, power=1.0, regularization=None, optimizer = "Adam",
                 loss_weights=(1.0, 1.0), training_percentage=0.9, th=0.5):
        """
        Initialize the VDP model.
        
        Args:
            activation: Activation function (default: torch.tanh)
            power: Power for activation function (default: 1.0)
            regularization: Regularization parameters (gamma, alpha)
            loss_weights: Weights for (value_loss, gradient_loss) (default: (1.0, 1.0))
            training_percentage: Fraction of data for training (default: 0.9)
            th: Interpolation parameter between L1 (th=0) and non-convex (th=1) (default: 0.5)
        """
        self.activation = activation
        self.power = power
        self.regularization = regularization
        self.loss_weights = loss_weights
        self.training_percentage = training_percentage
        self.optimizer_type = optimizer  # Store type temporarily, will be replaced by actual optimizer
        self.alpha = regularization[1]
        self.gamma = regularization[0]
        
        # Initialize training components
        self.net = None
        self.optimizer = None  # Will store the actual optimizer instance (standard PyTorch convention)
        self.loss_history = {'train_loss': [], 'val_loss': [], 'value_loss': [], 'grad_loss': []}
        self.th = th
        self.data = data
        
        # Configure loguru logger
        self._configure_logger()
        
    def _configure_logger(self):
        """Configure loguru logger for training output."""
        logger.remove()  # Remove default handler
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO"
        )
        
        # Also log to file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        log_file = os.path.join(current_dir, "training.log")
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB"
        )
        
        logger.info("VDPModel initialized")
    
    def _prepare_data(self, data):
        """
        Prepare and split data into training and validation sets.
        
        Args:
            data: Either a dictionary with keys 'x', 'v', 'dv' OR a structured numpy array
            
        Returns:
            Tuple of (train_tensors, valid_tensors) where each is (x, v, dv)
        """
        # Handle different data formats
        if isinstance(data, dict):
            # Already in dictionary format
            ob_x, ob_v, ob_dv = data["x"], data["v"], data["dv"]
        else:
            # Assume structured numpy array format: each element is (x, dv, v)
            x_list, dv_list, v_list = [], [], []
            for item in data:
                x_coords, dv_vals, v_val = item
                x_list.append(x_coords)
                dv_list.append(dv_vals)
                v_list.append(v_val)
            
            ob_x = np.array(x_list, dtype=np.float64)
            ob_v = np.array(v_list, dtype=np.float64)
            ob_dv = np.array(dv_list, dtype=np.float64)
        
        # Split data into training and validation sets
        num_samples = len(ob_x)
        split_idx = int(num_samples * self.training_percentage)
        train_indices = np.arange(split_idx)
        valid_indices = np.arange(split_idx, num_samples)
        
        # Create training and validation datasets
        train_x = ob_x[train_indices].astype(np.float64)
        train_v = ob_v[train_indices].astype(np.float64)
        train_dv = ob_dv[train_indices].astype(np.float64)
        
        valid_x = ob_x[valid_indices].astype(np.float64)
        valid_v = ob_v[valid_indices].astype(np.float64)
        valid_dv = ob_dv[valid_indices].astype(np.float64)
        
        logger.info(f"Training set: {len(train_x)} samples, Validation set: {len(valid_x)} samples")
        
        # Optional: Log data statistics for debugging
        logger.info(f"Data ranges - x: [{train_x.min():.2f}, {train_x.max():.2f}], "
                   f"v: [{train_v.min():.2f}, {train_v.max():.2f}], "
                   f"dv: [{train_dv.min():.2f}, {train_dv.max():.2f}]")
        
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
    
    def _create_network(self, inner_weights=None, inner_bias=None):
        """
        Create the shallow network.
        
        Args:
            inner_weights: Pre-defined inner weights (optional)
            inner_bias: Pre-defined inner bias (optional)
        """
        if inner_weights is None:
            # Default case - will be handled by ShallowNetwork
            n = 0
        else:
            # Number of neurons is the first dimension for PyTorch
            n = inner_weights.shape[0]
            logger.info(f"Creating network with {n} neurons")
        
        # Create the shallow network
        self.net = ShallowNetwork(
            [2, n, 1], self.activation, "zeros", p=self.power, 
            inner_weights=inner_weights, inner_bias=inner_bias, 
        )
    
    def _setup_optimizer(self):
        """Setup the optimizer based on optimizer type."""
        if self.optimizer_type == "SSN":
            # Use SSN optimizer for outer weights only
            output_params = [self.net.output.weight]
            self.optimizer = SSN(output_params, alpha=self.alpha, gamma=self.gamma)
            logger.info(f"Using SSN optimizer with alpha={self.alpha}, gamma={self.gamma}")
        elif self.optimizer_type == "Adam" or self.optimizer_type is None:
            # Use Adam optimizer for all parameters with lower learning rate
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001)
            logger.info("Using Adam optimizer with lr=0.0001")
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}. Use 'Adam' or 'SSN'.")
    
    def _phi(self, t):
        """
        Non-convex penalty function phi(t) for gamma > 0.
        th = 0: the L1 penalty
        th = 1: the full noncovex penalty
        """
        th = self.th
        gam = self.gamma / (1 - th)  # = 2*gamma
        return th * t + (1 - th) * torch.log(1 + gam * t) / gam
    
    def _compute_loss(self, x_input, target_v, target_dv):
        """
        Compute the combined MSE loss for value and gradient matching.
        
        Args:
            x_input: Input coordinates
            target_v: Target values
            target_dv: Target gradients
            
        Returns:
            Tuple of (total_loss, value_loss, grad_loss, hidden_activations)
        """
        # Create a fresh tensor that requires gradients for gradient computation
        x = x_input.clone().detach().requires_grad_(True)
        
        # Forward pass to get predicted value and hidden activations
        pred_v, hidden_activations = self.net.forward_with_hidden(x)
        
        # Compute predicted gradients using autograd
        pred_dv = torch.autograd.grad(
            outputs=pred_v, 
            inputs=x, 
            grad_outputs=torch.ones_like(pred_v), 
            create_graph=True, 
            retain_graph=True
        )[0]
        
        # MSE loss for value prediction
        value_loss = torch.mean((pred_v - target_v) ** 2)
        # MSE loss for gradient prediction
        grad_loss = torch.mean((pred_dv - target_dv) ** 2)
        total_loss = self.loss_weights[0] * value_loss + self.loss_weights[1] * grad_loss + self.alpha * torch.sum(self._phi(torch.abs(self.net.output.weight)))
 
        return total_loss, value_loss, grad_loss, hidden_activations
    
    def train(self, inner_weights=None, inner_bias=None, 
              iterations=20000, batch_size=1620, display_every=1000):
        """
        Train the model on the provided data.
        
        Args:
            data: Dictionary with keys 'x', 'v', 'dv'
            inner_weights: Pre-defined inner weights (optional)
            inner_bias: Pre-defined inner bias (optional)
            iterations: Number of training iterations (default: 20000)
            batch_size: Batch size (default: 1620)
            display_every: Display frequency (default: 1000)
            
        Returns:
            Tuple of (model_wrapper, weight, bias, output_weight)
        """
        logger.info("Starting network training session")
        
        # Prepare data
        train_tensors, valid_tensors = self._prepare_data(self.data)
        train_x_tensor, train_v_tensor, train_dv_tensor = train_tensors
        valid_x_tensor, valid_v_tensor, valid_dv_tensor = valid_tensors
        
        # Create network
        self._create_network(inner_weights, inner_bias)
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Training setup
        # Save to the root train_history folder
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_save_path = os.path.join(root_dir, "train_history")
        os.makedirs(model_save_path, exist_ok=True)
        logger.info(f"Training model, saving to {model_save_path}")
        
        logger.info(f"Training hyperparameters: iterations={iterations}, batch_size={batch_size}, display_every={display_every}")
        logger.info(f"Loss weights: value={self.loss_weights[0]}, gradient={self.loss_weights[1]}")
        
        # Reset loss history
        self.loss_history = {'train_loss': [], 'val_loss': [], 'value_loss': [], 'grad_loss': []}
        
        # Training loop
        for epoch in range(iterations):
            self.net.train()
            self.optimizer.zero_grad()
            
            # Define closure for optimizers that need it (like SSN)
            def closure():
                self.optimizer.zero_grad()
                total_loss, value_loss, grad_loss, hidden_activations = self._compute_loss(
                    train_x_tensor, train_v_tensor, train_dv_tensor
                )
                
                # For SSN optimizer, provide hidden activations for Gauss-Newton Hessian
                if isinstance(self.optimizer, SSN):
                    self.optimizer.set_hidden_activations(hidden_activations)
                    total_loss.backward(retain_graph=True)
                else:
                    total_loss.backward()
                return total_loss
            
            # Compute loss and backprop
            if isinstance(self.optimizer, SSN):
                loss = self.optimizer.step(closure)
            else:
                total_loss, value_loss, grad_loss, _ = self._compute_loss(
                    train_x_tensor, train_v_tensor, train_dv_tensor
                )
                total_loss.backward()
                self.optimizer.step()
                loss = total_loss
            
            # Validation loss and logging
            if epoch % display_every == 0:
                self.net.eval()
                # Note: We need gradients for gradient loss computation, so don't use no_grad()
                val_loss, val_value_loss, val_grad_loss, _ = self._compute_loss(
                    valid_x_tensor, valid_v_tensor, valid_dv_tensor
                )
                
                logger.info(f"Epoch {epoch}: Train Loss = {loss.item():.6f}, "
                           f"Val Loss = {val_loss.item():.6f}")
                
                self.loss_history['train_loss'].append(loss.item())
                self.loss_history['val_loss'].append(val_loss.item())
                self.loss_history['value_loss'].append(val_value_loss.item())
                self.loss_history['grad_loss'].append(val_grad_loss.item())
            
            # Save model checkpoint
            if epoch % 1000 == 0:
                torch.save(self.net.state_dict(), os.path.join(model_save_path, f"model_epoch_{epoch+1}.pt"))
        
        # Save final model
        torch.save(self.net.state_dict(), os.path.join(model_save_path, "model_final.pt"))
        logger.info(f"Final model saved to {os.path.join(model_save_path, 'model_final.pt')}")
        
        # Return results
        weight, bias = self.net.get_hidden_params()
        outer_weight = self.net.get_output_params()
        
        # Create a simple model wrapper to match original interface
        class ModelWrapper:
            def __init__(self, net, loss_history, loss_weights):
                self.net = net
                self.loss_weights = loss_weights
                self.losshistory = type('obj', (object,), {
                    'loss_train': loss_history['train_loss'],
                    'loss_test': loss_history['val_loss']
                })()
            
            def predict(self, x, operator=None):
                """Predict method for compatibility with greedy_insertion.py"""
                if isinstance(x, np.ndarray):
                    x = torch.tensor(x, dtype=torch.float64, requires_grad=True)
                
                self.net.eval()
                with torch.no_grad() if operator is None else torch.enable_grad():
                    pred = self.net(x)
                    if operator is not None:
                        # Apply custom operator (e.g., gradient computation)
                        return operator(x, pred).detach().cpu().numpy()
                    else:
                        return pred.detach().cpu().numpy()
        
        model_wrapper = ModelWrapper(self.net, self.loss_history, self.loss_weights)
        logger.info("Training completed successfully")
        
        return model_wrapper, weight.numpy(), bias.numpy(), outer_weight.numpy()


if __name__ == "__main__":
    # Example usage with the new class interface - simplified
    data = np.load("data_result/raw_data/VDP_beta_0.1_grid_30x30.npy", allow_pickle=True)
    
    # Use smaller network with better initialization
    n_neurons = 100
    weights = np.random.randn(n_neurons, 2) * 0.1
    bias = np.random.randn(n_neurons) * 0.1
    regularization = (1, 0.01)
    
    # Using the new class interface
    vdp_model = model(data, torch.relu, 2.0, regularization, optimizer='SSN', loss_weights=(1.0, 1.0))
    model_result, weight, bias, output_weight = vdp_model.train(
        inner_weights=weights, inner_bias=bias, 
        iterations=10000,
        display_every=1000
    )
    



