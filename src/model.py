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
from .ssn import SSN
from .ssn_tr import SSN_TR
from .net import ShallowNetwork
from .utils import _phi


class model:
    """
    Van der Pol model trainer using shallow neural networks.
    
    This class handles the complete training pipeline including:
    - Data preparation and splitting
    - Network initialization
    - Training with various optimizers (Adam, SSN)
    - Logging and checkpointing
    """
    
    def __init__(
        self, 
        training_percentage=0.9,
        optimizer = "Adam", activation=torch.tanh, power=2.1, regularization=None, lr = 0.01,
        loss_weights=(1.0, 1.0),  th=0.5, 
        verbose=True,
        train_outerweights = False
        ):
        """
        Initialize the VDP model.
        
        Args:
            activation: Activation function (default: torch.tanh)
            power: Power for activation function (default: 1.0)
            regularization: Regularization parameters (gamma, alpha)
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
        self.regularization = regularization
        self.alpha = regularization[1]
        self.gamma = regularization[0]
        self.lr = lr
        # loss parameters
        self.loss_weights = loss_weights
        self.th = th
        # verbose
        self.verbose = verbose
        # network parameters
        self.train_outerweights = train_outerweights
        
        # Initialize training components
        self.net = None
        self.optimizer = None  # Will store the actual optimizer instance (standard PyTorch convention)
        self.loss_history = {'train_loss': [], 'val_loss': [], 'value_loss': [], 'grad_loss': []}
        
        # Configure loguru logger
        self._configure_logger()
        self.config = None
        
    def _configure_logger(self):
        """Configure loguru logger for training output."""
        logger.remove()  # Remove default handler
        
        if self.verbose:
            # Add terminal output only if verbose is True
            logger.add(
                sys.stdout,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                level="INFO"
            )
        
        # Always log to file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        log_history_dir = os.path.join(current_dir, "..", "log_history")
        os.makedirs(log_history_dir, exist_ok=True)
        log_file = os.path.join(log_history_dir, "training.log")
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB"
        )
        
        if self.verbose:
            logger.info("Model initialized")
    
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
        
        if self.verbose:
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
    
    def _create_network(self, inner_weights=None, inner_bias=None, outer_weights=None):
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
                [2, n, 1], 
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
                [2, n, 1], 
                self.activation, 
                p=self.power, 
                inner_weights=inner_weights, inner_bias=inner_bias, outer_weights=outer_weights
            )
            # Freeze hidden layer so only output weights are trainable
            self.net.hidden.weight.requires_grad = False
            self.net.hidden.bias.requires_grad = False
        
    
    def _setup_optimizer(self):
        """Setup the optimizer based on optimizer type."""
        if self.optimizer_type in ["SSN", "SSN_TR"]:
            if self.train_outerweights == True:
                output_params = [self.net.output.weight]
                optimizer_class = SSN if self.optimizer_type == "SSN" else SSN_TR
                self.optimizer = optimizer_class(output_params, alpha=self.alpha, gamma=self.gamma, th=self.th, lr=self.lr)
                if self.verbose:
                    logger.info(f"Using {self.optimizer_type} optimizer with alpha={self.alpha}, gamma={self.gamma}, th={self.th}")
            else:
                # SSN optimizer is for outerweights only
                logger.debug(f"SSN optimizer is for outerweights only")
        else:
            if self.train_outerweights == True:
                output_params = [self.net.output.weight]
            else:
                output_params = self.net.parameters()

            optimizer_class = getattr(torch.optim, self.optimizer_type, None)
            if optimizer_class is None:
                print(f"Warning: {self.optimizer_type} not found, using SGD")
                optimizer_class = torch.optim.SGD

            self.optimizer = optimizer_class(output_params, lr=self.lr)
            if self.verbose:
                logger.info(f"Using {self.optimizer_type} optimizer with lr={self.lr}")
    
    
    def _compute_loss(self, x_input, target_v, target_dv):
        """
        Compute the combined MSE loss for value and gradient matching.
        
        Args:
            x_input: Input coordinates
            target_v: Target values
            target_dv: Target gradients
            
        Returns:
            Tuple of (total_loss, value_loss, grad_loss)
        """
        # Create a fresh tensor that requires gradients for gradient computation
        x = x_input.clone().detach().requires_grad_(True)
         
        # Forward pass to get predicted value
        pred_v = self.net(x)
        
        # Compute predicted gradients using autograd
        pred_dv = torch.autograd.grad(
            outputs=pred_v, 
            inputs=x, 
            grad_outputs=torch.ones_like(pred_v), 
            create_graph=True, 
            retain_graph=True
        )[0]
        
        value_loss = torch.mean((pred_v - target_v) ** 2)   # MSE loss for value prediction
        grad_loss = torch.mean((pred_dv - target_dv) ** 2)  # MSE loss for gradient prediction
        data_loss = self.loss_weights[0] * value_loss + self.loss_weights[1] * grad_loss

        if isinstance(self.optimizer, (SSN, SSN_TR)):
            total_loss = data_loss
        else:
            total_loss = data_loss + self.alpha * torch.sum(_phi(torch.abs(self.net.output.weight), self.th, self.gamma))
 
        return total_loss, value_loss, grad_loss

        

    def train(
        self, 
        data_train, data_valid, 
        inner_weights=None, inner_bias=None, outer_weights=None,
        iterations=5000, batch_size=1620, display_every=1000):
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
        if self.verbose:
            logger.info("Starting network training session")
        
        # pass data
        train_x_tensor, train_v_tensor, train_dv_tensor = data_train
        valid_x_tensor, valid_v_tensor, valid_dv_tensor = data_valid
        # Create network
        self._create_network(inner_weights, inner_bias, outer_weights)
        # Setup optimizer
        self._setup_optimizer()
        
        # Training setup
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_save_path = os.path.join(root_dir, "train_history")
        os.makedirs(model_save_path, exist_ok=True)
        if self.verbose:
            logger.info(f"Training hyperparameters: iterations={iterations}, batch_size={batch_size}, display_every={display_every}")
            logger.info(f"Loss weights: value={self.loss_weights[0]}, gradient={self.loss_weights[1]}")
        
        # Reset loss history
        self.loss_history = {'train_loss': [], 'val_loss': [], 'value_loss': [], 'grad_loss': []}
        best_val_loss = float('inf')    # Track and save running best model by validation loss
        best_epoch = -1
        best_model_path = os.path.join(model_save_path, "model_best.pt")

        # Define closure for SSN
        def closure():
            if isinstance(self.optimizer, (SSN, SSN_TR)):
                with torch.no_grad():
                    _, hidden_activations = self.net.forward_with_hidden(train_x_tensor.detach())
                self.optimizer.hidden_activations = hidden_activations.detach()
            # Compute loss
            total_loss, _, _ = self._compute_loss(
                train_x_tensor, train_v_tensor, train_dv_tensor
            )
            return total_loss

        # Training loop
        for epoch in range(iterations):
            # self.net.train() #not needed, no dropout or batchnorm
            self.optimizer.zero_grad()
            
            if isinstance(self.optimizer, (SSN, SSN_TR)):
                loss = self.optimizer.step(closure)
            else:
                total_loss, _, _ = self._compute_loss(train_x_tensor, train_v_tensor, train_dv_tensor)
                total_loss.backward()
                self.optimizer.step()
                loss = total_loss
            
            # Validation loss for logging
            if epoch % display_every == 0:
                self.net.eval() # set to evaluation mode (though may not be needed)
                val_loss, val_value_loss, val_grad_loss = self._compute_loss(valid_x_tensor, valid_v_tensor, valid_dv_tensor)
                
                if self.verbose:
                    logger.info(f"Epoch {epoch}: Train Loss = {loss.item():.6f}, "f"Val Loss = {val_loss.item():.6f}")
                
                self.loss_history['train_loss'].append(loss.item())
                self.loss_history['val_loss'].append(val_loss.item())
                self.loss_history['value_loss'].append(val_value_loss.item())
                self.loss_history['grad_loss'].append(val_grad_loss.item())
                # Save running best model
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_epoch = epoch
                    torch.save(self.net.state_dict(), best_model_path)
        
        # Restore the best model before returning and report best loss
        self.net.load_state_dict(torch.load(best_model_path))
        logger.info(f"Best validation loss: {best_val_loss:.6f}. Restored best model from {best_model_path}")
        
        # Prepare return artifacts
        self.config = {
            'optimizer': self.optimizer_type,
            'activation': getattr(self.activation, '__name__', str(self.activation)),
            'power': self.power,
            'regularization': {'gamma': self.gamma, 'alpha': self.alpha, 'th': self.th},
            'loss_weights': {'value': self.loss_weights[0], 'grad': self.loss_weights[1]},
            'training_percentage': self.training_percentage,
            'train_outerweights': self.train_outerweights,
            'iterations': iterations,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'loss_history': self.loss_history['train_loss'],
            'val_history': self.loss_history['val_loss'],
        }


