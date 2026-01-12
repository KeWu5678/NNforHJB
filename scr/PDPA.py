#!/usr/bin/env python3
"""
Simple training logger for VDP_Testing notebook
"""

from __future__ import annotations
from typing import Any, Dict, List, Callable, Tuple
from .model import model
import numpy as np
from loguru import logger
import torch

class PDPA:
    def __init__(
        self,
        data: dict,
        alpha: float,
        gamma: float, 
        power: float, 
        activation: torch.nn.Module = torch.relu,
        loss_weights: Tuple[float, float] = (1.0, 1.0),
        lr: Tuple[float, float] = (1e-3, 1.0),
        optimizer: Tuple[str, str] = ('SGD', 'SSN'),
        verbose = True
        ) -> None:

        # defineretrain-iteration histories
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.inner_weights: List[Dict[str, torch.Tensor]] = []
        self.outer_weights: List[torch.Tensor] = []

        # define the models
        self.alpha= alpha

        self.model1 = model(
            alpha=self.alpha,
            gamma=gamma, 
            optimizer=optimizer[0],
            activation=activation,
            power=power,
            lr=lr[0],
            loss_weights=loss_weights,
            verbose=verbose,
            train_outerweights=False
        )

        self.model2 = model(
            alpha=self.alpha,
            gamma=gamma,
            optimizer=optimizer[1], 
            activation=activation,
            power=power,
            lr=lr[1],
            loss_weights=loss_weights,
            verbose=verbose,
            train_outerweights=True
        )

        self.data_train, self.data_valid = self.model1._prepare_data(data)
        
    @staticmethod
    def sample_uniform_sphere_points(N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample N points uniformly on the unit sphere (in R^3) and return a (x, y) + z
        representation as torch tensors.
        
        Args:
            N: Number of points to sample
            
        Returns:
            Tuple (a, b) where:
            - a has shape (N, 2) and corresponds to (x, y)
            - b has shape (N,) and corresponds to z
        """

        # Hard-set float64 on CPU to match training data + network (.double()).
        v = torch.randn(N, 3, dtype=torch.float64, device="cpu")
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)

        a = v[:, :2].contiguous()  # (N,2) -> (x,y)
        b = v[:, 2].contiguous()   # (N,)  -> z
        return a, b


    def insertion(
        self,
        data_train: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        model: Any,
        N: int,
        alpha: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Insert N neurons that satisfies the NOC of the Primal-Dual-Proximal-Algorithm
        Args:
            data_train: 
            model: trained model in the last step. 
            N: number of neurons to be inserted
        return:
            weights: np.ndarray, shape = (N, 2) for PyTorch linear layer
            bias: np.ndarray, shape = (N,) for PyTorch linear layer
        """
        
        X_train, V_train, dV_train = data_train
        p = model.power
        
        # this computes the dual profile function with a, b (the weighs and bias).
        def profile(
            a: torch.Tensor, 
            b: torch.Tensor,
        ) -> torch.Tensor:
            """
            Dual profile function evaluated on the whole training set.
            Args:
                a: shape (2,)
                b: scalar tensor (or shape (1,))
            Returns:
                A scalar tensor.
            """
            model.net.eval()
            activation_fn = model.net.activation

            """ We need gradients w.r.t. inputs to get pred_dv. However, when optimizing
            over (a, b) we want pred_v/pred_dv treated as constants, so we detach
            them to avoid autograd graph reuse issues under LBFGS closures."""
            X_for_grad = X_train.detach().clone().requires_grad_(True)
            pred_v = model.net(X_for_grad)  # (K, 1) typically
            pred_dv = torch.autograd.grad(
                outputs=pred_v.sum(),
                inputs=X_for_grad,
                create_graph=False,
                retain_graph=False,
            )[0]  # (K, 2)
            pred_v = pred_v.detach()
            pred_dv = pred_dv.detach()
            X = X_train.detach()

            pre = X @ a.reshape(-1) + b.reshape(())  # (K,)
            act = activation_fn(pre).reshape(-1, 1)  # (K, 1)

            profile_v = (act ** p) * (pred_v - V_train)  # (K, 1)
            # (pred_dv - dV_train) · a  -> (K, 1)
            dv_dot = (pred_dv - dV_train) @ a.reshape(-1, 1)
            profile_dv = p * (act ** (p - 1)) * dv_dot  # (K, 1)

            return (model.loss_weights[0] * profile_v + model.loss_weights[1] * profile_dv).abs().sum()
    
        def local_maxiamize(
            a: torch.Tensor,
            b: torch.Tensor,
            profile: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            steps: int = 1000,
            lr: float = 1e-3,
            eps: float = 1e-12,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Locally maximize profile(a, b) over (a, b) constrained to the unit sphere in R^3.

            Args:
                a: tensor with shape (2,)
                b: tensor scalar or shape (1,)
                profile: callable returning a scalar tensor (or a tensor reducible to scalar)
                steps: number of optimizer steps
                lr: learning rate
                eps: numerical stability for normalization

            Returns:
                best_val: scalar tensor, the maximal value of profile(a, b) achieved.
                best_a: tensor with shape (2,)
                best_b: scalar tensor
            """
            # Optimize an unconstrained 3D vector and evaluate the profile on its
            # projection onto the unit sphere (keeps the LBFGS line search stable).
            w0 = torch.cat([a.reshape(-1), b.reshape(-1)]).detach()  # (3,)
            w = w0.clone().requires_grad_(True)

            opt = torch.optim.LBFGS(
                [w],
                lr=lr,
                max_iter=steps, # maximal iteration specified here
                line_search_fn="strong_wolfe",
            )

            best_val = torch.tensor(-float("inf"), device=w.device, dtype=w.dtype)

            def closure() -> torch.Tensor:
                opt.zero_grad()
                w_sphere = w / w.norm().clamp_min(eps)
                # LBFGS minimizes; we want to maximize the profile.
                obj = profile(w_sphere[:2], w_sphere[2])
                (-obj).backward()
                return -obj

            opt.step(closure)

            # profile() internally calls autograd.grad(...), so we must keep autograd enabled.
            # We still detach outputs since we don't need gradients here.
            with torch.enable_grad():
                w_sphere = w / w.norm().clamp_min(eps)
                val = profile(w_sphere[:2], w_sphere[2])
            best_val = val.detach().clone()
            best_a = w_sphere[:2].detach().clone()
            best_b = w_sphere[2].detach().clone()

            return best_val, best_a, best_b

        
        # Step 1: Sample candidate points
        a_t, b_t = self.sample_uniform_sphere_points(N)  # a_t: (N,2) tensor, b_t: (N,) tensor

        # Step 2: Insert the neuron by the algorithm in the paper. 
        accepted_a: List[torch.Tensor] = []
        accepted_b: List[torch.Tensor] = []
        for a0, b0 in zip(a_t, b_t):
            best_val, best_a, best_b = local_maxiamize(a0, b0, profile)
            if float(best_val.item()) > alpha:
                accepted_a.append(best_a)
                accepted_b.append(best_b)

        tried = int(N)
        accepted = int(len(accepted_a))
        logger.info(f"insertion - tried {tried} candidates, accepted {accepted} (alpha={alpha})")

        if len(accepted_a) == 0:
            return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)

        W = torch.stack(accepted_a, dim=0).detach().cpu().numpy()
        b = torch.stack(accepted_b, dim=0).detach().cpu().numpy()
        return W, b


    @staticmethod
    def prune_small_weights(
        weights: torch.Tensor,
        biases: torch.Tensor,
        outer_weights: torch.Tensor,
        threshold: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prune neurons with small outer weights (torch-in / torch-out)."""

        if not isinstance(weights, torch.Tensor) or not isinstance(biases, torch.Tensor) or not isinstance(outer_weights, torch.Tensor):
            raise TypeError("prune_small_weights expects torch.Tensor inputs for weights, biases, and outer_weights")

        # We only need values, not gradients.
        w = weights.detach()
        b = biases.detach()
        ow = outer_weights.detach()

        # logger.info(
        #     f"prune_small_weights - weights: {tuple(w.shape)}, biases: {tuple(b.shape)}, outer_weights: {tuple(ow.shape)}"
        # )

        # Flatten to 1D to build a per-neuron mask (handles (1, n), (n,), etc.)
        ow_flat = ow.reshape(-1)

        keep_mask = ow_flat.abs() >= threshold
        pruned_count = int((~keep_mask).sum().item())
        if pruned_count > 0:
            logger.info(f"Pruned {pruned_count} neurons with small weights")

        w_pruned = w[keep_mask]
        b_pruned = b[keep_mask]

        # Always return outer weights with shape (1, n_pruned) so it can be passed directly to model.train(...)
        ow_pruned = ow_flat[keep_mask].reshape(1, -1)

        # logger.info(
        #     f"After pruning - weights: {tuple(w_pruned.shape)}, biases: {tuple(b_pruned.shape)}, outer_weights: {tuple(ow_pruned.shape)}"
        # )

        return w_pruned, b_pruned, ow_pruned


    def retrain(
        self,
        num_iterations: int,
        num_insertion: int,
        threshold: float,
        verbose: bool = True
    ) -> int:
        
        # Track best model across all iterations
        best_iteration = 0
        best_val_loss = float('inf')
        W_hidden, b_hidden = self.sample_uniform_sphere_points(num_insertion)

        # Training loop
        for i in range(num_iterations):
            if verbose: logger.info(f"Iteration {i + 1} - Starting...")
            self.model1.train(
                self.data_train, 
                self.data_valid, 
                inner_weights=W_hidden, 
                inner_bias=b_hidden, 
                iterations = 1000, 
                display_every = 100
                )
            state_1 = self.model1.net.state_dict()
            W_hidden = state_1["hidden.weight"].detach().to(device="cpu").clone()
            b_hidden = state_1["hidden.bias"].detach().to(device="cpu").clone()
            W_out = state_1["output.weight"].detach().to(device="cpu").clone()
            self.model2.train(
                self.data_train, 
                self.data_valid, 
                inner_weights=W_hidden, 
                inner_bias=b_hidden, 
                outer_weights=W_out, 
                iterations = 1000, 
                display_every = 100
                )
            
            # Count and prune small weights
            state_2 = self.model2.net.state_dict()
            # Use torch-native ops to count small weights, then convert to int for logging
            small_mask = (state_2['output.weight'].abs().flatten() < threshold)
            small_count = int(small_mask.sum().item())
            if verbose: logger.info(f"Small weights count: {small_count}, Pruning...")
            
            # Prune neurons based on the trained outer weights from model_2
            W_hidden, b_hidden, W_out_pruned = self.prune_small_weights(
                W_hidden, 
                b_hidden,
                state_2["output.weight"],
                threshold,
            )

            # Evaluate losses for the *pruned* network snapshot we will store.
            # Note: we must keep autograd enabled because `_compute_loss` uses `autograd.grad`.
            self.model2._create_network(inner_weights=W_hidden, inner_bias=b_hidden, outer_weights=W_out_pruned)
            train_loss_t, _, _ = self.model2._compute_loss(*self.data_train)
            val_loss_t, _, _ = self.model2._compute_loss(*self.data_valid)
            train_loss_f = float(train_loss_t.detach().cpu().item())
            val_loss_f = float(val_loss_t.detach().cpu().item())

            # Persist per-iteration snapshots on `self`.
            self.train_loss.append(train_loss_f)
            self.val_loss.append(val_loss_f)
            self.inner_weights.append(
                {
                    "weight": W_hidden.detach().to(device="cpu").clone(),
                    "bias": b_hidden.detach().to(device="cpu").clone(),
                }
            )
            self.outer_weights.append(W_out_pruned.detach().to(device="cpu").clone())
            
            # logger.info(f"Recording...")
            if self.model2.config['best_val_loss'] < best_val_loss:
                best_val_loss = self.model2.config['best_val_loss']
                best_iteration = i
                if verbose: logger.info(f"New best model found at iteration {i} with validation loss: {best_val_loss:.6f}")
            
            # Insert neurons and train
            W_to_insert, b_to_insert = self.insertion(self.data_train, self.model1, num_insertion, self.alpha)
            W_to_insert_t = torch.as_tensor(W_to_insert, dtype=W_hidden.dtype, device=W_hidden.device)
            b_to_insert_t = torch.as_tensor(b_to_insert, dtype=b_hidden.dtype, device=b_hidden.device)
            W_hidden = torch.cat((W_hidden, W_to_insert_t), dim=0)
            b_hidden = torch.cat((b_hidden, b_to_insert_t), dim=0)

        return best_iteration