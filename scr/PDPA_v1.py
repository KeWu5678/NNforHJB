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


class PDPA_v1:
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

        # 1st stage training model
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

        # 2nd stage training model
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

        # store the activation function
        # (the passed `activation` is what the underlying `model` uses)
        self.activation_fn = activation

        # Data split is done when the model is initialized
        self.data_train, self.data_valid = self.model1._prepare_data(data)
        if self.model1.input_dim is None:
            raise ValueError("Could not infer input dimension from data. Ensure data['x'] has shape (N, d).")
        self.input_dim: int = int(self.model1.input_dim)
        
    def sample_uniform_sphere_points(self, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample N points uniformly on the unit sphere in R^(d+1), where d is the
        input dimension inferred from the data, and return an (a, b) representation
        where:
        - a has shape (N, d) and corresponds to the linear weights
        - b has shape (N,) and corresponds to the bias
        
        Args:
            N: Number of points to sample
            
        Returns:
            Tuple (a, b) where:
            - a has shape (N, d)
            - b has shape (N,)
        """

        # Hard-set float64 on CPU to match training data + network (.double()).
        d = int(self.input_dim)
        v = torch.randn(N, d + 1, dtype=torch.float64, device="cpu")
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)

        a = v[:, :d].contiguous()  # (N,d)
        b = v[:, d].contiguous()   # (N,)
        return a, b


    def insertion(
        self,
        data_train: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        N: int,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Insert N neurons that satisfies the NOC of the Primal-Dual-Proximal-Algorithm
        Args:
            data_train: 
            model: trained model in the last step. 
            N: number of neurons to be inserted
        return:
            weights: np.ndarray, shape = (N, d) for PyTorch linear layer
            bias: np.ndarray, shape = (N,) for PyTorch linear layer
        """
        
        X_train, V_train, dV_train = data_train
        p = self.model1.power
        alpha = self.alpha
        
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
            # 1-neuron candidate model defined by (a,b) with outer weight = 1.
            X_for_grad = X_train.detach().clone().requires_grad_(True)
            pre = X_for_grad @ a.reshape(-1) + b.reshape(())  # (K,)
            act = self.activation_fn(pre).reshape(-1, 1)  # (K, 1)

            # pred_v is the one-neuron output with outer weight = 1
            pred_v = act ** p  # (K, 1)
            # pred_dv is its input gradient; keep graph for gradients w.r.t (a,b)
            pred_dv = torch.autograd.grad(
                outputs=pred_v.sum(),
                inputs=X_for_grad,
                create_graph=True,
                retain_graph=True,
            )[0]  # (K, d)

            profile_v = (act ** p) * (pred_v - V_train)  # (K, 1)
            # (pred_dv - dV_train) · a  -> (K, 1)
            dv_dot = (pred_dv - dV_train) @ a.reshape(-1, 1)
            profile_dv = p * (act ** (p - 1)) * dv_dot  # (K, 1)

            return (self.model1.loss_weights[0] * profile_v + self.model1.loss_weights[1] * profile_dv).abs().sum()
    
        def local_maxiamize(
            a: torch.Tensor,
            b: torch.Tensor,
            profile: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            steps: int = 1000,
            lr: float = 1e-3,
            eps: float = 1e-12,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Locally maximize profile(a, b) over (a, b) constrained to the unit sphere in R^d.

            Args:
                a: tensor with shape (d - 1,)
                b: tensor scalar or shape (1,)
                profile: callable returning a scalar tensor (or a tensor reducible to scalar)
                steps: number of optimizer steps (L-BFGS)
                lr: learning rate
                eps: numerical stability for normalization

            Returns:
                best_val: scalar tensor, the maximal value of profile(a, b) achieved.
                best_a: tensor with shape (d,)
                best_b: scalar tensor
            """
            # Optimize an unconstrained 3D vector and evaluate the profile on its
            # projection onto the unit sphere (keeps the LBFGS line search stable).
            # Here w has dimension (d+1): (a in R^d, b in R).
            d = int(a.numel())
            w0 = torch.cat([a.reshape(-1), b.reshape(-1)]).detach()  # (d+1,)
            w = w0.clone().requires_grad_(True)

            opt = torch.optim.LBFGS(
                [w],
                lr=lr,
                max_iter=steps, # maximal iteration specified here.
                line_search_fn="strong_wolfe",
            )

            best_val = torch.tensor(-float("inf"), device=w.device, dtype=w.dtype)

            def closure() -> torch.Tensor:
                opt.zero_grad()
                w_sphere = w / w.norm().clamp_min(eps)      # projecting input on sphere.
                obj = profile(w_sphere[:d], w_sphere[d])    # maximize the profile.
                (-obj).backward()
                return -obj

            opt.step(closure)

            # profile() internally calls autograd.grad(...), so we must keep autograd enabled.
            # We still detach outputs since we don't need gradients here.
            with torch.enable_grad():
                w_sphere = w / w.norm().clamp_min(eps)
                val = profile(w_sphere[:d], w_sphere[d])
            best_val = val.detach().clone()
            best_a = w_sphere[:d].detach().clone()
            best_b = w_sphere[d].detach().clone()

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
        if verbose:
            logger.info(f"insertion: 1st NOC profile - tried {tried} candidates, accepted {accepted} (alpha={alpha})")

        if len(accepted_a) == 0:
            d = int(X_train.shape[1])
            return np.empty((0, d), dtype=np.float64), np.empty((0,), dtype=np.float64)

        W = torch.stack(accepted_a, dim=0).detach().cpu().numpy()
        b = torch.stack(accepted_b, dim=0).detach().cpu().numpy()
        return W, b


    def sparsify(
        self,
        data_train: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        weights: torch.Tensor,
        biases: torch.Tensor,
        outer_weights: torch.Tensor,
        *,
        tol: float = 1e-10,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One-step Carathéodory-style removal using feature null direction on train set."""
        W = weights.detach()
        b = biases.detach().reshape(-1)
        a = outer_weights.detach().reshape(-1)
        if W.numel() == 0 or int(W.shape[0]) <= 1:
            return W, b, a.reshape(1, -1)

        with torch.no_grad():
            X = data_train[0].detach()
            Phi = self.activation_fn(X @ W.t() + b) ** float(self.model1.power)  # (N,K)
            # Assuming N >= K (more samples than neurons): reduced SVD contains the
            # right singular vector associated with the smallest singular value.
            s = torch.linalg.svdvals(Phi)
            if s.numel() == 0 or float(s[-1].cpu().item()) > tol * float(s[0].cpu().item()):
                return W, b, a.reshape(1, -1)  # full column rank => nothing to do
            _, _, Vh = torch.linalg.svd(Phi, full_matrices=False)
            v = Vh[-1]  # approx null direction, Phi @ v ~= 0
            m = v.abs() > 1e-18
            if not bool(m.any()): # do nothing if all entries are 0s
                return W, b, a.reshape(1, -1)

            taus = a[m] / v[m]
            j = int(torch.argmin(taus.abs()).item())
            idx = int(torch.nonzero(m, as_tuple=False).reshape(-1)[j].item())
            tau = taus[j]
            a = a - tau * v # update the outer weights

            keep = torch.ones(int(W.shape[0]), dtype=torch.bool, device=W.device)
            keep[idx] = False
            if verbose:
                logger.info(f"Sparsify removed 1 dependent neuron (idx={idx})")
            return W[keep], b[keep], a[keep].reshape(1, -1)


    @staticmethod
    def check_linearity_neurons(
        W: torch.Tensor,
        b: torch.Tensor,
        tol: float = 1e-10,
        *,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Keep only "linearly independent" neurons under the strong criterion:
            (w_i, b_i) = a (w_j, b_j) with a > 0  (i.e., same direction in R^{d+1})

        Returns:
            W_kept, b_kept
        """
        b1 = b.reshape(-1)
        U = torch.cat([W, b1.reshape(-1, 1)], dim=1)  # (K, d+1)
        nrm = U.norm(dim=1)
        nonzero = nrm > 1e-12  # drop exact-zero (w,b) which yields identically-zero feature

        idx = torch.nonzero(nonzero, as_tuple=False).reshape(-1)
        if idx.numel() <= 1:
            keep_mask = nonzero.clone()
            if verbose and int((~keep_mask).sum().item()) > 0:
                logger.info(
                    f"Insertion: found {int((~keep_mask).sum().item())} redundant neuron(s) "
                    "(zero (w,b))"
                )
            return W[keep_mask], b1[keep_mask]

        U = (U[idx] / nrm[idx].unsqueeze(1).clamp_min(1e-12))  # normalize rows
        S = U @ U.t()  # cosine similarity; close to 1 => positive scalar multiple

        keep_small = torch.ones(int(idx.numel()), dtype=torch.bool, device=W.device)
        for i in range(int(idx.numel())):
            if bool(keep_small[i]):
                dep = (S[i] > 1.0 - tol)
                dep[i] = False
                keep_small[dep] = False

        keep_mask = torch.zeros(int(W.shape[0]), dtype=torch.bool, device=W.device)
        keep_mask[idx[keep_small]] = True
        if verbose:
            total = int(W.shape[0])
            kept = int(keep_mask.sum().item())
            dropped_zero = int((~nonzero).sum().item())
            dropped_dep = int(idx.numel() - int(keep_small.sum().item()))
            if dropped_zero + dropped_dep > 0:
                logger.info(
                    "Insertion: "
                    f"dropped={dropped_zero + dropped_dep} (zero={dropped_zero}, dependent={dropped_dep}), "
                    f"kept={kept}/{total}"
                )
        return W[keep_mask], b1[keep_mask]

    @staticmethod
    def prune_small_weights(
        weights: torch.Tensor,
        biases: torch.Tensor,
        outer_weights: torch.Tensor,
        threshold: float,
        verbose: bool = True,
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
            if verbose:
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
        decorrelation: bool = False,
        verbose: bool = True,
    ) -> tuple[int, int]:
        
        # Track best model across all iterations
        best_iteration_train = 0
        best_val_loss = float('inf')
        best_train_loss = float('inf')
        # Initialize hidden layer by inserting a first batch of neurons.
        if verbose: logger.info(f"Initialization")
        W_init_np, b_init_np = self.insertion(self.data_train, num_insertion, verbose=verbose)
        W_hidden = torch.as_tensor(W_init_np, dtype=torch.float64, device="cpu")
        b_hidden = torch.as_tensor(b_init_np, dtype=torch.float64, device="cpu")
        if decorrelation:
            W_hidden, b_hidden = self.check_linearity_neurons(W_hidden, b_hidden, verbose=verbose)

        # Training loop
        for i in range(num_iterations):
            """ 1. Train the model. """
            if verbose: logger.info(f"Iteration {i + 1} - Starting Training...")
            self.model1.train(
                self.data_train, 
                self.data_valid, 
                inner_weights=W_hidden, 
                inner_bias=b_hidden, 
                iterations = 1000, 
                display_every = 100
                )
            state = self.model1.net.state_dict()
            W_hidden = state["hidden.weight"].detach().to(device="cpu").clone()
            b_hidden = state["hidden.bias"].detach().to(device="cpu").clone()
            W_out = state["output.weight"].detach().to(device="cpu").clone()
            self.model2.train(
                self.data_train, 
                self.data_valid, 
                inner_weights=W_hidden, 
                inner_bias=b_hidden, 
                outer_weights=W_out, 
                iterations = 1000, 
                display_every = 100
                )
            
            # Pull the *current* trained weights from the model (train() restores its best checkpoint).
            W_hidden = self.model2.net.hidden.weight.detach().to(device="cpu").clone()
            b_hidden = self.model2.net.hidden.bias.detach().to(device="cpu").clone()
            W_outer = self.model2.net.output.weight.detach().to(device="cpu").clone()

            """ 2. record the loss and weights. """
            # Note: we must keep autograd enabled because `_compute_loss` uses `autograd.grad`.
            train_loss_t, _, _ = self.model2._compute_loss(*self.data_train)
            val_loss_t, _, _ = self.model2._compute_loss(*self.data_valid)
            train_loss_f = float(train_loss_t.detach().cpu().item())
            val_loss_f = float(val_loss_t.detach().cpu().item())

            self.train_loss.append(train_loss_f)
            self.val_loss.append(val_loss_f)
            self.inner_weights.append(
                {
                    "weight": W_hidden.detach().to(device="cpu").clone(),
                    "bias": b_hidden.detach().to(device="cpu").clone(),
                }
            )
            self.outer_weights.append(W_outer.detach().to(device="cpu").clone())
            
            # Record the best training and validation loss and choose the best iteration w.r.t. the training loss
            if val_loss_f < best_val_loss:
                best_val_loss = val_loss_f
            if train_loss_f < best_train_loss:
                best_train_loss = train_loss_f
                best_iteration_train = i
                if verbose:
                    logger.info(
                        f"New best train loss at iteration {i}: {best_train_loss:.6f}"
                    )
            
            """ 4. Insert neurons and train. """
            W_to_insert, b_to_insert = self.insertion(self.data_train, num_insertion, verbose=verbose)
            W_to_insert_t = torch.as_tensor(W_to_insert, dtype=W_hidden.dtype, device=W_hidden.device)
            b_to_insert_t = torch.as_tensor(b_to_insert, dtype=b_hidden.dtype, device=b_hidden.device)
            W_hidden = torch.cat((W_hidden, W_to_insert_t), dim=0)
            b_hidden = torch.cat((b_hidden, b_to_insert_t), dim=0)
            if decorrelation:
                W_hidden, b_hidden = self.check_linearity_neurons(W_hidden, b_hidden, verbose=verbose)

        best_neurons = int(self.inner_weights[best_iteration_train]["weight"].shape[0])
        return best_iteration_train, best_neurons