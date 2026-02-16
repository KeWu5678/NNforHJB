#!/usr/bin/env python3
"""
PDPA v1: Primal-Dual Proximal Algorithm for sparse neural networks.

Matches the MATLAB PDAPsemidiscrete.m default flow (optimize_x=false):
  1. Insert neurons (greedy, profile maximization on S^d)
  2. SSN on outer weights only (inner weights fixed after insertion)
  3. Prune zero-weight neurons
  4. Repeat
"""

from __future__ import annotations
from typing import Any, Dict, List, Callable, Optional, Tuple
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
        lr: float = 1.0,
        optimizer: str = 'SSN',
        verbose = True
        ) -> None:

        # retrain-iteration histories
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.inner_weights: List[Dict[str, torch.Tensor]] = []
        self.outer_weights: List[torch.Tensor] = []

        self.alpha = alpha

        # Single model: SSN on outer weights only (MATLAB default: optimize_x=false)
        self.model = model(
            alpha=self.alpha,
            gamma=gamma,
            optimizer=optimizer,
            activation=activation,
            power=power,
            lr=lr,
            loss_weights=loss_weights,
            verbose=verbose,
            train_outerweights=True
        )

        # Warm-start model: SGD on outer weights to move them away from zero
        # before SSN (which requires the NOC to approximately hold).
        # Matches MATLAB PDAPmultisemidiscrete.m lines 135-147 in spirit.
        self.model_warmstart = model(
            alpha=self.alpha,
            gamma=gamma,
            optimizer='Adam',
            activation=activation,
            power=power,
            lr=1e-3,
            loss_weights=loss_weights,
            verbose=False,
            train_outerweights=True
        )

        self.activation_fn = activation

        # Data split
        self.data_train, self.data_valid = self.model._prepare_data(data)
        if self.model.input_dim is None:
            raise ValueError("Could not infer input dimension from data. Ensure data['x'] has shape (N, d).")
        self.input_dim: int = int(self.model.input_dim)

    def sample_uniform_sphere_points(self, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample N points uniformly on the unit sphere in R^(d+1).

        Returns:
            Tuple (a, b) where a has shape (N, d) and b has shape (N,)
        """
        d = int(self.input_dim)
        v = torch.randn(N, d + 1, dtype=torch.float64, device="cpu")
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)

        a = v[:, :d].contiguous()
        b = v[:, d].contiguous()
        return a, b


    def insertion(
        self,
        data_train: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        N: int,
        net=None,
        max_insert: int = 15,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Insert neurons satisfying the NOC (cf. paper eq.(15), Appendix E).

        Maximizes |p_t(ω)| over candidates ω ∈ S^d and inserts those where
        |p_t(ω)| > α.

        Args:
            data_train: (X, V, dV) training tensors
            N: number of candidate neurons to sample
            net: current trained network (None for initialization = zero network)
        Returns:
            weights: np.ndarray, shape = (n_accepted, d)
            bias: np.ndarray, shape = (n_accepted,)
        """

        X_train, V_train, dV_train = data_train
        p = self.model.power
        alpha = self.alpha
        K = X_train.shape[0]
        w1, w2 = self.model.loss_weights

        # Compute residual of current model once for all candidates.
        # residual_v = N_current(x) - y(x),  residual_dv = ∇N_current(x) - ∇y(x)
        # When net is None (initialization), the current network is zero.
        if net is not None:
            X_for_model = X_train.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                model_v = net(X_for_model)
                model_dv = torch.autograd.grad(
                    model_v.sum(), X_for_model,
                    create_graph=False,
                )[0]
            residual_v = (model_v - V_train).detach()    # (K, 1)
            residual_dv = (model_dv - dV_train).detach() # (K, d)
        else:
            residual_v = -V_train.detach()    # (K, 1)
            residual_dv = -dV_train.detach()  # (K, d)

        def profile(
            a: torch.Tensor,
            b: torch.Tensor,
        ) -> torch.Tensor:
            """
            Dual profile |p_t(ω)| evaluated on the training set (cf. paper eq.(15)).

            p_t(ω) = w1/K * Σ_k σ^p(x_k;ω) · res_v(x_k)
                    + w2/K * Σ_k ∇_x σ^p(x_k;ω) · res_dv(x_k)
            Returns |p_t(ω)| as a scalar tensor differentiable w.r.t. (a, b).
            """
            X_cand = X_train.detach().clone().requires_grad_(True)
            pre = X_cand @ a.reshape(-1) + b.reshape(())  # (K,)
            act = self.activation_fn(pre).reshape(-1, 1)   # (K, 1)
            neuron_v = act ** p                             # σ^p: (K, 1)
            neuron_dv = torch.autograd.grad(
                outputs=neuron_v.sum(),
                inputs=X_cand,
                create_graph=True,
                retain_graph=True,
            )[0]  # ∇_x σ^p: (K, d)

            # |Σ·| with 1/K normalization (paper eq.(15))
            val_part = (neuron_v * residual_v).sum() / K       # scalar
            grad_part = (neuron_dv * residual_dv).sum() / K    # scalar
            return torch.abs(w1 * val_part + w2 * grad_part)

        def local_maximize(
            a: torch.Tensor,
            b: torch.Tensor,
            profile: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            steps: int = 1000,
            lr: float = 1e-3,
            eps: float = 1e-12,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Locally maximize profile(a, b) constrained to S^d via L-BFGS."""
            d = int(a.numel())
            w0 = torch.cat([a.reshape(-1), b.reshape(-1)]).detach()
            w = w0.clone().requires_grad_(True)

            opt = torch.optim.LBFGS(
                [w],
                lr=lr,
                max_iter=steps,
                line_search_fn="strong_wolfe",
            )

            def closure() -> torch.Tensor:
                opt.zero_grad()
                w_sphere = w / w.norm().clamp_min(eps)
                obj = profile(w_sphere[:d], w_sphere[d])
                (-obj).backward()
                return -obj

            opt.step(closure)

            with torch.enable_grad():
                w_sphere = w / w.norm().clamp_min(eps)
                val = profile(w_sphere[:d], w_sphere[d])
            best_val = val.detach().clone()
            best_a = w_sphere[:d].detach().clone()
            best_b = w_sphere[d].detach().clone()

            return best_val, best_a, best_b


        # Step 1: Sample candidate points
        a_t, b_t = self.sample_uniform_sphere_points(N)

        # Step 2: Accept neurons where |p_t(ω)| > α
        accepted_a: List[torch.Tensor] = []
        accepted_b: List[torch.Tensor] = []
        accepted_vals: List[float] = []
        for a0, b0 in zip(a_t, b_t):
            best_val, best_a, best_b = local_maximize(a0, b0, profile)
            if float(best_val.item()) > alpha:
                accepted_a.append(best_a)
                accepted_b.append(best_b)
                accepted_vals.append(float(best_val.item()))

        # Step 3: Sort by profile value (descending) and cap at max_insert
        # Matches MATLAB PDAPmultisemidiscrete.m lines 82-96:
        #   sort by norms_grad descending, then locs = locs(1:min(Npoint, length(locs)))
        if len(accepted_a) > max_insert:
            order = sorted(range(len(accepted_vals)), key=lambda i: accepted_vals[i], reverse=True)
            order = order[:max_insert]
            accepted_a = [accepted_a[i] for i in order]
            accepted_b = [accepted_b[i] for i in order]
            accepted_vals = [accepted_vals[i] for i in order]

        tried = int(N)
        accepted = int(len(accepted_a))
        if verbose:
            logger.info(f"insertion: tried {tried} candidates, accepted {accepted} (capped <={max_insert}, alpha={alpha})")

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
            Phi = self.activation_fn(X @ W.t() + b) ** float(self.model.power)  # (N,K)
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Keep only "linearly independent" neurons under the strong criterion:
            (w_i, b_i) = a (w_j, b_j) with a > 0  (i.e., same direction in R^{d+1})

        Returns:
            (W_kept, b_kept, keep_mask)
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
            return W[keep_mask], b1[keep_mask], keep_mask

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
        return W[keep_mask], b1[keep_mask], keep_mask

    @staticmethod
    def prune_small_weights(
        weights: torch.Tensor,
        biases: torch.Tensor,
        outer_weights: torch.Tensor,
        merge_tol: float = 1e-3,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Merge duplicate neurons whose (w, b) directions coincide on S^d.

        Matches MATLAB postprocess (setup_problem_NN_2d.m lines 204-216):
        computes pairwise distance between neuron positions, finds connected
        components within merge_tol, keeps one representative per cluster
        and sums their outer weights.

        Zero-weight neurons (|u| == 0) produced by the proximal operator
        are also removed (MATLAB PDAPmultisemidiscrete.m lines 176-179).

        Args:
            weights:       Inner weights, shape (n, d).
            biases:        Inner biases, shape (n,).
            outer_weights: Outer weights, shape (1, n) or (n,).
            merge_tol:     Cosine-similarity tolerance for merging
                           neurons with the same (w, b) direction.
            verbose:       Log info.

        Returns:
            (W_kept, b_kept, ow_kept) with ow_kept shaped (1, n_kept).
        """
        w = weights.detach()
        b = biases.detach().reshape(-1)
        ow = outer_weights.detach().reshape(-1)
        n = w.shape[0]

        if n <= 1:
            return w, b, ow.reshape(1, -1)

        # Merge neurons whose (w, b) directions coincide on S^d.
        U = torch.cat([w, b.reshape(-1, 1)], dim=1)  # (n, d+1)
        nrm = U.norm(dim=1, keepdim=True).clamp_min(1e-12)
        U_normed = U / nrm
        S = U_normed @ U_normed.T  # cosine similarity matrix

        # Union-find to cluster similar neurons
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n):
            for j in range(i + 1, n):
                if S[i, j] > 1.0 - merge_tol:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[rj] = ri

        # Collect clusters: keep representative (w, b), sum outer weights
        clusters: Dict[int, List[int]] = {}
        for i in range(n):
            clusters.setdefault(find(i), []).append(i)

        reps = sorted(clusters.keys())
        w_out = w[reps]
        b_out = b[reps]
        ow_out = torch.zeros(len(reps), dtype=ow.dtype, device=ow.device)
        for k, rep in enumerate(reps):
            ow_out[k] = ow[clusters[rep]].sum()

        # Remove exact zeros (from proximal operator)
        nonzero = ow_out.abs() > 0
        w_out, b_out, ow_out = w_out[nonzero], b_out[nonzero], ow_out[nonzero]

        merged_count = n - len(reps)
        pruned_zero = int((~nonzero).sum().item())
        if verbose and (merged_count > 0 or pruned_zero > 0):
            logger.info(
                f"Prune: merged {merged_count} duplicates, "
                f"removed {pruned_zero} zeros, "
                f"kept {w_out.shape[0]}/{n}"
            )

        return w_out, b_out, ow_out.reshape(1, -1)


    def retrain(
        self,
        num_iterations: int,
        num_insertion: int,
        threshold: float = 1e-4,
        max_insert: int = 15,
        merge_tol: float = 1e-3,
        decorrelation: bool = False,
        verbose: bool = True,
    ) -> tuple[int, int]:
        """
        PDPA training loop (MATLAB PDAPmultisemidiscrete.m, optimize_x=false):
          1. Insert neurons (capped at max_insert)
          2. Warm-start outer weights (Adam)
          3. SSN on outer weights
          4. Merge duplicates + remove zeros
          5. Repeat
        """
        best_iteration_train = 0
        best_val_loss = float('inf')
        best_train_loss = float('inf')

        # Insert initial batch of neurons
        if verbose: logger.info("Initialization")
        W_init_np, b_init_np = self.insertion(self.data_train, num_insertion, net=None, max_insert=max_insert, verbose=verbose)
        W_hidden = torch.as_tensor(W_init_np, dtype=torch.float64, device="cpu")
        b_hidden = torch.as_tensor(b_init_np, dtype=torch.float64, device="cpu")
        if decorrelation:
            W_hidden, b_hidden, _ = self.check_linearity_neurons(W_hidden, b_hidden, verbose=verbose)

        # Initialize outer weights to zeros (MATLAB line 70: uk_new.u = [uk.u, 0*coeff])
        n_neurons = W_hidden.shape[0]
        W_outer = torch.zeros(1, n_neurons, dtype=torch.float64, device="cpu")

        # Training loop
        for i in range(num_iterations):
            supp_before = W_hidden.shape[0]

            # 1a. Warm-start: SGD/Adam to move zero outer weights away from zero
            #     so SSN's NOC assumption approximately holds.
            self.model_warmstart.train(
                self.data_train,
                self.data_valid,
                inner_weights=W_hidden,
                inner_bias=b_hidden,
                outer_weights=W_outer,
                iterations=20,
                display_every=5
            )
            W_outer = self.model_warmstart.net.output.weight.detach().cpu().clone()

            # 1b. SSN on outer weights (inner weights frozen, warm-started outer weights)
            self.model.train(
                self.data_train,
                self.data_valid,
                inner_weights=W_hidden,
                inner_bias=b_hidden,
                outer_weights=W_outer,
                iterations=20,
                display_every=2
            )

            # Extract trained weights
            W_hidden = self.model.net.hidden.weight.detach().cpu().clone()
            b_hidden = self.model.net.hidden.bias.detach().cpu().clone()
            W_outer = self.model.net.output.weight.detach().cpu().clone()

            # 2. Merge duplicate neurons + remove zeros (MATLAB postprocess + lines 176-179)
            W_hidden, b_hidden, W_outer = self.prune_small_weights(
                W_hidden, b_hidden, W_outer, merge_tol=merge_tol, verbose=verbose
            )

            # Rebuild network with pruned weights for loss computation and insertion
            self.model._create_network(W_hidden, b_hidden, W_outer)

            # 3. Record loss and weights
            train_loss_t, _, _ = self.model._compute_loss(*self.data_train)
            val_loss_t, _, _ = self.model._compute_loss(*self.data_valid)
            train_loss_f = float(train_loss_t.detach().cpu().item())
            val_loss_f = float(val_loss_t.detach().cpu().item())

            self.train_loss.append(train_loss_f)
            self.val_loss.append(val_loss_f)
            self.inner_weights.append({
                "weight": W_hidden.clone(),
                "bias": b_hidden.clone(),
            })
            self.outer_weights.append(W_outer.clone())

            if val_loss_f < best_val_loss:
                best_val_loss = val_loss_f
            if train_loss_f < best_train_loss:
                best_train_loss = train_loss_f
                best_iteration_train = i

            supp_after = W_hidden.shape[0]
            # MATLAB-style summary: PDAP: iter, supp: before->after, j: (train, val)
            if verbose:
                logger.info(
                    f"PDAP: {i + 1:3d}, supp: {supp_before}->{supp_after}, "
                    f"train: {train_loss_f:.2e}, val: {val_loss_f:.2e}"
                )

            # 4. Insert new neurons
            W_new_np, b_new_np = self.insertion(
                self.data_train, num_insertion, net=self.model.net, max_insert=max_insert, verbose=verbose
            )
            W_new = torch.as_tensor(W_new_np, dtype=W_hidden.dtype, device=W_hidden.device)
            b_new = torch.as_tensor(b_new_np, dtype=b_hidden.dtype, device=b_hidden.device)

            # 5. Append new neurons with zero outer weights (warm start, MATLAB line 70)
            if W_new.shape[0] > 0:
                W_hidden = torch.cat((W_hidden, W_new), dim=0)
                b_hidden = torch.cat((b_hidden, b_new), dim=0)
                n_new = W_new.shape[0]
                W_outer = torch.cat(
                    (W_outer, torch.zeros(1, n_new, dtype=W_outer.dtype, device=W_outer.device)),
                    dim=1
                )

            # 6. Optional decorrelation (also filters outer weights)
            if decorrelation:
                W_hidden, b_hidden, keep_mask = self.check_linearity_neurons(
                    W_hidden, b_hidden, verbose=verbose
                )
                W_outer = W_outer.reshape(-1)[keep_mask].reshape(1, -1)

        best_neurons = int(self.inner_weights[best_iteration_train]["weight"].shape[0])
        final_neurons = int(W_hidden.shape[0])
        return best_iteration_train, best_neurons, final_neurons
