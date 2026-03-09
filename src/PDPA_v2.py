#!/usr/bin/env python3
"""
PDPA v2: Primal-Dual Proximal Algorithm for sparse neural networks.

Key change from v1: replaces Adam warm-start with MATLAB-style coordinate
descent (PDAPmultisemidiscrete.m lines 104-147).  This gives new neurons
initial outer weights at the characteristic scale where SSN's proximal
operator can actually zero them out, enabling gamma-dependent sparsity.
"""

from __future__ import annotations
from typing import Any, Dict, List, Callable, Optional, Tuple
from .model import model
from .utils import _phi_prox
import numpy as np
from loguru import logger
import torch


class PDPA_v2:
    def __init__(
        self,
        data: dict,
        alpha: float,
        gamma: float,
        power: float,
        activation: torch.nn.Module = torch.relu,
        loss_weights: Tuple[float, float] | str = "h1",
        lr: float = 1.0,
        optimizer: str = 'SSN',
        use_sphere: bool = True,
        verbose=True
    ) -> None:

        # retrain-iteration histories
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.err_l2_train: List[float] = []
        self.err_l2_val: List[float] = []
        self.err_h1_train: List[float] = []
        self.err_h1_val: List[float] = []
        self.inner_weights: List[Dict[str, torch.Tensor]] = []
        self.outer_weights: List[torch.Tensor] = []

        self.alpha = alpha

        # Resolve string shorthand for loss_weights
        if isinstance(loss_weights, str):
            _loss_weight_map = {"l2": (1.0, 0.0), "h1": (1.0, 1.0)}
            key = loss_weights.lower()
            if key not in _loss_weight_map:
                raise ValueError(f"loss_weights must be 'l2', 'h1', or a tuple, got '{loss_weights}'")
            loss_weights = _loss_weight_map[key]

        # Match MATLAB: use all data for training, leave 1 point for validation
        N_total = data["x"].shape[0]
        training_pct = (N_total - 1) / N_total

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
            train_outerweights=True,
            training_percentage=training_pct
        )

        self.activation_fn = activation

        # use_sphere=True: parameterize neurons on S^d (natural for
        # positively homogeneous activations like ReLU).
        # use_sphere=False: unconstrained (a, b) ∈ R^{d+1}.
        self._use_sphere = use_sphere

        # Data split
        self.data_train, self.data_valid = self.model._prepare_data(data)
        if self.model.input_dim is None:
            raise ValueError("Could not infer input dimension from data. Ensure data['x'] has shape (N, d).")
        self.input_dim: int = int(self.model.input_dim)

    def sample_uniform_sphere_points(self, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample N candidate neurons uniformly on S^d in R^{d+1}.

        For non-homogeneous activations (use_sphere=False), the direction is
        sampled here and the optimal scale is determined later in insertion().

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
        merge_tol: float = 1e-2,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Insert neurons satisfying the NOC (cf. paper eq.(15), Appendix E).

        Matches MATLAB find_max (setup_problem_NN_2d.m:361-427):
          1. Generate N random candidates + existing support on S^d
          2. Optimize each via L-BFGS (moderate iterations)
          3. Merge nearby candidates (cosine similarity on S^d)
          4. Re-optimize merged set; repeat up to max_refine rounds
          5. Accept those with |p_t(ω)| > α, sorted descending, capped at max_insert

        Args:
            data_train: (X, V, dV) training tensors
            N: number of candidate neurons to sample
            net: current trained network (None for initialization = zero network)
            merge_tol: cosine-similarity tolerance for merging (1 - cos_sim < merge_tol)
        Returns:
            weights: np.ndarray, shape = (n_accepted, d)
            bias: np.ndarray, shape = (n_accepted,)
        """

        X_train, V_train, dV_train = data_train
        p = self.model.power
        alpha = self.alpha
        K = X_train.shape[0]
        d_dim = X_train.shape[1]
        Kx = K * d_dim  # Match MATLAB: numel(xhat) = d * N_points
        w1, w2 = self.model.loss_weights
        use_sphere = self._use_sphere

        # Compute residual of current model once for all candidates.
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

        # Normalize residual (matches MATLAB find_max line 385: y_norm = y/norm(y))
        res_norm = torch.sqrt(residual_v.pow(2).sum() + residual_dv.pow(2).sum()).clamp_min(1e-30)
        residual_v_n = residual_v / res_norm
        residual_dv_n = residual_dv / res_norm

        def profile(
            a: torch.Tensor,
            b: torch.Tensor,
        ) -> torch.Tensor:
            """Dual profile |p_t(ω)| on normalized residual."""
            X_cand = X_train.detach().clone().requires_grad_(True)
            pre = X_cand @ a.reshape(-1) + b.reshape(())
            act = self.activation_fn(pre).reshape(-1, 1)
            neuron_v = act ** p
            neuron_dv = torch.autograd.grad(
                outputs=neuron_v.sum(),
                inputs=X_cand,
                create_graph=True,
                retain_graph=True,
            )[0]

            val_part = (neuron_v * residual_v_n).sum() / Kx
            grad_part = (neuron_dv * residual_dv_n).sum() / Kx
            return torch.abs(w1 * val_part + w2 * grad_part)

        def profile_unnorm(
            a: torch.Tensor,
            b: torch.Tensor,
        ) -> torch.Tensor:
            """Dual profile |p_t(ω)| on unnormalized residual (for threshold check)."""
            X_cand = X_train.detach().clone().requires_grad_(True)
            pre = X_cand @ a.reshape(-1) + b.reshape(())
            act = self.activation_fn(pre).reshape(-1, 1)
            neuron_v = act ** p
            neuron_dv = torch.autograd.grad(
                outputs=neuron_v.sum(),
                inputs=X_cand,
                create_graph=True,
                retain_graph=True,
            )[0]

            val_part = (neuron_v * residual_v).sum() / Kx
            grad_part = (neuron_dv * residual_dv).sum() / Kx
            return torch.abs(w1 * val_part + w2 * grad_part)

        def local_maximize_batch(
            a_batch: torch.Tensor,
            b_batch: torch.Tensor,
            steps: int = 200,
            lr: float = 1e-2,
            eps: float = 1e-12,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Locally maximize profile for each candidate independently.
            Uses moderate L-BFGS iterations to avoid all converging to global max.
            """
            results_a = []
            results_b = []
            d = d_dim
            for a0, b0 in zip(a_batch, b_batch):
                w = torch.cat([a0.reshape(-1), b0.reshape(-1)]).detach().clone().requires_grad_(True)
                opt = torch.optim.LBFGS(
                    [w], lr=lr, max_iter=steps,
                    line_search_fn="strong_wolfe",
                )

                def closure() -> torch.Tensor:
                    opt.zero_grad()
                    w_s = w / w.norm().clamp_min(eps)
                    obj = profile(w_s[:d], w_s[d])
                    (-obj).backward()
                    return -obj

                opt.step(closure)
                w_s = (w / w.norm().clamp_min(eps)).detach()
                results_a.append(w_s[:d])
                results_b.append(w_s[d:d+1])

            return torch.stack(results_a), torch.stack(results_b).reshape(-1)

        def merge_candidates(
            a_cands: torch.Tensor,
            b_cands: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Merge nearby candidates by cosine similarity on S^d.

            During the refine loop all candidates are on the sphere
            (direction only), so cosine similarity is the right metric.
            """
            n = a_cands.shape[0]
            if n <= 1:
                return a_cands, b_cands
            U = torch.cat([a_cands, b_cands.reshape(-1, 1)], dim=1)
            nrm = U.norm(dim=1, keepdim=True).clamp_min(1e-12)
            U_normed = U / nrm
            sim = U_normed @ U_normed.T
            keep = torch.ones(n, dtype=torch.bool)
            for i in range(n):
                if keep[i]:
                    for j in range(i + 1, n):
                        if not keep[j]:
                            continue
                        if sim[i, j] > 1.0 - merge_tol:
                            keep[j] = False
            return a_cands[keep], b_cands[keep]

        # Step 1: Generate candidates = random samples + existing support
        a_t, b_t = self.sample_uniform_sphere_points(N)

        # Include existing support as starting candidates (MATLAB find_max:368-371,383)
        if net is not None and hasattr(net, 'hidden'):
            W_exist = net.hidden.weight.detach()
            b_exist = net.hidden.bias.detach()
            if W_exist.shape[0] > 0:
                U_exist = torch.cat([W_exist, b_exist.reshape(-1, 1)], dim=1)
                nrm = U_exist.norm(dim=1, keepdim=True).clamp_min(1e-12)
                U_exist = U_exist / nrm
                # If too many, subsample (MATLAB: cap at Nguess/2)
                n_exist = U_exist.shape[0]
                if n_exist > N // 2:
                    perm = torch.randperm(n_exist)[:N // 2]
                    U_exist = U_exist[perm]
                a_t = torch.cat([a_t, U_exist[:, :d_dim]], dim=0)
                b_t = torch.cat([b_t, U_exist[:, d_dim]], dim=0)

        # Step 2: Iterative optimize + merge (MATLAB find_max:390-414)
        max_refine = 5
        for refine_iter in range(max_refine):
            a_t, b_t = local_maximize_batch(a_t, b_t, steps=200)
            n_before = a_t.shape[0]
            a_t, b_t = merge_candidates(a_t, b_t)
            n_after = a_t.shape[0]
            if n_before == n_after:
                break  # no merging happened, converged

        # Step 2.5: For non-sphere activations, find optimal scale per direction.
        # L-BFGS found the best direction on S^d; now find r* > 0 such that
        # the neuron sigma(r * (x^T a_hat + b_hat)) maximizes the profile.
        if not use_sphere:
            scaled_a: List[torch.Tensor] = []
            scaled_b: List[torch.Tensor] = []
            for a_hat, b_hat in zip(a_t, b_t):
                s = torch.tensor([0.0], dtype=torch.float64,
                                 requires_grad=True)
                opt_s = torch.optim.LBFGS(
                    [s], lr=0.1, max_iter=20,
                    line_search_fn="strong_wolfe",
                )
                a_h = a_hat.detach()
                b_h = b_hat.detach()

                def closure_s() -> torch.Tensor:
                    opt_s.zero_grad()
                    r = torch.exp(s.clamp(-3, 5))
                    obj = profile(r * a_h, r * b_h)
                    (-obj).backward()
                    return -obj

                opt_s.step(closure_s)
                best_r = torch.exp(s.clamp(-3, 5)).detach()
                scaled_a.append((best_r * a_hat).detach())
                scaled_b.append((best_r * b_hat).detach())
            a_t = torch.stack(scaled_a)
            b_t = torch.stack(scaled_b).reshape(-1)

        # Step 3: Evaluate unnormalized profile for threshold check
        accepted_a: List[torch.Tensor] = []
        accepted_b: List[torch.Tensor] = []
        accepted_vals: List[float] = []
        with torch.enable_grad():
            for a_i, b_i in zip(a_t, b_t):
                val = profile_unnorm(a_i, b_i)
                v = float(val.item())
                if v > alpha:
                    accepted_a.append(a_i.detach())
                    accepted_b.append(b_i.detach())
                    accepted_vals.append(v)

        # Step 4: Sort by profile value (descending) and cap at max_insert
        # MATLAB PDAPmultisemidiscrete.m:82-96: sort descending, take top Npoint=15
        if len(accepted_vals) > max_insert:
            order = sorted(range(len(accepted_vals)), key=lambda i: accepted_vals[i], reverse=True)
            order = order[:max_insert]
            accepted_a = [accepted_a[i] for i in order]
            accepted_b = [accepted_b[i] for i in order]
            accepted_vals = [accepted_vals[i] for i in order]

        tried = int(N)
        accepted = int(len(accepted_a))
        if verbose:
            logger.info(f"insertion: tried {tried}, after merge {n_after}, accepted {accepted} (capped <={max_insert}, alpha={alpha})")

        if len(accepted_a) == 0:
            return np.empty((0, d_dim), dtype=np.float64), np.empty((0,), dtype=np.float64)

        W = torch.stack(accepted_a, dim=0).detach().cpu().numpy()
        b = torch.stack(accepted_b, dim=0).detach().cpu().numpy()
        return W, b


    def _coordinate_descent_init(
        self,
        W_new: torch.Tensor,
        b_new: torch.Tensor,
        net,
        data_train: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Compute initial outer weights for new neurons via coordinate descent.

        Matches MATLAB PDAPmultisemidiscrete.m lines 104-147.
        Solves the 1D proximal problem along the combined direction of all
        new neurons to find a single step size tau.

        Args:
            W_new:      Inner weights of new neurons, shape (n_new, d).
            b_new:      Inner biases of new neurons, shape (n_new,).
            net:        Current trained network (None at first iteration).
            data_train: (X, V, dV) training tensors.
            verbose:    Log info.

        Returns:
            Outer weights for new neurons, shape (1, n_new).
        """
        X_train, V_train, dV_train = data_train
        N = X_train.shape[0]
        d = X_train.shape[1]
        Nx = N * d  # Match MATLAB: numel(xhat) = d * N_points
        p = self.model.power
        w1, w2 = self.model.loss_weights
        alpha = self.alpha
        th = self.model.th
        gamma = self.model.gamma
        n_new = W_new.shape[0]

        if n_new == 0:
            return torch.zeros(1, 0, dtype=torch.float64)

        # --- Compute residuals of current network ---
        X_det = X_train.detach()
        if net is not None:
            X_req = X_det.clone().requires_grad_(True)
            with torch.enable_grad():
                pred_v = net(X_req)
                pred_dv = torch.autograd.grad(
                    pred_v.sum(), X_req, create_graph=False
                )[0]
            res_val = (pred_v - V_train).detach().reshape(-1)      # (N,)
            res_grad = (pred_dv - dV_train).detach().reshape(-1)   # (N*d,)
        else:
            res_val = -V_train.detach().reshape(-1)                # (N,)
            res_grad = -dV_train.detach().reshape(-1)              # (N*d,)

        with torch.no_grad():
            # --- Kernel matrices for new neurons ---
            pre = X_det @ W_new.T + b_new                         # (N, n_new)
            act = self.activation_fn(pre)                          # (N, n_new)
            S_val_new = act ** p                                   # (N, n_new)

            # dS/dz where S(z) = σ(z)^p, z = x^T a + b.
            # Chain rule: dS/dz = p * σ(z)^{p-1} * σ'(z).
            # For ReLU, σ'(z) = 1_{z>0}. For general activations, compute
            # σ'(z) via autograd (need enable_grad since we are in no_grad).
            if self._use_sphere:
                act_deriv = (pre > 0).double()                     # (N, n_new)
            else:
                pre_tmp = pre.detach().requires_grad_(True)
                with torch.enable_grad():
                    act_tmp = self.activation_fn(pre_tmp)
                    act_deriv = torch.autograd.grad(
                        act_tmp.sum(), pre_tmp, create_graph=False
                    )[0].detach()

            if p == 1.0:
                dS_dz = act_deriv                                  # (N, n_new)
            else:
                dS_dz = p * act ** (p - 1) * act_deriv             # (N, n_new)

            # S_grad_new: (N*d, n_new)
            dS_dx = dS_dz.unsqueeze(2) * W_new.unsqueeze(0)       # (N, n_new, d)
            S_grad_new = dS_dx.permute(0, 2, 1).reshape(-1, n_new)

            # --- Signed profiles = data-loss gradient w.r.t. each outer weight ---
            # profile_i = (w1/Nx)*S_val_i^T @ res_val + (w2/Nx)*S_grad_i^T @ res_grad
            profiles = (w1 / Nx) * (S_val_new.T @ res_val) \
                     + (w2 / Nx) * (S_grad_new.T @ res_grad)       # (n_new,)

            abs_profiles = profiles.abs()
            best_idx = int(abs_profiles.argmax().item())

            # --- Descent direction coeff (MATLAB lines 105-106) ---
            # Non-best: -sqrt(eps) * sign(profile_i)   (magnitude ~1.5e-8)
            # Best:     -sign(profile_best)             (magnitude 1)
            eps_sqrt = float(torch.finfo(torch.float64).eps) ** 0.5
            safe_abs = abs_profiles.clamp_min(1e-30)
            coeff = -eps_sqrt * profiles / safe_abs
            coeff[best_idx] = -profiles[best_idx] / safe_abs[best_idx]

            # --- Forward evaluation of combined direction ---
            Kvhat_val  = S_val_new  @ coeff                        # (N,)
            Kvhat_grad = S_grad_new @ coeff                        # (N*d,)

            # --- Slope phat and curvature what ---
            phat = float(
                (w1 / Nx) * Kvhat_val.dot(res_val) +
                (w2 / Nx) * Kvhat_grad.dot(res_grad)
            )
            what = float(
                (w1 / Nx) * Kvhat_val.dot(Kvhat_val) +
                (w2 / Nx) * Kvhat_grad.dot(Kvhat_grad)
            )

            # --- Solve 1D proximal problem (MATLAB lines 140-144) ---
            if phat <= -alpha and what > 1e-30:
                sigma = alpha / what
                g = -phat / what
                qq = 2.0 / (p + 1.0)
                tau = _phi_prox(sigma, g, th, gamma, q=qq)
            else:
                tau = 0.0

            W_outer_new = tau * coeff                              # (n_new,)

            if verbose:
                logger.info(
                    f"Coord. descent: phat={phat:.2e}, what={what:.2e}, "
                    f"tau={tau:.4e}, "
                    f"|W_outer| range=[{W_outer_new.abs().min():.2e}, "
                    f"{W_outer_new.abs().max():.2e}]"
                )

        return W_outer_new.reshape(1, -1)


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
        use_sphere: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Merge duplicate neurons and remove zeros.

        For ReLU (use_sphere=True): merges by cosine similarity on S^d
        (only direction matters due to homogeneity).
        For non-ReLU: merges by Euclidean distance in R^{d+1}
        (scale matters).

        Zero-weight neurons (|u| == 0) produced by the proximal operator
        are also removed (MATLAB PDAPmultisemidiscrete.m lines 176-179).

        Args:
            weights:       Inner weights, shape (n, d).
            biases:        Inner biases, shape (n,).
            outer_weights: Outer weights, shape (1, n) or (n,).
            merge_tol:     Cosine-similarity tol (ReLU) or Euclidean
                           distance tol (non-ReLU) for merging.
            verbose:       Log info.
            use_sphere:       Whether to use sphere parameterization.

        Returns:
            (W_kept, b_kept, ow_kept) with ow_kept shaped (1, n_kept).
        """
        w = weights.detach()
        b = biases.detach().reshape(-1)
        ow = outer_weights.detach().reshape(-1)
        n = w.shape[0]

        if n <= 1:
            return w, b, ow.reshape(1, -1)

        # Merge nearby neurons: cosine similarity (ReLU) or Euclidean distance (non-ReLU).
        U = torch.cat([w, b.reshape(-1, 1)], dim=1)  # (n, d+1)
        if use_sphere:
            nrm = U.norm(dim=1, keepdim=True).clamp_min(1e-12)
            U_normed = U / nrm
            sim = U_normed @ U_normed.T  # cosine similarity matrix
        else:
            dists = torch.cdist(U.unsqueeze(0), U.unsqueeze(0)).squeeze(0)

        # Union-find to cluster similar neurons
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n):
            for j in range(i + 1, n):
                if use_sphere:
                    should_merge = sim[i, j] > 1.0 - merge_tol
                else:
                    should_merge = dists[i, j] < merge_tol
                if should_merge:
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
    ) -> dict:
        """
        PDPA training loop matching MATLAB PDAPmultisemidiscrete.m:
          1. Insert neurons
          2. Coordinate descent → initial outer weights for new neurons
          3. SSN on all outer weights
          4. Merge duplicates + remove zeros
          5. Repeat

        Returns a flat result dict containing hyperparameters, training
        config, per-iteration histories, and summary statistics.
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

        # Coordinate descent for initial outer weights (MATLAB lines 104-147)
        W_outer = self._coordinate_descent_init(
            W_hidden, b_hidden, net=None,
            data_train=self.data_train, verbose=verbose
        )

        # Training loop
        for i in range(num_iterations):
            supp_before = W_hidden.shape[0]

            # 1. SSN on outer weights (inner weights frozen)
            ssn_made_progress = self.model.train(
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
                W_hidden, b_hidden, W_outer, merge_tol=merge_tol, verbose=verbose,
                use_sphere=self._use_sphere,
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

            # Relative L2 and H1 errors (equation 45)
            l2_tr, h1_tr = self.model._compute_relative_errors(*self.data_train)
            l2_va, h1_va = self.model._compute_relative_errors(*self.data_valid)
            self.err_l2_train.append(l2_tr)
            self.err_l2_val.append(l2_va)
            self.err_h1_train.append(h1_tr)
            self.err_h1_val.append(h1_va)

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
                    f"train: {train_loss_f:.2e}, val: {val_loss_f:.2e}, "
                    f"L2: {l2_va:.2e}, H1: {h1_va:.2e}"
                )

            # 4. Insert new neurons (always insert, matching MATLAB behavior)
            if not ssn_made_progress and verbose:
                logger.info("SSN made no progress this round")

            W_new_np, b_new_np = self.insertion(
                self.data_train, num_insertion, net=self.model.net, max_insert=max_insert, verbose=verbose
            )
            W_new = torch.as_tensor(W_new_np, dtype=W_hidden.dtype, device=W_hidden.device)
            b_new = torch.as_tensor(b_new_np, dtype=b_hidden.dtype, device=b_hidden.device)

            # 5. Coordinate descent for new neurons' outer weights
            if W_new.shape[0] > 0:
                W_outer_new = self._coordinate_descent_init(
                    W_new, b_new, net=self.model.net,
                    data_train=self.data_train, verbose=verbose
                )
                W_hidden = torch.cat((W_hidden, W_new), dim=0)
                b_hidden = torch.cat((b_hidden, b_new), dim=0)
                W_outer = torch.cat((W_outer, W_outer_new), dim=1)

            # 6. Optional decorrelation (also filters outer weights)
            if decorrelation:
                W_hidden, b_hidden, keep_mask = self.check_linearity_neurons(
                    W_hidden, b_hidden, verbose=verbose
                )
                W_outer = W_outer.reshape(-1)[keep_mask].reshape(1, -1)

        best_neurons = int(self.inner_weights[best_iteration_train]["weight"].shape[0])
        final_neurons = int(W_hidden.shape[0])

        return {
            # hyperparameters
            "alpha": self.alpha,
            "gamma": self.model.gamma,
            "power": self.model.power,
            "loss_weights": tuple(self.model.loss_weights),
            "activation": self.activation_fn,
            "use_sphere": self._use_sphere,
            "optimizer": self.model.optimizer_type,
            # training config
            "num_iterations": num_iterations,
            "num_insertion": num_insertion,
            "threshold": threshold,
            # per-iteration histories
            "train_loss": list(self.train_loss),
            "val_loss": list(self.val_loss),
            "err_l2_train": list(self.err_l2_train),
            "err_l2_val": list(self.err_l2_val),
            "err_h1_train": list(self.err_h1_train),
            "err_h1_val": list(self.err_h1_val),
            "inner_weights": list(self.inner_weights),
            "outer_weights": list(self.outer_weights),
            # summary
            "best_iteration": best_iteration_train,
            "best_neurons": best_neurons,
            "final_neurons": final_neurons,
            "best_err_l2_train": self.err_l2_train[best_iteration_train],
            "best_err_h1_train": self.err_h1_train[best_iteration_train],
        }
