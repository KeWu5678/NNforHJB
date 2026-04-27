#!/usr/bin/env python3
"""
PDPA v3: Finite-step insertion for |c|^q penalty (q < 1).

Key change from v2: replaces the insertion criterion |p(omega)| > alpha
with a finite-step criterion Delta J(c*; omega) < 0, where c* is the
optimal outer weight for inserting a neuron at omega. This handles the
infinite penalty barrier at c=0 for concave penalties phi(z) = z^q.

The coordinate descent warm-start is removed — c* from the insertion
criterion is used directly as the initial outer weight.
"""

from __future__ import annotations
from typing import Any, Dict, List, Callable, Optional, Tuple
from .model import model
import numpy as np
from loguru import logger
import torch


def _solve_insertion_weight(
    p_omega: float,
    S_sq: float,
    alpha: float,
    q: float,
    newton_tol: float = 1e-12,
    max_iter: int = 50,
) -> Optional[Tuple[float, float]]:
    """
    Solve min_c Delta J(c; omega) and return (c*, Delta J(c*)).

    Delta J(c; omega) = c * p_omega + (1/2) * c^2 * S_sq + (alpha/q) * |c|^q

    For q >= 1 (e.g. q=1), falls back to the classical criterion |p| > alpha.

    Handles both signs of p_omega by considering c > 0 (when p_omega < 0)
    and c < 0 (when p_omega > 0).

    Returns None if no profitable insertion exists (Delta J >= 0 for all c).
    """
    if S_sq < 1e-30:
        return None

    abs_p = abs(p_omega)
    if abs_p < 1e-30:
        return None

    # For q >= 1, use the classical criterion
    if q >= 1.0:
        if abs_p <= alpha:
            return None
        # Optimal c for q=1: h(c) = S^2*c + alpha = |p|, so c = (|p| - alpha) / S^2
        c_opt = (abs_p - alpha) / S_sq
        dJ = -abs_p * c_opt + 0.5 * S_sq * c_opt**2 + alpha * c_opt
        # c > 0 when p_omega < 0 (positive weight helps), c < 0 when p_omega > 0
        sign = 1.0 if p_omega < 0 else -1.0
        return (sign * c_opt, dJ)

    # q < 1: finite-step insertion
    # h(c) = S^2 * c + alpha * c^{q-1}, defined for c > 0
    # h has a unique minimum at c_star = (alpha*(1-q)/S^2)^{1/(2-q)}
    c_star = (alpha * (1.0 - q) / S_sq) ** (1.0 / (2.0 - q))
    h_min = S_sq * c_star + alpha * c_star ** (q - 1.0)

    if abs_p < h_min:
        return None  # no solution to h(c) = |p|

    # Newton's method to find c_2 (right branch, c > c_star)
    # F(c) = S^2 * c + alpha * c^{q-1} - |p| = 0
    # F'(c) = S^2 + alpha * (q-1) * c^{q-2}
    # Start from a point on the right branch
    c = max(abs_p / S_sq, 2.0 * c_star)
    for _ in range(max_iter):
        F = S_sq * c + alpha * c ** (q - 1.0) - abs_p
        dF = S_sq + alpha * (q - 1.0) * c ** (q - 2.0)
        if abs(dF) < 1e-30:
            break
        dc = -F / dF
        c_new = c + dc
        # Stay on right branch
        if c_new <= c_star:
            c_new = 0.5 * (c + c_star)
        c = c_new
        if abs(F) < newton_tol * (abs_p + 1e-30):
            break

    # Evaluate Delta J at c_2
    dJ = -abs_p * c + 0.5 * S_sq * c**2 + (alpha / q) * c**q
    if dJ >= 0:
        return None

    # c > 0 when p_omega < 0 (positive weight helps), c < 0 when p_omega > 0
    sign = 1.0 if p_omega < 0 else -1.0
    return (sign * c, dJ)


class PDPA_v3:
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
        self.err_grad_train: List[float] = []
        self.err_grad_val: List[float] = []
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
        self._use_sphere = use_sphere

        # Data split
        self.data_train, self.data_valid = self.model._prepare_data(data)
        if self.model.input_dim is None:
            raise ValueError("Could not infer input dimension from data. Ensure data['x'] has shape (N, d).")
        self.input_dim: int = int(self.model.input_dim)

    def sample_uniform_sphere_points(self, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample N candidate neurons uniformly on S^d in R^{d+1}."""
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Insert neurons using the finite-step criterion Delta J(c*; omega) < 0.

        Instead of checking |p(omega)| > alpha, we solve min_c Delta J(c; omega)
        for each candidate and accept those where the minimum is negative.

        Returns:
            weights:       np.ndarray, shape (n_accepted, d)
            bias:          np.ndarray, shape (n_accepted,)
            outer_weights: np.ndarray, shape (n_accepted,)  — the c* values
        """
        X_train, V_train, dV_train = data_train
        p = self.model.power
        alpha = self.alpha
        q = 2.0 / (p + 1.0)
        K = X_train.shape[0]
        d_dim = X_train.shape[1]
        Kx = K * d_dim
        w1, w2 = self.model.loss_weights
        use_sphere = self._use_sphere

        # Compute residual of current model
        if net is not None:
            X_for_model = X_train.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                model_v = net(X_for_model)
                model_dv = torch.autograd.grad(
                    model_v.sum(), X_for_model, create_graph=False,
                )[0]
            residual_v = (model_v - V_train).detach()
            residual_dv = (model_dv - dV_train).detach()
        else:
            residual_v = -V_train.detach()
            residual_dv = -dV_train.detach()

        # Normalized residual for L-BFGS optimization of direction
        res_norm = torch.sqrt(residual_v.pow(2).sum() + residual_dv.pow(2).sum()).clamp_min(1e-30)
        residual_v_n = residual_v / res_norm
        residual_dv_n = residual_dv / res_norm

        def profile(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """Dual profile |p_t(omega)| on normalized residual (for L-BFGS)."""
            X_cand = X_train.detach().clone().requires_grad_(True)
            pre = X_cand @ a.reshape(-1) + b.reshape(())
            act = self.activation_fn(pre).reshape(-1, 1)
            neuron_v = act ** p
            neuron_dv = torch.autograd.grad(
                outputs=neuron_v.sum(), inputs=X_cand,
                create_graph=True, retain_graph=True,
            )[0]
            val_part = (neuron_v * residual_v_n).sum() / Kx
            grad_part = (neuron_dv * residual_dv_n).sum() / Kx
            return torch.abs(w1 * val_part + w2 * grad_part)

        def compute_signed_profile_and_S_sq(
            a: torch.Tensor, b: torch.Tensor,
        ) -> Tuple[float, float]:
            """
            Compute signed p(omega) and S(omega)^2 on unnormalized residual.

            p(omega) = (w1/Kx) * S_val^T @ res_val + (w2/Kx) * S_grad^T @ res_grad
            S_sq     = (w1/Kx) * ||S_val||^2 + (w2/Kx) * ||S_grad||^2
            """
            X_cand = X_train.detach().clone().requires_grad_(True)
            pre = X_cand @ a.reshape(-1) + b.reshape(())
            act = self.activation_fn(pre).reshape(-1, 1)
            neuron_v = act ** p
            neuron_dv = torch.autograd.grad(
                outputs=neuron_v.sum(), inputs=X_cand,
                create_graph=True, retain_graph=True,
            )[0]

            S_val = neuron_v.detach().reshape(-1)
            S_grad = neuron_dv.detach().reshape(-1)
            res_v = residual_v.reshape(-1)
            res_dv = residual_dv.reshape(-1)

            p_omega = float(
                (w1 / Kx) * S_val.dot(res_v) + (w2 / Kx) * S_grad.dot(res_dv)
            )
            S_sq = float(
                (w1 / Kx) * S_val.dot(S_val) + (w2 / Kx) * S_grad.dot(S_grad)
            )
            return p_omega, S_sq

        def local_maximize_batch(
            a_batch: torch.Tensor,
            b_batch: torch.Tensor,
            steps: int = 200,
            lr: float = 1e-2,
            eps: float = 1e-12,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Locally maximize profile for each candidate independently."""
            results_a, results_b = [], []
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
            """Merge nearby candidates by cosine similarity on S^d."""
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

        if net is not None and hasattr(net, 'hidden'):
            W_exist = net.hidden.weight.detach()
            b_exist = net.hidden.bias.detach()
            if W_exist.shape[0] > 0:
                U_exist = torch.cat([W_exist, b_exist.reshape(-1, 1)], dim=1)
                nrm = U_exist.norm(dim=1, keepdim=True).clamp_min(1e-12)
                U_exist = U_exist / nrm
                n_exist = U_exist.shape[0]
                if n_exist > N // 2:
                    perm = torch.randperm(n_exist)[:N // 2]
                    U_exist = U_exist[perm]
                a_t = torch.cat([a_t, U_exist[:, :d_dim]], dim=0)
                b_t = torch.cat([b_t, U_exist[:, d_dim]], dim=0)

        # Step 2: Iterative optimize + merge
        max_refine = 5
        for refine_iter in range(max_refine):
            a_t, b_t = local_maximize_batch(a_t, b_t, steps=200)
            n_before = a_t.shape[0]
            a_t, b_t = merge_candidates(a_t, b_t)
            n_after = a_t.shape[0]
            if n_before == n_after:
                break

        # Step 2.5: For non-sphere activations, find optimal scale per direction.
        if not use_sphere:
            scaled_a: List[torch.Tensor] = []
            scaled_b: List[torch.Tensor] = []
            for a_hat, b_hat in zip(a_t, b_t):
                s = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
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

        # Step 3: Finite-step insertion criterion
        accepted_a: List[torch.Tensor] = []
        accepted_b: List[torch.Tensor] = []
        accepted_c: List[float] = []
        accepted_dJ: List[float] = []
        with torch.enable_grad():
            for a_i, b_i in zip(a_t, b_t):
                p_omega, S_sq = compute_signed_profile_and_S_sq(a_i, b_i)
                result = _solve_insertion_weight(p_omega, S_sq, alpha, q)
                if result is not None:
                    c_star, dJ = result
                    accepted_a.append(a_i.detach())
                    accepted_b.append(b_i.detach())
                    accepted_c.append(c_star)
                    accepted_dJ.append(dJ)

        # Step 4: Sort by Delta J (most negative first) and cap at max_insert
        if len(accepted_dJ) > max_insert:
            order = sorted(range(len(accepted_dJ)), key=lambda i: accepted_dJ[i])
            order = order[:max_insert]
            accepted_a = [accepted_a[i] for i in order]
            accepted_b = [accepted_b[i] for i in order]
            accepted_c = [accepted_c[i] for i in order]
            accepted_dJ = [accepted_dJ[i] for i in order]

        tried = int(N)
        accepted = int(len(accepted_a))
        if verbose:
            logger.info(
                f"insertion: tried {tried}, after merge {n_after}, "
                f"accepted {accepted} (capped <={max_insert}, alpha={alpha}, q={q:.3f})"
            )

        if len(accepted_a) == 0:
            return (
                np.empty((0, d_dim), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
            )

        W = torch.stack(accepted_a, dim=0).detach().cpu().numpy()
        b = torch.stack(accepted_b, dim=0).detach().cpu().numpy()
        c = np.array(accepted_c, dtype=np.float64)
        return W, b, c

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
        """One-step Caratheodory-style removal using feature null direction."""
        W = weights.detach()
        b = biases.detach().reshape(-1)
        a = outer_weights.detach().reshape(-1)
        if W.numel() == 0 or int(W.shape[0]) <= 1:
            return W, b, a.reshape(1, -1)

        with torch.no_grad():
            X = data_train[0].detach()
            Phi = self.activation_fn(X @ W.t() + b) ** float(self.model.power)
            s = torch.linalg.svdvals(Phi)
            if s.numel() == 0 or float(s[-1].cpu().item()) > tol * float(s[0].cpu().item()):
                return W, b, a.reshape(1, -1)
            _, _, Vh = torch.linalg.svd(Phi, full_matrices=False)
            v = Vh[-1]
            m = v.abs() > 1e-18
            if not bool(m.any()):
                return W, b, a.reshape(1, -1)

            taus = a[m] / v[m]
            j = int(torch.argmin(taus.abs()).item())
            idx = int(torch.nonzero(m, as_tuple=False).reshape(-1)[j].item())
            tau = taus[j]
            a = a - tau * v

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
        """Keep only linearly independent neurons (same direction criterion)."""
        b1 = b.reshape(-1)
        U = torch.cat([W, b1.reshape(-1, 1)], dim=1)
        nrm = U.norm(dim=1)
        nonzero = nrm > 1e-12

        idx = torch.nonzero(nonzero, as_tuple=False).reshape(-1)
        if idx.numel() <= 1:
            keep_mask = nonzero.clone()
            if verbose and int((~keep_mask).sum().item()) > 0:
                logger.info(
                    f"Insertion: found {int((~keep_mask).sum().item())} redundant neuron(s) "
                    "(zero (w,b))"
                )
            return W[keep_mask], b1[keep_mask], keep_mask

        U = (U[idx] / nrm[idx].unsqueeze(1).clamp_min(1e-12))
        S = U @ U.t()

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
        """Merge duplicate neurons and remove zeros."""
        w = weights.detach()
        b = biases.detach().reshape(-1)
        ow = outer_weights.detach().reshape(-1)
        n = w.shape[0]

        if n <= 1:
            return w, b, ow.reshape(1, -1)

        U = torch.cat([w, b.reshape(-1, 1)], dim=1)
        if use_sphere:
            nrm = U.norm(dim=1, keepdim=True).clamp_min(1e-12)
            U_normed = U / nrm
            sim = U_normed @ U_normed.T
        else:
            dists = torch.cdist(U.unsqueeze(0), U.unsqueeze(0)).squeeze(0)

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

        clusters: Dict[int, List[int]] = {}
        for i in range(n):
            clusters.setdefault(find(i), []).append(i)

        reps = sorted(clusters.keys())
        w_out = w[reps]
        b_out = b[reps]
        ow_out = torch.zeros(len(reps), dtype=ow.dtype, device=ow.device)
        for k, rep in enumerate(reps):
            ow_out[k] = ow[clusters[rep]].sum()

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
        PDPA v3 training loop with finite-step insertion:
          1. Insert neurons via Delta J(c*; omega) < 0 criterion
          2. SSN on all outer weights (with |c|^q penalty)
          3. Merge duplicates + remove zeros
          4. Repeat
        """
        best_iteration_train = 0
        best_val_loss = float('inf')
        best_train_loss = float('inf')

        # Insert initial batch of neurons
        if verbose:
            logger.info("Initialization")
        W_init_np, b_init_np, c_init_np = self.insertion(
            self.data_train, num_insertion, net=None,
            max_insert=max_insert, verbose=verbose
        )
        W_hidden = torch.as_tensor(W_init_np, dtype=torch.float64, device="cpu")
        b_hidden = torch.as_tensor(b_init_np, dtype=torch.float64, device="cpu")
        W_outer = torch.as_tensor(c_init_np, dtype=torch.float64, device="cpu").reshape(1, -1)

        if decorrelation:
            W_hidden, b_hidden, keep_mask = self.check_linearity_neurons(
                W_hidden, b_hidden, verbose=verbose
            )
            W_outer = W_outer.reshape(-1)[keep_mask].reshape(1, -1)

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

            # 2. Merge duplicate neurons + remove zeros
            W_hidden, b_hidden, W_outer = self.prune_small_weights(
                W_hidden, b_hidden, W_outer, merge_tol=merge_tol, verbose=verbose,
                use_sphere=self._use_sphere,
            )

            # Rebuild network with pruned weights
            self.model._create_network(W_hidden, b_hidden, W_outer)

            # 3. Record loss and weights
            train_loss_t, _, _ = self.model._compute_loss(*self.data_train)
            val_loss_t, _, _ = self.model._compute_loss(*self.data_valid)
            train_loss_f = float(train_loss_t.detach().cpu().item())
            val_loss_f = float(val_loss_t.detach().cpu().item())

            self.train_loss.append(train_loss_f)
            self.val_loss.append(val_loss_f)

            l2_tr, grad_tr, h1_tr = self.model._compute_relative_errors(*self.data_train)
            l2_va, grad_va, h1_va = self.model._compute_relative_errors(*self.data_valid)
            self.err_l2_train.append(l2_tr)
            self.err_l2_val.append(l2_va)
            self.err_grad_train.append(grad_tr)
            self.err_grad_val.append(grad_va)
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
            if verbose:
                logger.info(
                    f"PDAP: {i + 1:3d}, supp: {supp_before}->{supp_after}, "
                    f"train: {train_loss_f:.2e}, val: {val_loss_f:.2e}, "
                    f"L2: {l2_va:.2e}, H1: {h1_va:.2e}"
                )

            # 4. Insert new neurons
            if not ssn_made_progress and verbose:
                logger.info("SSN made no progress this round")

            W_new_np, b_new_np, c_new_np = self.insertion(
                self.data_train, num_insertion, net=self.model.net,
                max_insert=max_insert, verbose=verbose
            )
            W_new = torch.as_tensor(W_new_np, dtype=W_hidden.dtype, device=W_hidden.device)
            b_new = torch.as_tensor(b_new_np, dtype=b_hidden.dtype, device=b_hidden.device)

            # Use c* directly as initial outer weights (no coordinate descent)
            if W_new.shape[0] > 0:
                W_outer_new = torch.as_tensor(
                    c_new_np, dtype=W_outer.dtype, device=W_outer.device
                ).reshape(1, -1)
                W_hidden = torch.cat((W_hidden, W_new), dim=0)
                b_hidden = torch.cat((b_hidden, b_new), dim=0)
                W_outer = torch.cat((W_outer, W_outer_new), dim=1)

            # 5. Optional decorrelation
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
            "err_grad_train": list(self.err_grad_train),
            "err_grad_val": list(self.err_grad_val),
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
