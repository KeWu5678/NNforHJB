"""The unified PDAP outer loop.

One ``PDAP`` class replaces the former ``PDPA_v1/v2/v3``.  It is configured by
two axes (plus convenience aliases in :mod:`registry`):

  * ``model``     — ``"signed"`` (pure network) or ``"semiconcave"`` (V = 0.5 C||x||^2 - g).
  * ``insertion`` — ``"profile"`` (dual-threshold) or ``"finite_step"`` (Delta J < 0).

The loop is model-agnostic: it drives the model through the uniform interface
(``set_atoms`` / ``get_atoms`` / ``warm_start`` / ``fit_outer_weights`` /
``predict_tensors``) and the insertion strategy through :mod:`insertion`.

  init:  insert -> warm-start -> set_atoms
  loop:  fit_outer_weights -> get_atoms -> prune -> set_atoms -> record
         -> insert -> warm-start -> append -> set_atoms
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch

from ..models.signed import SignedModel
from ..models.semiconcave import SemiconcaveModel
from .insertion import profile_threshold, finite_step

logger = logging.getLogger(__name__)


class PDAP:
    def __init__(
        self,
        data: dict,
        alpha: float,
        gamma: float,
        power: float,
        model: str = "signed",
        insertion: str = "profile",
        activation=torch.relu,
        loss_weights: Tuple[float, float] | str = "h1",
        lr: float = 1.0,
        th: float = 0.5,
        use_sphere: bool = True,
        c_init: float = 1.0,
        verbose: bool = True,
    ) -> None:
        if model not in ("signed", "semiconcave"):
            raise ValueError(f"model must be 'signed' or 'semiconcave', got {model!r}")
        if insertion not in ("profile", "finite_step"):
            raise ValueError(f"insertion must be 'profile' or 'finite_step', got {insertion!r}")

        # Resolve the loss_weights shorthand here (SignedModel does not).
        if isinstance(loss_weights, str):
            _map = {"l2": (1.0, 0.0), "h1": (1.0, 1.0)}
            key = loss_weights.lower()
            if key not in _map:
                raise ValueError(f"loss_weights must be 'l2', 'h1', or a tuple, got {loss_weights!r}")
            loss_weights = _map[key]

        self.alpha = float(alpha)
        self.insertion_kind = insertion
        self.activation_fn = activation
        self._use_sphere = bool(use_sphere)
        self.verbose = verbose

        # histories
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

        if model == "signed":
            # signed network: convex atoms not required => two-sided dual test.
            self.two_sided = True
            N_total = data["x"].shape[0]
            self.model = SignedModel(
                alpha=alpha, gamma=gamma, optimizer="SSN", activation=activation,
                power=power, lr=lr, loss_weights=loss_weights, th=th, verbose=verbose,
                train_outerweights=True, training_percentage=(N_total - 1) / N_total,
            )
            self.data_train, self.data_valid = self.model._prepare_data(data)
        else:
            # semiconcave: convex g => one-sided dual test (nonnegative mass).
            self.two_sided = False
            self.model = SemiconcaveModel(
                alpha=alpha, gamma=gamma, power=power, th=th, activation=activation,
                loss_weights=loss_weights, lr=lr, c_init=c_init, verbose=verbose,
            )
            self.data_train, self.data_valid = self.model._prepare_data(data)
            self.model._ensure_affine(int(self.model.input_dim))

        if self.model.input_dim is None:
            raise ValueError("Could not infer input dimension from data['x'] (N, d).")
        self.input_dim = int(self.model.input_dim)

    # ------------------------------------------------------------------ #
    # Shared helpers (formerly on PDPA_v2)
    # ------------------------------------------------------------------ #
    def sample_uniform_sphere_points(self, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample N candidate neurons uniformly on S^d in R^{d+1}."""
        d = int(self.input_dim)
        v = torch.randn(N, d + 1, dtype=torch.float64, device="cpu")
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return v[:, :d].contiguous(), v[:, d].contiguous()

    @staticmethod
    def prune_small_weights(
        weights: torch.Tensor, biases: torch.Tensor, outer_weights: torch.Tensor,
        merge_tol: float = 1e-3, verbose: bool = True, use_sphere: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Merge duplicate neurons (cosine on S^d / Euclidean) and drop zeros."""
        w = weights.detach()
        b = biases.detach().reshape(-1)
        ow = outer_weights.detach().reshape(-1)
        n = w.shape[0]
        if n <= 1:
            return w, b, ow.reshape(1, -1)

        U = torch.cat([w, b.reshape(-1, 1)], dim=1)
        if use_sphere:
            U_normed = U / U.norm(dim=1, keepdim=True).clamp_min(1e-12)
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
                should_merge = (sim[i, j] > 1.0 - merge_tol) if use_sphere else (dists[i, j] < merge_tol)
                if should_merge:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[rj] = ri

        clusters: Dict[int, List[int]] = {}
        for i in range(n):
            clusters.setdefault(find(i), []).append(i)
        reps = sorted(clusters.keys())
        w_out, b_out = w[reps], b[reps]
        ow_out = torch.zeros(len(reps), dtype=ow.dtype, device=ow.device)
        for k, rep in enumerate(reps):
            ow_out[k] = ow[clusters[rep]].sum()

        nonzero = ow_out.abs() > 0
        w_out, b_out, ow_out = w_out[nonzero], b_out[nonzero], ow_out[nonzero]
        merged_count = n - len(reps)
        pruned_zero = int((~nonzero).sum().item())
        if verbose and (merged_count > 0 or pruned_zero > 0):
            logger.info(
                f"Prune: merged {merged_count} duplicates, removed {pruned_zero} zeros, "
                f"kept {w_out.shape[0]}/{n}"
            )
        return w_out, b_out, ow_out.reshape(1, -1)

    @staticmethod
    def check_linearity_neurons(
        W: torch.Tensor, b: torch.Tensor, tol: float = 1e-10, *, verbose: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Drop neurons that are positive scalar multiples of another (same S^d direction)."""
        b1 = b.reshape(-1)
        U = torch.cat([W, b1.reshape(-1, 1)], dim=1)
        nrm = U.norm(dim=1)
        nonzero = nrm > 1e-12
        idx = torch.nonzero(nonzero, as_tuple=False).reshape(-1)
        if idx.numel() <= 1:
            keep_mask = nonzero.clone()
            return W[keep_mask], b1[keep_mask], keep_mask
        Un = U[idx] / nrm[idx].unsqueeze(1).clamp_min(1e-12)
        S = Un @ Un.t()
        keep_small = torch.ones(int(idx.numel()), dtype=torch.bool, device=W.device)
        for i in range(int(idx.numel())):
            if bool(keep_small[i]):
                dep = (S[i] > 1.0 - tol)
                dep[i] = False
                keep_small[dep] = False
        keep_mask = torch.zeros(int(W.shape[0]), dtype=torch.bool, device=W.device)
        keep_mask[idx[keep_small]] = True
        return W[keep_mask], b1[keep_mask], keep_mask

    # ------------------------------------------------------------------ #
    # Insertion dispatch (residuals + existing support read from the model)
    # ------------------------------------------------------------------ #
    # The insertion-candidate merge tolerance is independent of the prune
    # tolerance: the former PDPA_v* called insertion() with its own default 1e-2
    # while passing retrain's merge_tol (default 1e-3) only to prune_small_weights.
    INSERTION_MERGE_TOL = 1e-2

    def _insert(self, num_insertion: int, max_insert: int, verbose: bool):
        """Return (W, b, c) where c is None for the profile strategy (needs warm-start)."""
        X, V, dV = self.data_train
        # Residual = current prediction - target.  The semiconcave model predicts
        # its envelope even with no atoms; the signed network has no prediction
        # until its net is built (then residual = -target, the zero network).
        try:
            Vp, dVp = self.model.predict_tensors(X)
            res_v = (Vp - V).detach()
            res_dv = (dVp - dV).detach()
        except RuntimeError:
            res_v = -V.detach()
            res_dv = -dV.detach()
        existing = None
        if self.model.n_neurons > 0:
            Wc, bc, _ = self.model.get_atoms()
            existing = (Wc, bc) if Wc.shape[0] > 0 else None

        common = dict(
            activation=self.activation_fn, power=self.model.power,
            loss_weights=self.model.loss_weights, alpha=self.alpha,
            sample_sphere=self.sample_uniform_sphere_points, N=num_insertion,
            max_insert=max_insert, merge_tol=self.INSERTION_MERGE_TOL,
            use_sphere=self._use_sphere, existing_atoms=existing, verbose=verbose,
        )
        if self.insertion_kind == "profile":
            W, b = profile_threshold(X, res_v, res_dv, two_sided=self.two_sided, **common)
            return W, b, None
        W, b, c = finite_step(X, res_v, res_dv, **common)
        return W, b, c

    # ------------------------------------------------------------------ #
    # The PDAP outer loop
    # ------------------------------------------------------------------ #
    def fit(
        self,
        num_iterations: int,
        num_insertion: int,
        threshold: float = 1e-4,
        max_insert: int = 15,
        merge_tol: float = 1e-3,
        decorrelation: bool = False,
        verbose: bool = True,
    ) -> dict:
        best_iteration_train = 0
        best_val_loss = float("inf")
        best_train_loss = float("inf")

        # --- initialization: insert + warm-start ---
        if verbose:
            logger.info("Initialization")
        W_np, b_np, c = self._insert(num_insertion, max_insert, verbose)
        W = torch.as_tensor(W_np, dtype=torch.float64)
        b = torch.as_tensor(b_np, dtype=torch.float64)
        if W.shape[0] == 0:
            raise RuntimeError("PDAP: initial insertion accepted no atoms")
        if decorrelation:
            W, b, keep = self.check_linearity_neurons(W, b, verbose=verbose)
        if c is None:  # profile strategy: nonneg/signed coordinate-descent warm-start.
            # warm_start computes its own residual (signed: zero network -> -target;
            # semiconcave: the envelope, valid with no atoms), so no eager net build
            # is needed here -- keeping the signed RNG sequence aligned with the loop.
            c = self.model.warm_start(W, b, self.data_train, use_sphere=self._use_sphere, verbose=verbose)
        else:
            c = torch.as_tensor(c, dtype=torch.float64).reshape(-1)
            if decorrelation:
                c = c[keep]
        self.model.set_atoms(W, b, c)

        for i in range(num_iterations):
            supp_before = self.model.n_neurons

            # 1. SSN on outer weights (inner weights frozen)
            self.model.fit_outer_weights(self.data_train, self.data_valid, iterations=20, display_every=2)

            # 2. prune: merge duplicates + remove zeros
            W, b, c = self.model.get_atoms()
            W, b, c_row = self.prune_small_weights(
                W, b, c.reshape(1, -1), merge_tol=merge_tol, verbose=verbose, use_sphere=self._use_sphere,
            )
            c = c_row.reshape(-1)
            self.model.set_atoms(W, b, c)

            # 3. record losses / errors / weights
            tl = float(self.model._compute_loss(*self.data_train)[0].detach())
            vl = float(self.model._compute_loss(*self.data_valid)[0].detach())
            self.train_loss.append(tl)
            self.val_loss.append(vl)
            l2t, gt, h1t = self.model._compute_relative_errors(*self.data_train)
            l2v, gv, h1v = self.model._compute_relative_errors(*self.data_valid)
            self.err_l2_train.append(l2t); self.err_l2_val.append(l2v)
            self.err_grad_train.append(gt); self.err_grad_val.append(gv)
            self.err_h1_train.append(h1t); self.err_h1_val.append(h1v)
            self.inner_weights.append({"weight": W.clone(), "bias": b.clone()})
            self.outer_weights.append(c.reshape(1, -1).clone())
            if vl < best_val_loss:
                best_val_loss = vl
            if tl < best_train_loss:
                best_train_loss = tl
                best_iteration_train = i

            if verbose:
                logger.info(
                    f"PDAP: {i + 1:3d}, supp: {supp_before}->{self.model.n_neurons}, "
                    f"train: {tl:.2e}, val: {vl:.2e}, L2: {l2v:.2e}, H1: {h1v:.2e}"
                )

            # 4. insert new neurons + warm-start
            W_np, b_np, c_new = self._insert(num_insertion, max_insert, verbose)
            W_new = torch.as_tensor(W_np, dtype=torch.float64)
            b_new = torch.as_tensor(b_np, dtype=torch.float64)
            if W_new.shape[0] > 0:
                if c_new is None:
                    c_new = self.model.warm_start(W_new, b_new, self.data_train, use_sphere=self._use_sphere, verbose=verbose)
                else:
                    c_new = torch.as_tensor(c_new, dtype=torch.float64).reshape(-1)
                W = torch.cat([W, W_new], dim=0)
                b = torch.cat([b, b_new], dim=0)
                c = torch.cat([c, c_new], dim=0)
                if decorrelation:
                    W, b, keep = self.check_linearity_neurons(W, b, verbose=verbose)
                    c = c[keep]
                self.model.set_atoms(W, b, c)

        best_neurons = int(self.inner_weights[best_iteration_train]["weight"].shape[0])
        final_neurons = int(self.model.n_neurons)

        result = {
            "alpha": self.alpha, "gamma": self.model.gamma, "power": self.model.power,
            "loss_weights": tuple(self.model.loss_weights), "activation": self.activation_fn,
            "use_sphere": self._use_sphere, "model": type(self.model).__name__,
            "insertion": self.insertion_kind,
            "num_iterations": num_iterations, "num_insertion": num_insertion, "threshold": threshold,
            "train_loss": list(self.train_loss), "val_loss": list(self.val_loss),
            "err_l2_train": list(self.err_l2_train), "err_l2_val": list(self.err_l2_val),
            "err_grad_train": list(self.err_grad_train), "err_grad_val": list(self.err_grad_val),
            "err_h1_train": list(self.err_h1_train), "err_h1_val": list(self.err_h1_val),
            "inner_weights": list(self.inner_weights), "outer_weights": list(self.outer_weights),
            "best_iteration": best_iteration_train, "best_neurons": best_neurons,
            "final_neurons": final_neurons,
            "best_err_l2_train": self.err_l2_train[best_iteration_train],
            "best_err_h1_train": self.err_h1_train[best_iteration_train],
        }
        if isinstance(self.model, SemiconcaveModel):
            result["C"] = float(self.model.C)
        return result

    def predict(self, x):
        return self.model.predict(x)
