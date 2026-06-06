"""The unified PDAP outer loop.

One ``PDAP`` class is configured by two explicit axes:

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
    def __init__(self, cfg, data: dict) -> None:
        """Build a PDAP from a composed config (``ExperimentConfig`` / ``DictConfig``).

        ``cfg`` carries the ``model`` / ``training`` / ``env`` sections; defaults
        live in ``src.config.schema``, not here. ``data`` is an in-memory
        ``{x, v, dv}`` dict — use :meth:`from_config` to load it from ``cfg.data``.
        """
        from ..config.activations import get_activation

        m, t = cfg.model, cfg.training
        if m.kind not in ("signed", "semiconcave"):
            raise ValueError(f"model.kind must be 'signed' or 'semiconcave', got {m.kind!r}")
        if m.insertion not in ("profile", "finite_step"):
            raise ValueError(f"model.insertion must be 'profile' or 'finite_step', got {m.insertion!r}")

        self.alpha = float(m.alpha)
        self.insertion_kind = m.insertion
        self.activation_fn = get_activation(m.activation)
        self._use_sphere = bool(m.use_sphere)
        self.verbose = bool(cfg.env.verbose)
        # outer-loop + insertion settings (threaded into fit / _insert)
        self.fit_outer_iterations = int(t.fit_outer_iterations)
        self.display_every = int(t.display_every)
        self.ins_merge_tol = float(t.ins_merge_tol)
        self.lbfgs_lr = float(t.lbfgs_lr)
        self.lbfgs_steps = int(t.lbfgs_steps)
        self.newton_tol = float(t.newton_tol)
        self.newton_max_iter = int(t.newton_max_iter)
        _solver_kwargs = dict(
            method=t.method, max_ls_iter=t.max_ls_iter, tolerance_ls=t.tolerance_ls,
            tolerance_grad=t.tolerance_grad, sigmamax=t.sigmamax,
        )

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

        loss_weights = tuple(m.loss_weights)
        if m.kind == "signed":
            # signed network: convex atoms not required => two-sided dual test.
            self.two_sided = True
            N_total = data["x"].shape[0]
            self.model = SignedModel(
                alpha=m.alpha, gamma=m.gamma, optimizer="SSN", activation=self.activation_fn,
                power=m.power, lr=t.lr, loss_weights=loss_weights, th=m.th, verbose=self.verbose,
                train_outerweights=True, training_percentage=(N_total - 1) / N_total,
                **_solver_kwargs,
            )
            self.data_train, self.data_valid = self.model._prepare_data(data)
        else:
            # semiconcave: convex g => one-sided dual test (nonnegative mass).
            self.two_sided = False
            self.model = SemiconcaveModel(
                alpha=m.alpha, gamma=m.gamma, power=m.power, th=m.th, activation=self.activation_fn,
                loss_weights=loss_weights, lr=t.lr, c_init=m.c_init, verbose=self.verbose,
                **_solver_kwargs,
            )
            self.data_train, self.data_valid = self.model._prepare_data(data)
            self.model._ensure_affine(int(self.model.input_dim))

        if self.model.input_dim is None:
            raise ValueError("Could not infer input dimension from data['x'] (N, d).")
        self.input_dim = int(self.model.input_dim)
        if self.verbose:
            train_count = int(self.data_train[0].shape[0])
            valid_count = int(self.data_valid[0].shape[0])
            logger.info("PDAP run")
            logger.info("  +------------------+--------------------------+")
            logger.info("  | %-16s | %-24s |", "model", type(self.model).__name__)
            logger.info("  | %-16s | %-24s |", "insertion rule", self.insertion_kind.replace("_", " "))
            logger.info("  | %-16s | %-24s |", "samples", f"{train_count} train, {valid_count} validation")
            logger.info("  | %-16s | %-24d |", "input dimension", self.input_dim)
            logger.info("  | %-16s | %-24.2e |", "alpha", self.alpha)
            logger.info("  | %-16s | %-24.2e |", "gamma", self.model.gamma)
            logger.info("  | %-16s | %-24.3g |", "activation power", self.model.power)
            logger.info("  +------------------+--------------------------+")

    # ------------------------------------------------------------------ #
    # Construction / run from a structured config
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, cfg) -> "PDAP":
        """Build a PDAP from a composed config, loading the data from ``cfg.data``.

        Loads ``cfg.data.path`` (resolved under ``DATA_DIR``) and applies max-abs
        normalization when ``cfg.data.normalize`` is set, then builds the model.
        """
        from ..data import load_value_samples, normalize_value_samples

        data = load_value_samples(cfg.data.path)
        if cfg.data.normalize:
            data, _ = normalize_value_samples(data)
        return cls(cfg, data)

    def fit_from_config(self, training, *, verbose: bool = True) -> dict:
        """Run :meth:`fit` using a ``TrainingConfig`` section."""
        return self.fit(
            num_iterations=training.num_iterations,
            num_insertion=training.num_insertion,
            max_insert=training.max_insert,
            amp_tol=training.prune_amp_tol,
            verbose=verbose,
        )

    # ------------------------------------------------------------------ #
    # Shared helpers
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
        amp_tol: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Defensive gate: drop atoms whose outer weight is negligible.

        An atom with ``|c| <= amp_tol`` is effectively the zero measure (the
        regularizer ``alpha * ||mu||_M`` charges per ``|c|``), so it is removed.
        No clustering/merging is performed; redundant near-duplicate atoms are
        harmless and handled by the next solver iteration.

        Returns ``(weights, biases, outer_weights, merged_count, pruned)``;
        ``merged_count`` is always 0 (kept so the PDAP progress-table columns are
        unchanged) and ``pruned`` is the number of atoms dropped.
        """
        w = weights.detach()
        b = biases.detach().reshape(-1)
        ow = outer_weights.detach().reshape(-1)
        n = w.shape[0]

        keep = ow.abs() > amp_tol
        w_out, b_out, ow_out = w[keep], b[keep], ow[keep]
        pruned = n - int(keep.sum().item())
        return w_out, b_out, ow_out.reshape(1, -1), 0, pruned

    # ------------------------------------------------------------------ #
    # Insertion dispatch (residuals + existing support read from the model)
    # ------------------------------------------------------------------ #
    # The insertion-candidate merge tolerance (self.ins_merge_tol, default 1e-2)
    # is independent of the prune amplitude gate (fit's amp_tol, used only in
    # prune_small_weights to drop negligible atoms).
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
            max_insert=max_insert, merge_tol=self.ins_merge_tol,
            use_sphere=self._use_sphere, existing_atoms=existing, verbose=verbose,
            lbfgs_lr=self.lbfgs_lr, lbfgs_steps=self.lbfgs_steps,
        )
        if self.insertion_kind == "profile":
            W, b = profile_threshold(X, res_v, res_dv, two_sided=self.two_sided, **common)
            return W, b, None
        W, b, c = finite_step(
            X, res_v, res_dv,
            newton_tol=self.newton_tol, newton_max_iter=self.newton_max_iter, **common,
        )
        return W, b, c

    # ------------------------------------------------------------------ #
    # The PDAP outer loop
    # ------------------------------------------------------------------ #
    def fit(
        self,
        num_iterations: int,
        num_insertion: int,
        max_insert: int = 15,
        amp_tol: float = 1e-8,
        verbose: bool = True,
    ) -> dict:
        best_iteration_train = 0
        best_val_loss = float("inf")
        best_train_loss = float("inf")

        # --- initialization: insert + warm-start ---
        W_np, b_np, c = self._insert(num_insertion, max_insert, verbose)
        W = torch.as_tensor(W_np, dtype=torch.float64)
        b = torch.as_tensor(b_np, dtype=torch.float64)
        if W.shape[0] == 0:
            raise RuntimeError("PDAP: initial insertion accepted no atoms")
        if c is None:  # profile strategy: nonneg/signed coordinate-descent warm-start.
            # warm_start computes its own residual (signed: zero network -> -target;
            # semiconcave: the envelope, valid with no atoms), so no eager net build
            # is needed here -- keeping the signed RNG sequence aligned with the loop.
            c = self.model.warm_start(W, b, self.data_train, use_sphere=self._use_sphere, verbose=verbose)
        else:
            c = torch.as_tensor(c, dtype=torch.float64).reshape(-1)
        self.model.set_atoms(W, b, c)
        if verbose:
            max_weight = float(c.abs().max().item()) if c.numel() else 0.0
            logger.debug("Initial support  neurons=%d  max |output|=%.2e", int(W.shape[0]), max_weight)
            logger.info("Progress")
            logger.info("  +---------+---------+--------+--------+--------------+--------------+------------+------------+")
            logger.info("  | %-7s | %7s | %6s | %6s | %12s | %12s | %10s | %10s |",
                        "iter", "neurons", "merged", "pruned", "train loss", "val loss", "val L2", "val H1")
            logger.info("  +---------+---------+--------+--------+--------------+--------------+------------+------------+")

        for i in range(num_iterations):
            supp_before = self.model.n_neurons

            # 1. SSN on outer weights (inner weights frozen)
            self.model.fit_outer_weights(
                self.data_train, self.data_valid,
                iterations=self.fit_outer_iterations, display_every=self.display_every,
            )
            fit_summary = getattr(self.model, "last_fit_summary", {})

            # 2. prune: defensive gate — drop negligible atoms
            W, b, c = self.model.get_atoms()
            W, b, c_row, merged_count, pruned_zero = self.prune_small_weights(
                W, b, c.reshape(1, -1), amp_tol=amp_tol,
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
                if supp_before != self.model.n_neurons:
                    logger.debug(
                        "Support changed during pruning at iteration %d: %d -> %d neurons",
                        i + 1, supp_before, self.model.n_neurons,
                    )
                logger.debug(
                    "Output-weight solver at iteration %d selected inner step %s",
                    i + 1, fit_summary.get("best_step", "n/a"),
                )
                logger.info(
                    "  | %-7s | %7d | %6d | %6d | %12.3e | %12.3e | %10.3e | %10.3e |",
                    f"{i + 1}/{num_iterations}", self.model.n_neurons,
                    merged_count, pruned_zero, tl, vl, l2v, h1v,
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
                self.model.set_atoms(W, b, c)

        best_neurons = int(self.inner_weights[best_iteration_train]["weight"].shape[0])
        final_neurons = int(self.model.n_neurons)

        result = {
            "alpha": self.alpha, "gamma": self.model.gamma, "power": self.model.power,
            "loss_weights": tuple(self.model.loss_weights), "activation": self.activation_fn,
            "use_sphere": self._use_sphere, "model": type(self.model).__name__,
            "insertion": self.insertion_kind,
            "num_iterations": num_iterations, "num_insertion": num_insertion,
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
        if verbose:
            logger.info("  +---------+---------+--------+--------+--------------+--------------+------------+------------+")
            logger.info("Result")
            logger.info("  +------------------+--------------------------+")
            logger.info("  | %-16s | %-24d |", "best iteration", best_iteration_train + 1)
            logger.info("  | %-16s | %-24.3e |", "best train loss", best_train_loss)
            logger.info("  | %-16s | %-24d |", "best neurons", best_neurons)
            logger.info("  +------------------+--------------------------+")
        return result

    def predict(self, x):
        return self.model.predict(x)
