"""The unified PDAP outer loop.

``PDAP`` is a pure trainer: it holds only configuration (the objective, the SSN
solver settings, and the insertion constants).  The model and the data are
arguments to :meth:`PDAP.fit`, which mutates the model's support in place and
returns a :class:`History` of per-iteration metrics — the runtime state lives
with the caller, not the trainer.

The loop is model-agnostic: it drives the model through its contract
(:class:`src.models.base.PDAPModel`) and runs the SSN outer solve, warm start,
and insertion as trainer steps (:mod:`ssn_solve`, :mod:`warmstart`,
:mod:`insertion`); evaluation is recorded into the :class:`History`.

  init:  insert -> warm-start -> set_atoms
  loop:  ssn_solve -> prune -> record -> insert -> warm-start -> set_atoms
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch

from ..models.signed import SignedModel
from .insertion import profile_threshold, finite_step
from .warmstart import warm_start
from .ssn_solve import ssn_solve, Objective, SolverConfig
from .history import History

logger = logging.getLogger(__name__)


class PDAP:
    def __init__(self, cfg) -> None:
        """Configure the trainer from a composed config (``model`` / ``training``).

        Only configuration is read here; the model is built by
        :func:`src.models.build_model` and passed to :meth:`fit` along with the
        data — both owned by the caller.
        """
        m, t = cfg.model, cfg.training
        if m.insertion not in ("profile", "finite_step"):
            raise ValueError(f"model.insertion must be 'profile' or 'finite_step', got {m.insertion!r}")

        self.insertion_kind = m.insertion
        self._use_sphere = m.use_sphere
        self.verbose = cfg.env.verbose
        # The objective (what is minimized) and the SSN solver settings.
        self.objective = Objective(
            alpha=m.alpha, gamma=m.gamma, th=m.th, loss_weights=tuple(m.loss_weights),
        )
        self.solver = SolverConfig(
            lr=t.lr, method=t.method, max_ls_iter=t.max_ls_iter,
            tolerance_ls=t.tolerance_ls, tolerance_grad=t.tolerance_grad, sigmamax=t.sigmamax,
        )
        # outer-loop + insertion settings
        self.fit_outer_iterations = t.fit_outer_iterations
        self.ins_merge_tol = t.ins_merge_tol
        self.lbfgs_lr = t.lbfgs_lr
        self.lbfgs_steps = t.lbfgs_steps
        self.newton_tol = t.newton_tol
        self.newton_max_iter = t.newton_max_iter

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #
    def _sphere(self, d: int, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample N candidate neurons uniformly on S^d in R^{d+1}."""
        v = torch.randn(N, d + 1, dtype=torch.float64, device="cpu")
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return v[:, :d].contiguous(), v[:, d].contiguous()

    @staticmethod
    def prune_small_weights(
        weights: torch.Tensor, biases: torch.Tensor, outer_weights: torch.Tensor,
        amp_tol: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Defensive gate: drop atoms whose outer weight is negligible.

        An atom with ``|c| <= amp_tol`` is effectively the zero measure (the
        regularizer ``alpha * ||mu||_M`` charges per ``|c|``), so it is removed.
        No clustering/merging is performed; redundant near-duplicate atoms are
        harmless and handled by the next solver iteration.

        Returns ``(W (n,d), b (n,), c (n,), pruned)`` with ``pruned`` the number
        of atoms dropped.
        """
        w = weights.detach()
        b = biases.detach().reshape(-1)
        ow = outer_weights.detach().reshape(-1)
        n = w.shape[0]

        keep = ow.abs() > amp_tol
        w_out, b_out, ow_out = w[keep], b[keep], ow[keep]
        pruned = n - int(keep.sum().item())
        return w_out, b_out, ow_out, pruned

    # ------------------------------------------------------------------ #
    # Per-step helpers (model + data are arguments, not state).
    # ------------------------------------------------------------------ #
    def _residual(self, model, data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(prediction - target)`` for value and gradient.

        Both models predict with no atoms — the semiconcave envelope, or the
        signed zero network (V = 0) — so the first insertion sees ``-target``.
        """
        X, V, dV = data
        Vp, dVp = model.predict_tensors(X)
        return (Vp - V).detach(), (dVp - dV).detach()

    def _warm_start(self, model, data_train, W, b, verbose: bool) -> torch.Tensor:
        """Coordinate-descent initial outer weights for new atoms (W, b)."""
        o = self.objective
        return warm_start(
            W, b, self._residual(model, data_train), data_train[0],
            activation=model.activation, power=model.power,
            loss_weights=o.loss_weights, alpha=o.alpha, th=o.th, gamma=o.gamma,
            use_sphere=self._use_sphere, nonneg=not isinstance(model, SignedModel),
            verbose=verbose,
        )

    def _initial_outer_weights(self, model, data_train, W, b, c, verbose: bool) -> torch.Tensor:
        """Outer weights for freshly inserted atoms: warm-start (profile, c is
        None) or the finite-step's own coefficients."""
        if c is None:
            return self._warm_start(model, data_train, W, b, verbose)
        return torch.as_tensor(c, dtype=torch.float64).reshape(-1)

    # ------------------------------------------------------------------ #
    # Insertion dispatch.  The insertion-candidate merge tolerance
    # (self.ins_merge_tol, default 1e-2) is independent of the prune amplitude
    # gate (fit's amp_tol, used only in prune_small_weights).
    # ------------------------------------------------------------------ #
    def _insert(self, model, data_train, num_insertion: int, max_insert: int, verbose: bool):
        """Return (W, b, c) where c is None for the profile strategy (needs warm-start)."""
        X = data_train[0]
        res_v, res_dv = self._residual(model, data_train)
        existing = None
        if model.n_neurons > 0:
            Wc, bc, _ = model.get_atoms()
            existing = (Wc, bc) if Wc.shape[0] > 0 else None

        d = int(model.input_dim)
        two_sided = isinstance(model, SignedModel)
        common = dict(
            activation=model.activation, power=model.power,
            loss_weights=self.objective.loss_weights, alpha=self.objective.alpha,
            sample_sphere=lambda N: self._sphere(d, N), N=num_insertion,
            max_insert=max_insert, merge_tol=self.ins_merge_tol,
            use_sphere=self._use_sphere, existing_atoms=existing, verbose=verbose,
            lbfgs_lr=self.lbfgs_lr, lbfgs_steps=self.lbfgs_steps,
        )
        if self.insertion_kind == "profile":
            W, b = profile_threshold(X, res_v, res_dv, two_sided=two_sided, **common)
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
        model,
        data_train,
        data_valid,
        *,
        num_iterations: int,
        num_insertion: int,
        max_insert: int = 15,
        amp_tol: float = 1e-8,
        verbose: bool = True,
    ) -> History:
        """Train ``model`` in place on ``data_train``; return the :class:`History`."""
        history = History()
        o = self.objective
        if verbose:
            logger.info("PDAP run")
            logger.info("  +------------------+--------------------------+")
            logger.info("  | %-16s | %-24s |", "model", type(model).__name__)
            logger.info("  | %-16s | %-24s |", "insertion rule", self.insertion_kind.replace("_", " "))
            logger.info("  | %-16s | %-24s |", "samples",
                        f"{int(data_train[0].shape[0])} train, {int(data_valid[0].shape[0])} validation")
            logger.info("  | %-16s | %-24d |", "input dimension", int(model.input_dim))
            logger.info("  | %-16s | %-24.2e |", "alpha", o.alpha)
            logger.info("  | %-16s | %-24.2e |", "gamma", o.gamma)
            logger.info("  | %-16s | %-24.3g |", "activation power", model.power)
            logger.info("  +------------------+--------------------------+")

        # --- initialization: insert + warm-start ---
        W_np, b_np, c = self._insert(model, data_train, num_insertion, max_insert, verbose)
        W = torch.as_tensor(W_np, dtype=torch.float64)
        b = torch.as_tensor(b_np, dtype=torch.float64)
        if W.shape[0] == 0:
            raise RuntimeError("PDAP: initial insertion accepted no atoms")
        c = self._initial_outer_weights(model, data_train, W, b, c, verbose)
        model.set_atoms(W, b, c)
        if verbose:
            max_weight = float(c.abs().max().item()) if c.numel() else 0.0
            logger.debug("Initial support  neurons=%d  max |output|=%.2e", int(W.shape[0]), max_weight)
            logger.info("Progress")
            logger.info("  +---------+---------+--------+--------------+--------------+------------+------------+")
            logger.info("  | %-7s | %7s | %6s | %12s | %12s | %10s | %10s |",
                        "iter", "neurons", "pruned", "train loss", "val loss", "val L2", "val H1")
            logger.info("  +---------+---------+--------+--------------+--------------+------------+------------+")

        for i in range(num_iterations):
            supp_before = model.n_neurons

            # 1. SSN on outer weights (inner weights frozen)
            ssn_solve(
                model, data_train, self.objective, self.solver,
                iterations=self.fit_outer_iterations, verbose=verbose,
            )

            # 2. prune: defensive gate — drop negligible atoms
            W, b, c = model.get_atoms()
            W, b, c, pruned = self.prune_small_weights(W, b, c, amp_tol=amp_tol)
            model.set_atoms(W, b, c)

            # 3. record (evaluation lives in History.record, not the loop)
            history.record(model, self.objective, data_train, data_valid)

            if verbose:
                if supp_before != model.n_neurons:
                    logger.debug(
                        "Support changed during pruning at iteration %d: %d -> %d neurons",
                        i + 1, supp_before, model.n_neurons,
                    )
                logger.info(
                    "  | %-7s | %7d | %6d | %12.3e | %12.3e | %10.3e | %10.3e |",
                    f"{i + 1}/{num_iterations}", model.n_neurons, pruned,
                    history.train_loss[-1], history.val_loss[-1],
                    history.err_l2_val[-1], history.err_h1_val[-1],
                )

            # 4. insert new neurons + warm-start
            W_np, b_np, c_new = self._insert(model, data_train, num_insertion, max_insert, verbose)
            W_new = torch.as_tensor(W_np, dtype=torch.float64)
            b_new = torch.as_tensor(b_np, dtype=torch.float64)
            if W_new.shape[0] > 0:
                c_new = self._initial_outer_weights(model, data_train, W_new, b_new, c_new, verbose)
                W = torch.cat([W, W_new], dim=0)
                b = torch.cat([b, b_new], dim=0)
                c = torch.cat([c, c_new], dim=0)
                model.set_atoms(W, b, c)

        history.final_neurons = int(model.n_neurons)
        if verbose:
            logger.info("  +---------+---------+--------+--------------+--------------+------------+------------+")
            logger.info("Result")
            logger.info("  +------------------+--------------------------+")
            logger.info("  | %-16s | %-24d |", "best iteration", history.best_iteration + 1)
            logger.info("  | %-16s | %-24.3e |", "best train loss", history.best_train_loss)
            logger.info("  | %-16s | %-24d |", "best neurons", history.best_neurons)
            logger.info("  +------------------+--------------------------+")
        return history
