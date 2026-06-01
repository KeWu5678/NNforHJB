"""PDPA_v1: the semiconcave model fitted with the *same* pipeline as PDPA_v2.

This subclasses :class:`src.PDPA_v2.PDPA_v2` and keeps the identical loop
(insertion -> coordinate-descent warm-start -> SSN -> prune -> repeat).  The only
thing that differs is the parametric model: instead of the signed shallow
network it fits the semiconcave ansatz ``V = 0.5 C ||x||^2 - g(x)`` with a convex
``g`` (nonnegative outer weights).  Consequences forced by convexity, all handled
here:

  * residuals/insertion profiles are computed from the semiconcave ``V`` (not a
    bare ``net``),
  * atoms enter with a *one-sided* dual test ``profile > alpha`` (convex atoms
    can only carry nonnegative mass),
  * the coordinate-descent warm-start and the SSN proximal keep ``c >= 0``.

Insertion candidate optimisation (sphere sampling + L-BFGS maximisation + cosine
merge) reuses the v2 machinery in spirit; ``prune_small_weights`` and
``sample_uniform_sphere_points`` are inherited unchanged.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from loguru import logger

from .PDPA_v2 import PDPA_v2
from .models.semiconcave import SemiconcaveModel
from .utils import _phi_prox


class PDPA_v1(PDPA_v2):
    def __init__(
        self,
        data: dict,
        alpha: float,
        gamma: float,
        power: float,
        activation=torch.relu,
        loss_weights: Tuple[float, float] | str = "h1",
        lr: float = 1.0,
        th: float = 0.5,
        use_sphere: bool = True,
        c_init: float = 1.0,
        verbose: bool = True,
    ) -> None:
        # histories (same fields the base class exposes)
        self.train_loss, self.val_loss = [], []
        self.err_l2_train, self.err_l2_val = [], []
        self.err_grad_train, self.err_grad_val = [], []
        self.err_h1_train, self.err_h1_val = [], []
        self.inner_weights, self.outer_weights = [], []

        self.alpha = float(alpha)
        self._use_sphere = bool(use_sphere)
        self.activation_fn = activation

        self.model = SemiconcaveModel(
            alpha=alpha, gamma=gamma, power=power, th=th,
            activation=activation, loss_weights=loss_weights, lr=lr,
            c_init=c_init, verbose=verbose,
        )
        self.data_train, self.data_valid = self.model._prepare_data(data)
        self.input_dim = int(self.model.input_dim)
        self.model._ensure_affine(self.input_dim)
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    # Residuals from the semiconcave model
    # ------------------------------------------------------------------ #
    def _residuals(self, data_train):
        X, V, dV = data_train
        Vp, dVp = self.model.predict_tensors(X)
        return (Vp - V).detach(), (dVp - dV).detach()

    # ------------------------------------------------------------------ #
    # Insertion (model residuals, one-sided profile > alpha)
    # ------------------------------------------------------------------ #
    def insertion_semiconcave(self, data_train, N, max_insert=15,
                              merge_tol=1e-2, verbose=True):
        X, V, dV = data_train
        p = self.model.power
        K, d = X.shape
        Nx = K * d
        w1, w2 = self.model.loss_weights
        eps = 1e-12

        res_v, res_dv = self._residuals(data_train)
        rnorm = torch.sqrt(res_v.pow(2).sum() + res_dv.pow(2).sum()).clamp_min(1e-30)
        res_v_n, res_dv_n = res_v / rnorm, res_dv / rnorm

        def profile(a, b, res_value, res_grad):
            Xc = X.detach().clone().requires_grad_(True)
            pre = Xc @ a.reshape(-1) + b.reshape(())
            act = self.activation_fn(pre).reshape(-1, 1)
            nv = act ** p
            # create_graph=True is required: the profile objective contains ndv
            # (the neuron's input-gradient) and the L-BFGS closure backprops the
            # objective w.r.t. the sphere params, differentiating through ndv.
            ndv = torch.autograd.grad(nv.sum(), Xc, create_graph=True, retain_graph=True)[0]
            val = (nv * res_value).sum() / Nx
            grad = (ndv * res_grad).sum() / Nx
            return w1 * val + w2 * grad  # signed (one-sided)

        a_t, b_t = self.sample_uniform_sphere_points(N)
        if self.model.n_neurons > 0:
            U = torch.cat([self.model.W, self.model.b.reshape(-1, 1)], dim=1)
            U = U / U.norm(dim=1, keepdim=True).clamp_min(1e-12)
            ne = U.shape[0]
            if ne > N // 2:
                U = U[torch.randperm(ne)[:N // 2]]
            a_t = torch.cat([a_t, U[:, :d]], dim=0)
            b_t = torch.cat([b_t, U[:, d]], dim=0)

        def maximize(a0, b0):
            w = torch.cat([a0.reshape(-1), b0.reshape(-1)]).detach().clone().requires_grad_(True)
            opt = torch.optim.LBFGS([w], lr=1e-2, max_iter=200, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                ws = w / w.norm().clamp_min(eps)
                obj = profile(ws[:d], ws[d], res_v_n, res_dv_n)
                (-obj).backward()
                return -obj
            opt.step(closure)
            ws = (w / w.norm().clamp_min(eps)).detach()
            return ws[:d], ws[d:d + 1]

        opt_a, opt_b = [], []
        for a0, b0 in zip(a_t, b_t):
            wa, wb = maximize(a0, b0)
            opt_a.append(wa)
            opt_b.append(wb)
        a_t = torch.stack(opt_a)
        b_t = torch.stack(opt_b).reshape(-1)

        # cosine merge on S^d
        Uc = torch.cat([a_t, b_t.reshape(-1, 1)], dim=1)
        Un = Uc / Uc.norm(dim=1, keepdim=True).clamp_min(1e-12)
        sim = Un @ Un.T
        keep = torch.ones(a_t.shape[0], dtype=torch.bool)
        for i in range(a_t.shape[0]):
            if keep[i]:
                for j in range(i + 1, a_t.shape[0]):
                    if keep[j] and sim[i, j] > 1.0 - merge_tol:
                        keep[j] = False
        a_t, b_t = a_t[keep], b_t[keep]

        # accept one-sided profile > alpha (unnormalised residual)
        acc_a, acc_b, acc_v = [], [], []
        for a_i, b_i in zip(a_t, b_t):
            val = float(profile(a_i, b_i, res_v, res_dv).item())
            if val > self.alpha:
                acc_a.append(a_i.detach())
                acc_b.append(b_i.detach())
                acc_v.append(val)
        if len(acc_v) > max_insert:
            order = sorted(range(len(acc_v)), key=lambda i: acc_v[i], reverse=True)[:max_insert]
            acc_a = [acc_a[i] for i in order]
            acc_b = [acc_b[i] for i in order]
        if verbose:
            logger.info(f"semiconcave insertion: accepted {len(acc_a)} (cap {max_insert})")
        if not acc_a:
            return (np.empty((0, d), dtype=np.float64), np.empty((0,), dtype=np.float64))
        return (torch.stack(acc_a).cpu().numpy(), torch.stack(acc_b).cpu().numpy())

    # ------------------------------------------------------------------ #
    # Coordinate-descent warm-start (nonneg c for new atoms)
    # ------------------------------------------------------------------ #
    def _coord_descent(self, W_new, b_new, data_train):
        X, V, dV = data_train
        K, d = X.shape
        Nx = K * d
        p = self.model.power
        w1, w2 = self.model.loss_weights
        n_new = W_new.shape[0]
        if n_new == 0:
            return torch.zeros(0, dtype=torch.float64)

        res_v, res_dv = self._residuals(data_train)
        res_v = res_v.reshape(-1)
        res_dv = res_dv.reshape(-1)

        with torch.no_grad():
            pre = X @ W_new.T + b_new
            act = self.activation_fn(pre)
            S_val = act ** p                                  # (K, n_new)
            if self._use_sphere:
                dz = (pre > 0).double()
            else:
                pt = pre.detach().requires_grad_(True)
                with torch.enable_grad():
                    at = self.activation_fn(pt)
                    dz = torch.autograd.grad(at.sum(), pt)[0].detach()
            dS_dz = dz if p == 1.0 else p * act ** (p - 1) * dz
            S_grad = (dS_dz.unsqueeze(2) * W_new.unsqueeze(0)).permute(0, 2, 1).reshape(-1, n_new)

            # data-loss gradient wrt c_i at c=0 for V = quad - g  =>  +c_i adds -feature to V
            # profile_i = (w1/Nx) S_val_i.res_v + (w2/Nx) S_grad_i.res_dv  (the v1 sign)
            profiles = (w1 / Nx) * (S_val.T @ res_v) + (w2 / Nx) * (S_grad.T @ res_dv)
            abs_p = profiles.abs()
            best = int(abs_p.argmax().item())
            eps_sqrt = float(torch.finfo(torch.float64).eps) ** 0.5
            safe = abs_p.clamp_min(1e-30)
            # nonneg c: move each accepted atom in +profile direction
            coeff = eps_sqrt * profiles / safe
            coeff[best] = profiles[best] / safe[best]   # ~ +1 for positive profile

            Kv = S_val @ coeff
            Kg = S_grad @ coeff
            # directional slope of data loss for V-change of -(Kv) : phat = -(profile along coeff)
            phat = -((w1 / Nx) * Kv.dot(res_v) + (w2 / Nx) * Kg.dot(res_dv))
            what = (w1 / Nx) * Kv.dot(Kv) + (w2 / Nx) * Kg.dot(Kg)
            if phat <= -self.alpha and what > 1e-30:
                sigma = self.alpha / what
                g = -phat / what
                tau = _phi_prox(sigma, g, self.model.th, self.model.gamma, q=self.model.q)
            else:
                tau = 0.0
            c_new = (tau * coeff).clamp_min(0.0)
        return c_new.reshape(-1)

    # ------------------------------------------------------------------ #
    # Pipeline (same structure as PDPA_v2.retrain)
    # ------------------------------------------------------------------ #
    def retrain(self, num_iterations, num_insertion, threshold=1e-4,
                max_insert=15, merge_tol=1e-3, verbose=True, **kwargs) -> dict:
        best_iteration_train = 0
        best_train_loss = float("inf")

        # initial insertion + warm-start
        W_np, b_np = self.insertion_semiconcave(self.data_train, num_insertion,
                                                max_insert=max_insert, verbose=verbose)
        W = torch.as_tensor(W_np, dtype=torch.float64)
        b = torch.as_tensor(b_np, dtype=torch.float64)
        if W.shape[0] == 0:
            raise RuntimeError("semiconcave: initial insertion accepted no atoms")
        self.model.set_atoms(W, b, torch.zeros(W.shape[0], dtype=torch.float64))
        c = self._coord_descent(W, b, self.data_train)
        self.model.set_atoms(W, b, c)

        for i in range(num_iterations):
            supp_before = W.shape[0]

            # 1. SSN (jointly optimises c >= 0, C >= 0, affine)
            self.model.train_ssn(*self.data_train, iterations=20)
            W, b, c = self.model.W.clone(), self.model.b.clone(), self.model.c.clone()

            # 2. prune: merge duplicates + remove zeros (inherited)
            W, b, c_row = self.prune_small_weights(
                W, b, c.reshape(1, -1), merge_tol=merge_tol,
                verbose=verbose, use_sphere=self._use_sphere)
            c = c_row.reshape(-1)
            if W.shape[0] == 0:
                self.model.W = self.model.b = self.model.c = None
            else:
                self.model.set_atoms(W, b, c)

            # 3. record losses/errors
            tl, _, _ = self.model._compute_loss(*self.data_train)
            vl, _, _ = self.model._compute_loss(*self.data_valid)
            tl, vl = float(tl.detach()), float(vl.detach())
            self.train_loss.append(tl)
            self.val_loss.append(vl)
            l2t, gt, h1t = self.model._compute_relative_errors(*self.data_train)
            l2v, gv, h1v = self.model._compute_relative_errors(*self.data_valid)
            self.err_l2_train.append(l2t); self.err_l2_val.append(l2v)
            self.err_grad_train.append(gt); self.err_grad_val.append(gv)
            self.err_h1_train.append(h1t); self.err_h1_val.append(h1v)
            self.inner_weights.append({"weight": W.clone(), "bias": b.clone()})
            self.outer_weights.append(c.reshape(1, -1).clone())
            if tl < best_train_loss:
                best_train_loss = tl
                best_iteration_train = i

            if verbose:
                logger.info(f"PDAP(semiconcave): {i+1}, supp {supp_before}->{W.shape[0]}, "
                            f"train {tl:.2e}, val {vl:.2e}, H1 {h1v:.2e}, C={self.model.C:.3e}")

            # 4. insert + warm-start new atoms
            W_np, b_np = self.insertion_semiconcave(self.data_train, num_insertion,
                                                    max_insert=max_insert, verbose=verbose)
            W_new = torch.as_tensor(W_np, dtype=torch.float64)
            b_new = torch.as_tensor(b_np, dtype=torch.float64)
            if W_new.shape[0] > 0:
                # warm-start computed against the current model residual
                c_new = self._coord_descent(W_new, b_new, self.data_train)
                W = torch.cat([W, W_new], dim=0)
                b = torch.cat([b, b_new], dim=0)
                c = torch.cat([c, c_new], dim=0)
                self.model.set_atoms(W, b, c)

        best_neurons = int(self.inner_weights[best_iteration_train]["weight"].shape[0])
        final_neurons = int(0 if self.model.W is None else self.model.W.shape[0])
        return {
            "alpha": self.alpha, "gamma": self.model.gamma, "power": self.model.power,
            "loss_weights": tuple(self.model.loss_weights), "activation": self.activation_fn,
            "use_sphere": self._use_sphere, "optimizer": "SSN_semiconcave",
            "model": "semiconcave",
            "num_iterations": num_iterations, "num_insertion": num_insertion,
            "threshold": threshold,
            "train_loss": list(self.train_loss), "val_loss": list(self.val_loss),
            "err_l2_train": list(self.err_l2_train), "err_l2_val": list(self.err_l2_val),
            "err_grad_train": list(self.err_grad_train), "err_grad_val": list(self.err_grad_val),
            "err_h1_train": list(self.err_h1_train), "err_h1_val": list(self.err_h1_val),
            "inner_weights": list(self.inner_weights), "outer_weights": list(self.outer_weights),
            "best_iteration": best_iteration_train,
            "best_neurons": best_neurons, "final_neurons": final_neurons,
            "best_err_l2_train": self.err_l2_train[best_iteration_train],
            "best_err_h1_train": self.err_h1_train[best_iteration_train],
            # semiconcave-specific
            "C": float(self.model.C),
        }

    # convenience: predict via the model
    def predict(self, x):
        return self.model.predict(x)
