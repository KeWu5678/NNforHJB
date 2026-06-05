"""Golden characterization tests locking the behaviour of the SSN optimizers.

These tests pin the *current* step-by-step iterates of every SSN variant so the
planned merge (SSN + SSN_TR + SSN_semiconcave -> one configurable ``SSN``) can be
verified to leave behaviour bit-for-bit unchanged.  Each phase of the refactor
must keep these green; the construction calls may be re-pointed to the new API
as classes are folded together, but the golden trajectories (the invariant) do
not change.

Golden values live in ``tests/fixtures/ssn_golden.npz``.  Regenerate them only
from a known-good tree:  ``SSN_UPDATE_GOLDEN=1 pytest tests/test_ssn_equivalence.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from src.SSN import SSN
from src.SSN.penalty import _phi

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "ssn_golden.npz"
N_STEPS = 6
# Tolerance chosen to admit floating-point *reassociation* noise while still
# catching real behavioural drift.  The merge replaced scalar ``alpha/c`` with a
# per-coordinate ``alpha_vec/c`` tensor in the proximal arithmetic; for q=1 this
# reorders a few float ops and seeds a ~1e-9 difference at step 0 that *decays to
# machine epsilon* as the iterate converges to the identical fixed point (q!=1
# stays bit-identical).  A genuine regression is ~1e-4 (cf. the alpha-drift
# sanity check), six orders above this noise floor.
RTOL, ATOL = 1e-6, 1e-7


def _make_quadratic(P: int, Nx: int, seed: int):
    """Deterministic least-squares problem: A (Nx,P), y (Nx,), H = (1/Nx) A^T A."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(Nx, P, generator=g, dtype=torch.float64)
    theta_true = torch.randn(P, generator=g, dtype=torch.float64)
    y = A @ theta_true + 0.01 * torch.randn(Nx, generator=g, dtype=torch.float64)
    H = (1.0 / Nx) * (A.T @ A)
    return A, y, H


def _penalty(theta, mask, alpha, th, gamma, q):
    """alpha * sum_{penalised} phi(|theta|^q) with nonneg clamp, matching closures."""
    sel = theta[mask]
    sel = sel.clamp_min(0.0)
    arg = sel if q == 1.0 else sel.clamp_min(1e-30) ** q
    return alpha * torch.sum(_phi(arg, th, gamma))


def _run_trajectory(config: dict) -> np.ndarray:
    """Run one optimizer config for N_STEPS, returning the (N_STEPS, P) theta trace."""
    torch.manual_seed(0)
    P, Nx = config["P"], config["Nx"]
    alpha, gamma, th = config["alpha"], config["gamma"], config["th"]
    power = config["power"]
    q = 2.0 / (power + 1.0)

    A, y, H = _make_quadratic(P, Nx, seed=config["seed"])
    theta = torch.nn.Parameter(config["theta0"].clone())

    pen_mask = config.get("penalized_mask", torch.ones(P, dtype=torch.bool))

    kind = config["kind"]
    if kind == "ssn":
        opt = SSN([theta], alpha=alpha, gamma=gamma, th=th, lr=1.0, power=power)
    elif kind == "ssn_tr":
        # folded into the base SSN: trust-region (Steihaug-CG) globalization
        opt = SSN([theta], alpha=alpha, gamma=gamma, th=th, lr=1.0, power=power,
                  method="steihaug_cg")
    elif kind == "ssn_semiconcave":
        # folded into the base SSN: masked-penalty + nonnegative-prox configuration
        opt = SSN(
            [theta], alpha=alpha, gamma=gamma, th=th, lr=1.0, power=power,
            penalized_mask=pen_mask, nonneg_mask=config["nonneg_mask"],
        )
    else:  # pragma: no cover
        raise ValueError(kind)
    opt.data_hessian = H

    def closure():
        opt.zero_grad()
        r = A @ theta.reshape(-1) - y
        data = 0.5 / Nx * (r @ r)
        return data + _penalty(theta.reshape(-1), pen_mask, alpha, th, gamma, q)

    traj = np.zeros((N_STEPS, P), dtype=np.float64)
    for k in range(N_STEPS):
        opt.step(closure)
        traj[k] = theta.detach().reshape(-1).numpy()
    return traj


def _configs() -> dict[str, dict]:
    t4 = torch.tensor([0.5, -0.3, 0.4, 0.2], dtype=torch.float64)
    return {
        "ssn_signed_q1": dict(
            kind="ssn", P=4, Nx=20, alpha=1e-2, gamma=1.0, th=0.5, power=1.0,
            seed=1, theta0=t4,
        ),
        "ssn_signed_q_half": dict(
            kind="ssn", P=4, Nx=20, alpha=1e-2, gamma=1.0, th=0.5, power=3.0,
            seed=2, theta0=t4,
        ),
        "ssn_tr_signed": dict(
            kind="ssn_tr", P=4, Nx=20, alpha=1e-2, gamma=1.0, th=0.5, power=1.0,
            seed=3, theta0=t4,
        ),
        "ssn_semiconcave_free_coord": dict(
            kind="ssn_semiconcave", P=4, Nx=20, alpha=1e-2, gamma=1.0, th=0.5,
            power=1.0, seed=4, theta0=torch.tensor([0.5, 0.5, 0.3, 0.0], dtype=torch.float64),
            penalized_mask=torch.tensor([True, True, False, False]),
            nonneg_mask=torch.tensor([True, True, True, False]),
        ),
    }


CONFIGS = _configs()


def _maybe_regenerate() -> None:
    if GOLDEN_PATH.exists() and not os.environ.get("SSN_UPDATE_GOLDEN"):
        return
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(GOLDEN_PATH, **{name: _run_trajectory(cfg) for name, cfg in CONFIGS.items()})


_maybe_regenerate()


@pytest.mark.parametrize("name", list(CONFIGS))
def test_ssn_trajectory_matches_golden(name: str) -> None:
    golden = np.load(GOLDEN_PATH)
    np.testing.assert_allclose(
        _run_trajectory(CONFIGS[name]), golden[name], rtol=RTOL, atol=ATOL,
        err_msg=f"SSN trajectory drift for config '{name}'",
    )
