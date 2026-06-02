"""Golden characterization tests for the PDAP outer loop.

Pins the current behaviour of the three explicit PDAP configurations.  PDAP has
internal randomness (sphere sampling + L-BFGS insertion), so we fix seeds and
assert on the summary of a short run: the final neuron count (exact) and the
error/loss scalars (tight tolerance, guarding against float reassociation while
still catching real drift).

Golden values: ``tests/fixtures/pdap_golden.npz``.  Regenerate from a known-good
tree with ``PDAP_UPDATE_GOLDEN=1 pytest tests/test_pdap_equivalence.py``.

Note: the semiconcave baseline was deliberately re-captured when its insertion
was unified onto the shared ``profile_threshold`` strategy.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from src.PDAP import PDAP

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "pdap_golden.npz"
SEED = 123
RTOL, ATOL = 1e-5, 1e-7

CONFIGS = {
    "signed_profile": {"model": "signed", "insertion": "profile"},
    "semiconcave_profile": {"model": "semiconcave", "insertion": "profile"},
    "signed_finite_step": {"model": "signed", "insertion": "finite_step"},
}


def _make_data(n: int = 60, seed: int = 0) -> dict:
    """Synthetic target V = log(1 + ||x||^2), dV = 2x/(1+||x||^2).

    Genuinely nonlinear so all three variants insert atoms non-trivially (a pure
    convex quadratic is degenerate for the semiconcave configuration: it fits with g=0).
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 2, generator=g, dtype=torch.float64) * 3 - 1.5
    r2 = (x * x).sum(1, keepdim=True)
    v = torch.log(1 + r2)
    dv = 2 * x / (1 + r2)
    return {"x": x.numpy(), "v": v.numpy(), "dv": dv.numpy()}


def _run_summary(name: str) -> np.ndarray:
    """Run one variant for a short loop; return [final_neurons, l2, h1, last_loss]."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    pdap = PDAP(
        _make_data(), alpha=1e-4, gamma=0.1, power=1.0,
        verbose=False, **CONFIGS[name],
    )
    out = pdap.fit(num_iterations=3, num_insertion=20, verbose=False)
    return np.array(
        [
            float(out["final_neurons"]),
            float(out["best_err_l2_train"]),
            float(out["best_err_h1_train"]),
            float(out["train_loss"][-1]),
        ],
        dtype=np.float64,
    )


def _maybe_regenerate() -> None:
    if GOLDEN_PATH.exists() and not os.environ.get("PDAP_UPDATE_GOLDEN"):
        return
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(GOLDEN_PATH, **{name: _run_summary(name) for name in CONFIGS})


_maybe_regenerate()


@pytest.mark.parametrize("name", list(CONFIGS))
def test_pdap_summary_matches_golden(name: str) -> None:
    golden = np.load(GOLDEN_PATH)
    got = _run_summary(name)
    # final neuron count must match exactly; error/loss scalars within tolerance.
    assert int(got[0]) == int(golden[name][0]), f"{name}: neuron count drift"
    np.testing.assert_allclose(
        got[1:], golden[name][1:], rtol=RTOL, atol=ATOL,
        err_msg=f"PDAP summary drift for variant '{name}'",
    )
