#!/usr/bin/env python3
"""[DEBUG] Fast feedback loop for the branch-restriction fix (issue #18).

The slow part of the pendulum solve is PMP integration (~200 s for 256
trajectories); the geometry under test (nonsmooth curve + restriction) is cheap.
So we integrate the raw trajectories ONCE, pickle them, and then re-run only the
geometry on every iteration -> sub-second loop.

Pass/fail signal: with the buggy value-ordered-polyline cut, retained samples cap
at value ~1.13 and sit 0.56+ away from the ridge. The fix should let retained
samples extend out toward the switching set (max value >> 1, near-ridge samples
present). This harness prints those numbers so each geometry change is graded.

    python scripts/investigation/diagnose_restriction_loop.py          # run loop
    python scripts/investigation/diagnose_restriction_loop.py --regen   # rebuild cache
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.OpenLoop.pendulum.problem import PendulumSwingUpProblem  # noqa: E402
from src.OpenLoop.pendulum.solver import PendulumPmpSolver, PendulumPmpSolverConfig  # noqa: E402

CACHE = REPO_ROOT / "rawdata" / "data" / "_debug_raw_trajectories_256.pkl"


def get_raw_trajectories(regen: bool):
    """Integrate the 256 raw PMP trajectories once; cache to disk."""
    if CACHE.exists() and not regen:
        with CACHE.open("rb") as f:
            raw = pickle.load(f)
        print(f"[DEBUG] loaded {len(raw)} cached raw trajectories from {CACHE.name}")
        return raw
    print("[DEBUG] integrating raw trajectories (~200 s, one-time)...")
    solver = _solver()
    started = time.perf_counter()
    raw, failures = solver.generate_raw_trajectories()
    print(f"[DEBUG] integrated {len(raw)} trajectories "
          f"({len(failures)} failed) in {time.perf_counter()-started:.1f}s")
    with CACHE.open("wb") as f:
        pickle.dump(raw, f)
    return raw


def _solver() -> PendulumPmpSolver:
    return PendulumPmpSolver(
        problem=PendulumSwingUpProblem(control_limit=None),
        config=PendulumPmpSolverConfig(
            epsilon=2e-4, value_max=100.0, t_final=50.0, max_step=0.005,
            rtol=1e-10, atol=1e-12, num_trajectories=256,
        ),
    )


def grade(raw) -> None:
    """Run the current geometry on cached trajectories and grade the result."""
    solver = _solver()
    started = time.perf_counter()
    curve = solver.compute_nonsmooth_curve(raw)
    samples, restricted, discarded = solver.build_value_samples(raw, curve)
    elapsed = time.perf_counter() - started

    x, v = samples.x, samples.v
    print(f"\n[DEBUG] geometry ran in {elapsed:.3f}s")
    print(f"  ridge points         : {curve.points.shape[0]}")
    print(f"  retained / discarded : {samples.size} / {discarded}")
    print(f"  retained value max   : {v.max():.3f}   (BUG signal: ~1.13; want >> 20)")
    print(f"  retained |x| max     : {np.linalg.norm(x, axis=1).max():.3f}")
    print(f"  theta range          : [{x[:,0].min():.2f}, {x[:,0].max():.2f}]")
    print(f"  omega range          : [{x[:,1].min():.2f}, {x[:,1].max():.2f}]")
    if curve.points.shape[0] and samples.size:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(curve.points).query(x, k=1)
        print(f"  min dist to ridge    : {d.min():.3f}   (BUG signal: ~0.56; want ~0)")
        print(f"  PASS: retained reaches arms? {v.max() > 20.0 and d.min() < 0.5}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regen", action="store_true", help="rebuild trajectory cache")
    args = parser.parse_args()
    raw = get_raw_trajectories(args.regen)
    grade(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
