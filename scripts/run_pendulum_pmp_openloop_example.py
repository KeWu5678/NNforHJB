#!/usr/bin/env python3
"""Generate Han-Yang infinite-horizon pendulum value samples."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.OpenLoop.pendulum import (  # noqa: E402
    PendulumPmpSolver,
    PendulumPmpSolverConfig,
    PendulumSwingUpProblem,
)
from src.paths import DATA_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-trajectories", type=int, default=256)
    parser.add_argument("--epsilon", type=float, default=2e-4)
    parser.add_argument("--value-max", type=float, default=100.0)
    parser.add_argument("--t-final", type=float, default=50.0)
    parser.add_argument("--max-step", type=float, default=0.005)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--control-limit", type=float, default=None)
    parser.add_argument("--uniform-boundary-sampling", action="store_true")
    parser.add_argument("--reference-value", type=float, default=None)
    parser.add_argument("--boundary-distance-power", type=float, default=0.8)
    parser.add_argument("--contour-delta", type=float, default=1.0)
    parser.add_argument("--periodic-copies", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def progress_printer(quiet: bool):
    last_bucket = {"value": -1}

    def _progress(done: int, total: int) -> None:
        if quiet or total == 0:
            return
        bucket = int(10 * done / total)
        if bucket != last_bucket["value"]:
            last_bucket["value"] = bucket
            print(f"processed {done}/{total}")

    return _progress


def main() -> None:
    args = parse_args()
    tag = args.tag or time.strftime("%Y%m%d_%H%M%S")
    problem = PendulumSwingUpProblem(control_limit=args.control_limit)
    config = PendulumPmpSolverConfig(
        epsilon=args.epsilon,
        value_max=args.value_max,
        t_final=args.t_final,
        max_step=args.max_step,
        rtol=args.rtol,
        atol=args.atol,
        num_trajectories=args.num_trajectories,
        adaptive_sampling=not args.uniform_boundary_sampling,
        reference_value=args.reference_value,
        boundary_distance_power=args.boundary_distance_power,
        contour_delta=args.contour_delta,
        periodic_copies=args.periodic_copies,
    )
    solver = PendulumPmpSolver(problem=problem, config=config)

    started = time.perf_counter()
    solution = solver.solve(progress=progress_printer(args.quiet))
    elapsed = time.perf_counter() - started

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / f"PENDULUM_pmp_value_samples_{args.num_trajectories}_{tag}.npz"
    meta_path = args.output_dir / f"PENDULUM_pmp_value_samples_meta_{tag}.json"

    solution.value_samples.save_npz(data_path)
    metadata = {
        "source_paper": "https://arxiv.org/pdf/2312.17467",
        "description": "Backward-PMP infinite-horizon pendulum ValueSamples.",
        "data_path": str(data_path),
        "elapsed_seconds": elapsed,
        "problem": {
            "mass": problem.mass,
            "length": problem.length,
            "damping": problem.damping,
            "gravity": problem.gravity,
            "state_weights": list(problem.state_weights),
            "control_weight": problem.control_weight,
            "control_limit": problem.control_limit,
        },
        "config": config.__dict__,
        "diagnostics": solution.diagnostics.__dict__,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if not args.quiet:
        print(f"saved data: {data_path}")
        print(f"saved metadata: {meta_path}")
        print(f"samples: {solution.value_samples.size}, seconds: {elapsed:.2f}")


if __name__ == "__main__":
    main()
