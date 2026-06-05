"""Generate pendulum finite-horizon training data with reduced-gradient OCP solves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.OpenLoop.pendulum.finite_horizon_generator import (  # noqa: E402
    PendulumFiniteHorizonDataGenerator,
)
from src.paths import DATA_DIR  # noqa: E402


def parse_point(raw: str) -> np.ndarray:
    values = [float(part) for part in raw.split(",")]
    if len(values) != 2:
        raise argparse.ArgumentTypeError("points must have form theta,omega")
    return np.array(values, dtype=float)


def parse_range(raw: str) -> tuple[float, float]:
    values = [float(part) for part in raw.split(",")]
    if len(values) != 2:
        raise argparse.ArgumentTypeError("ranges must have form lo,hi")
    return values[0], values[1]


def parse_seed_amplitudes(raw: str) -> tuple[float, ...]:
    if not raw:
        return (0.0,)
    return tuple(float(part) for part in raw.split(","))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", nargs="*", type=parse_point)
    parser.add_argument("--nx-theta", type=int, default=7)
    parser.add_argument("--nx-omega", type=int, default=7)
    parser.add_argument("--random-samples", type=int, default=None)
    parser.add_argument("--theta-range", type=parse_range, default=(-np.pi, np.pi))
    parser.add_argument("--omega-range", type=parse_range, default=(-4.0, 4.0))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--T-final", type=float, default=3.0)
    parser.add_argument("--terminal-weight", type=float, default=1.0)
    parser.add_argument("--num-basis", type=int, default=20)
    parser.add_argument("--time-steps", type=int, default=301)
    parser.add_argument("--stationarity-tol", type=float, default=1e-5)
    parser.add_argument("--coefficient-gradient-tol", type=float, default=1e-5)
    parser.add_argument("--optimizer-method", default="BFGS")
    parser.add_argument("--max-it", type=int, default=500)
    parser.add_argument("--ivp-rtol", type=float, default=1e-8)
    parser.add_argument("--ivp-atol", type=float, default=1e-10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument(
        "--control-seed-amplitudes",
        type=parse_seed_amplitudes,
        default=(0.0, 2.0, -2.0, 5.0, -5.0, 10.0, -10.0),
    )
    parser.add_argument("--tag", default=None)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    generator = PendulumFiniteHorizonDataGenerator(
        T_final=args.T_final,
        terminal_weight=args.terminal_weight,
        num_basis=args.num_basis,
        time_steps=args.time_steps,
        stationarity_tol=args.stationarity_tol,
        coefficient_gradient_tol=args.coefficient_gradient_tol,
        optimizer_method=args.optimizer_method,
        max_iter=args.max_it,
        ivp_rtol=args.ivp_rtol,
        ivp_atol=args.ivp_atol,
        control_seed_amplitudes=args.control_seed_amplitudes,
    )

    if args.points:
        generator.x0_values = np.vstack(args.points)
    elif args.random_samples is not None:
        generator.apply_random_initial_sampling(
            args.random_samples,
            theta_range=args.theta_range,
            omega_range=args.omega_range,
            seed=args.seed,
        )
    else:
        generator.apply_initial_gridding(
            args.nx_theta,
            args.nx_omega,
            theta_range=args.theta_range,
            omega_range=args.omega_range,
        )

    checkpoint_dir = None
    checkpoint_tag = None
    if args.checkpoint_every > 0:
        checkpoint_dir = DATA_DIR
        checkpoint_tag = args.tag or "pendulum_transient"

    dataset, failed, diagnostics = generator.data_generation_parallel(
        args.workers,
        checkpoint_dir=checkpoint_dir,
        checkpoint_tag=checkpoint_tag,
        checkpoint_every=max(args.checkpoint_every, 1),
    )
    print(f"accepted={dataset.shape[0]} failed={len(failed)} attempts={len(diagnostics)}")
    if dataset.shape[0] > 0:
        print(f"value range: {dataset['v'].min():.6e} to {dataset['v'].max():.6e}")

    accepted_diagnostics = [row for row in diagnostics if row["accepted"]]
    if accepted_diagnostics:
        max_residual = max(row["residual_l2_squared"] for row in accepted_diagnostics)
        print(f"max accepted residual_l2_squared: {max_residual:.6e}")

    if args.save:
        output_dir = DATA_DIR
        data_path, failed_path, diagnostics_path = generator.data_save(
            dataset,
            failed,
            diagnostics,
            output_dir=output_dir,
            tag=args.tag,
        )
        meta_path = data_path.with_name(data_path.stem + "_meta.json")
        meta = {
            "method": "forward-backward reduced-gradient open-loop solve",
            "optimizer_method": args.optimizer_method,
            "state": ["theta", "omega"],
            "source_paper": "https://arxiv.org/abs/2312.17467",
            "T_final": args.T_final,
            "terminal_weight": args.terminal_weight,
            "num_basis": args.num_basis,
            "time_steps": args.time_steps,
            "stationarity_tol": args.stationarity_tol,
            "coefficient_gradient_tol": args.coefficient_gradient_tol,
            "max_it": args.max_it,
            "workers": args.workers,
            "checkpoint_every": args.checkpoint_every,
            "ivp_rtol": args.ivp_rtol,
            "ivp_atol": args.ivp_atol,
            "control_seed_amplitudes": list(args.control_seed_amplitudes),
            "accepted": int(dataset.shape[0]),
            "failed": int(len(failed)),
            "data_path": str(data_path.relative_to(REPO_ROOT)),
            "failed_path": str(failed_path.relative_to(REPO_ROOT)),
            "diagnostics_path": str(diagnostics_path.relative_to(REPO_ROOT)),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"saved data: {data_path}")
        print(f"saved failed: {failed_path}")
        print(f"saved diagnostics: {diagnostics_path}")
        print(f"saved meta: {meta_path}")


if __name__ == "__main__":
    main()
