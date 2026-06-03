"""Generate pendulum swing-up samples with the repo's BVP/BB optimizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.OpenLoop.pendulum.bb_generator import PendulumBBDataGenerator  # noqa: E402
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
    parser.add_argument("--nx-theta", type=int, default=None)
    parser.add_argument("--nx-omega", type=int, default=None)
    parser.add_argument("--random-samples", type=int, default=None)
    parser.add_argument("--theta-range", type=parse_range, default=(-np.pi, np.pi))
    parser.add_argument("--omega-range", type=parse_range, default=(-8.0, 8.0))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--T-final", type=float, default=6.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--num-basis", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--max-it", type=int, default=200)
    parser.add_argument("--bvp-tol", type=float, default=1e-6)
    parser.add_argument("--terminal-weight", type=float, default=1.0)
    parser.add_argument("--line-search-cost-tol", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--control-seed-amplitudes", type=parse_seed_amplitudes, default=(0.0, 2.0, -2.0))
    parser.add_argument("--tag", default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    generator = PendulumBBDataGenerator(
        terminal_weight=args.terminal_weight,
        T_final=args.T_final,
        dt=args.dt,
        num_basis=args.num_basis,
        control_seed_amplitudes=args.control_seed_amplitudes,
        bvp_tol=args.bvp_tol,
        line_search_cost_tol=args.line_search_cost_tol,
        alpha_max=args.alpha_max,
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
        nx_theta = 3 if args.nx_theta is None else args.nx_theta
        nx_omega = 3 if args.nx_omega is None else args.nx_omega
        generator.apply_initial_gridding(
            nx_theta,
            nx_omega,
            theta_range=args.theta_range,
            omega_range=args.omega_range,
        )

    dataset, failed = generator.data_generation(
        tol=args.tol,
        max_it=args.max_it,
        verbose=args.verbose,
    )

    print(f"generated={dataset.shape[0]} failed={len(failed)}")
    if dataset.shape[0] > 0:
        finite = np.isfinite(dataset["v"])
        if np.any(finite):
            print(f"value range: {dataset['v'][finite].min():.6e} to {dataset['v'][finite].max():.6e}")

    if args.save:
        output_dir = DATA_DIR
        data_path, failed_path = generator.data_save(
            dataset,
            failed,
            output_dir=output_dir,
            tag=args.tag,
        )
        meta_path = data_path.with_name(data_path.stem + "_meta.json")
        meta = {
            "source_paper": "https://arxiv.org/abs/2312.17467",
            "method": "finite-horizon BVP/BB open-loop solve",
            "state": ["theta", "omega"],
            "horizon": args.T_final,
            "dt": args.dt,
            "num_basis": args.num_basis,
            "tol": args.tol,
            "max_it": args.max_it,
            "terminal_weight": args.terminal_weight,
            "line_search_cost_tol": args.line_search_cost_tol,
            "alpha_max": args.alpha_max,
            "control_seed_amplitudes": list(args.control_seed_amplitudes),
            "data_path": str(data_path.relative_to(REPO_ROOT)),
            "failed_path": None if failed_path is None else str(failed_path.relative_to(REPO_ROOT)),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"saved data: {data_path}")
        print(f"saved meta: {meta_path}")
        if failed_path is not None:
            print(f"saved failed: {failed_path}")


if __name__ == "__main__":
    main()
