#!/usr/bin/env python3
"""Generate the Han-Yang inverted-pendulum open-loop value samples.

This follows the released MATLAB construction for arXiv:2312.17467:
compute a local LQR value near the upright equilibrium, sample the local LQR
level-set boundary, and integrate the PMP system backward to obtain x, V(x),
and dV(x) samples.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.OpenLoop.pendulum.pmp_sampler import (  # noqa: E402
    PendulumPmpParameters,
    PendulumPmpSampler,
)
from src.paths import DATA_DIR, PLOTS_DIR  # noqa: E402


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
    parser.add_argument(
        "--angle-sampling",
        choices=("uniform", "adaptive"),
        default="uniform",
    )
    parser.add_argument("--refinement-value", type=float, default=20.0)
    parser.add_argument("--boundary-distance-power", type=float, default=0.8)
    parser.add_argument("--periodic-copies", type=int, default=1)
    parser.add_argument("--theta-range", type=float, nargs=2, default=(-8.0, 8.0))
    parser.add_argument("--omega-range", type=float, nargs=2, default=(-8.0, 8.0))
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--training-grid-size", type=int, nargs=2, default=(80, 80))
    parser.add_argument("--samples-per-cell", type=int, default=1)
    parser.add_argument(
        "--cell-selection",
        choices=("min-value", "center"),
        default="min-value",
    )
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--plot-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-subsample", type=int, default=40000)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def select_training_grid_samples(
    data: np.ndarray,
    theta_range: tuple[float, float],
    omega_range: tuple[float, float],
    grid_size: tuple[int, int],
    samples_per_cell: int,
    cell_selection: str = "min-value",
) -> tuple[np.ndarray, dict[str, object]]:
    """Select representative PMP samples on a rectangular coverage grid."""
    nx, ny = (int(grid_size[0]), int(grid_size[1]))
    if nx <= 0 or ny <= 0:
        raise ValueError("training grid dimensions must be positive")
    if samples_per_cell <= 0:
        raise ValueError("samples_per_cell must be positive")
    if cell_selection not in {"min-value", "center"}:
        raise ValueError("cell_selection must be 'min-value' or 'center'")

    theta_min, theta_max = theta_range
    omega_min, omega_max = omega_range
    if theta_max <= theta_min or omega_max <= omega_min:
        raise ValueError("training ranges must be increasing")

    theta = data["x"][:, 0]
    omega = data["x"][:, 1]
    theta_width = (theta_max - theta_min) / nx
    omega_width = (omega_max - omega_min) / ny
    ix = np.floor((theta - theta_min) / theta_width).astype(np.int64)
    iy = np.floor((omega - omega_min) / omega_width).astype(np.int64)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    if not np.any(valid):
        return data[:0], {
            "grid_size": [nx, ny],
            "theta_range": [theta_min, theta_max],
            "omega_range": [omega_min, omega_max],
            "samples_per_cell": samples_per_cell,
            "cell_selection": cell_selection,
            "occupied_cells": 0,
            "total_cells": nx * ny,
            "occupancy_fraction": 0.0,
        }

    valid_indices = np.flatnonzero(valid)
    ix_valid = ix[valid]
    iy_valid = iy[valid]
    cell_ids = iy_valid * nx + ix_valid
    theta_center = theta_min + (ix_valid + 0.5) * theta_width
    omega_center = omega_min + (iy_valid + 0.5) * omega_width
    dist2 = (theta[valid] - theta_center) ** 2 + (omega[valid] - omega_center) ** 2

    if cell_selection == "min-value":
        order = np.lexsort((dist2, data["v"][valid], cell_ids))
    else:
        order = np.lexsort((dist2, cell_ids))
    sorted_cells = cell_ids[order]
    _cells, starts, counts = np.unique(
        sorted_cells,
        return_index=True,
        return_counts=True,
    )

    chosen_chunks = [
        order[start : start + min(samples_per_cell, count)]
        for start, count in zip(starts, counts)
    ]
    chosen_local = np.concatenate(chosen_chunks)
    chosen_indices = valid_indices[chosen_local]
    selected = data[chosen_indices]

    metadata = {
        "grid_size": [nx, ny],
        "theta_range": [theta_min, theta_max],
        "omega_range": [omega_min, omega_max],
        "samples_per_cell": samples_per_cell,
        "cell_selection": cell_selection,
        "occupied_cells": int(_cells.size),
        "total_cells": int(nx * ny),
        "occupancy_fraction": float(_cells.size / (nx * ny)),
        "num_selected": int(selected.shape[0]),
    }
    return selected, metadata


def progress_printer(quiet: bool):
    last_bucket = {"value": -1}

    def _progress(done: int, total: int) -> None:
        if quiet or total == 0:
            return
        bucket = int(10 * done / total)
        if bucket != last_bucket["value"]:
            last_bucket["value"] = bucket
            print(f"sampled {done}/{total} trajectories")

    return _progress


def save_plots(data: np.ndarray, plot_dir: Path, tag: str, max_points: int) -> list[str]:
    cache_dir = Path(tempfile.gettempdir()) / "sparsenn_hjb_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    if data.shape[0] > max_points:
        rng = np.random.default_rng(20260528)
        index = rng.choice(data.shape[0], size=max_points, replace=False)
        plot_data = data[index]
    else:
        plot_data = data

    paths: list[str] = []

    scatter_path = plot_dir / f"PENDULUM_pmp_openloop_scatter_{tag}.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        plot_data["x"][:, 0],
        plot_data["x"][:, 1],
        c=plot_data["v"],
        s=2,
        cmap="viridis",
        linewidths=0,
    )
    ax.set_xlabel("theta")
    ax.set_ylabel("omega")
    ax.set_title("Pendulum PMP value samples")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax, label="V")
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=200)
    plt.close(fig)
    paths.append(str(scatter_path))

    arrows_path = plot_dir / f"PENDULUM_pmp_openloop_gradient_{tag}.png"
    arrow_stride = max(1, plot_data.shape[0] // 3000)
    arrow_data = plot_data[::arrow_stride]
    grad = arrow_data["dv"]
    grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
    grad_unit = grad / np.maximum(grad_norm, 1e-12)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        plot_data["x"][:, 0],
        plot_data["x"][:, 1],
        c=plot_data["v"],
        s=1,
        cmap="viridis",
        linewidths=0,
        alpha=0.45,
    )
    ax.quiver(
        arrow_data["x"][:, 0],
        arrow_data["x"][:, 1],
        grad_unit[:, 0],
        grad_unit[:, 1],
        color="black",
        width=0.0015,
        scale=40,
        alpha=0.7,
    )
    ax.set_xlabel("theta")
    ax.set_ylabel("omega")
    ax.set_title("Pendulum PMP gradient directions")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(arrows_path, dpi=200)
    plt.close(fig)
    paths.append(str(arrows_path))

    return paths


def main() -> None:
    args = parse_args()
    tag = args.tag or time.strftime("%Y%m%d_%H%M%S")

    parameters = PendulumPmpParameters(
        epsilon=args.epsilon,
        value_max=args.value_max,
        t_final=args.t_final,
        max_step=args.max_step,
        rtol=args.rtol,
        atol=args.atol,
        control_limit=args.control_limit,
    )
    sampler = PendulumPmpSampler(parameters)

    started = time.perf_counter()
    angles = None
    refinement_value_used = None
    if args.angle_sampling == "adaptive":
        refinement_value_used = min(args.refinement_value, args.value_max)
        angles = sampler.adaptive_angles(
            args.num_trajectories,
            reference_value=refinement_value_used,
            boundary_distance_power=args.boundary_distance_power,
            progress=progress_printer(args.quiet),
        )

    trajectories, failures = sampler.sample_trajectories(
        args.num_trajectories,
        angles=angles,
        progress=progress_printer(args.quiet),
        skip_failures=True,
    )
    elapsed = time.perf_counter() - started

    theta_range = None if args.no_filter else tuple(float(v) for v in args.theta_range)
    omega_range = None if args.no_filter else tuple(float(v) for v in args.omega_range)
    data = sampler.trajectories_to_dataset(
        trajectories,
        periodic_copies=args.periodic_copies,
        theta_range=theta_range,
        omega_range=omega_range,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / f"PENDULUM_pmp_openloop_{args.num_trajectories}_{tag}.npy"
    train_path = (
        args.output_dir
        / (
            "PENDULUM_pmp_openloop_train_grid_"
            f"{args.training_grid_size[0]}x{args.training_grid_size[1]}_"
            f"{args.num_trajectories}_{tag}.npy"
        )
    )
    meta_path = args.output_dir / f"PENDULUM_pmp_openloop_meta_{tag}.json"
    failed_path = args.output_dir / f"PENDULUM_pmp_openloop_failed_{tag}.json"

    np.save(data_path, data)
    training_data = np.zeros(0, dtype=data.dtype)
    training_grid_metadata = None
    if data.shape[0] > 0:
        if theta_range is None:
            theta_for_grid = (float(np.min(data["x"][:, 0])), float(np.max(data["x"][:, 0])))
        else:
            theta_for_grid = theta_range
        if omega_range is None:
            omega_for_grid = (float(np.min(data["x"][:, 1])), float(np.max(data["x"][:, 1])))
        else:
            omega_for_grid = omega_range
        training_data, training_grid_metadata = select_training_grid_samples(
            data,
            theta_for_grid,
            omega_for_grid,
            tuple(args.training_grid_size),
            args.samples_per_cell,
            cell_selection=args.cell_selection,
        )
    np.save(train_path, training_data)

    with failed_path.open("w", encoding="utf-8") as handle:
        json.dump(failures, handle, indent=2)

    plot_paths: list[str] = []
    if not args.no_plot and data.shape[0] > 0:
        plot_paths = save_plots(data, args.plot_dir, tag, args.plot_subsample)

    metadata = {
        "source_paper": "https://arxiv.org/pdf/2312.17467",
        "source_code": "https://github.com/ComputationalRobotics/InvertedPendulumOptimalValue",
        "description": "Backward-PMP inverted-pendulum open-loop samples.",
        "num_trajectories_requested": args.num_trajectories,
        "num_trajectories_saved": len(trajectories),
        "num_failed": len(failures),
        "num_samples": int(data.shape[0]),
        "angle_sampling": args.angle_sampling,
        "refinement_value": args.refinement_value,
        "refinement_value_used": refinement_value_used,
        "boundary_distance_power": args.boundary_distance_power,
        "cell_selection": args.cell_selection,
        "periodic_copies": args.periodic_copies,
        "theta_range": None if theta_range is None else list(theta_range),
        "omega_range": None if omega_range is None else list(omega_range),
        "elapsed_seconds": elapsed,
        "parameters": parameters.__dict__,
        "local_lqr_matrix": sampler.local_lqr_matrix.tolist(),
        "hit_value_event_count": int(sum(t.hit_value_event for t in trajectories)),
        "data_path": str(data_path),
        "training_data_path": str(train_path),
        "training_num_samples": int(training_data.shape[0]),
        "training_grid": training_grid_metadata,
        "failed_path": str(failed_path),
        "plot_paths": plot_paths,
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    if not args.quiet:
        print(f"saved data: {data_path}")
        print(f"saved training data: {train_path}")
        print(f"saved metadata: {meta_path}")
        print(f"saved failures: {failed_path}")
        for path in plot_paths:
            print(f"saved plot: {path}")
        print(f"samples: {data.shape[0]}, failures: {len(failures)}, seconds: {elapsed:.2f}")


if __name__ == "__main__":
    main()
