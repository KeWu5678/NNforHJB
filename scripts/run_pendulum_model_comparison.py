#!/usr/bin/env python3
"""Compare semiconcave-profile and signed-profile PDAP on pendulum swing-up data.

This mirrors the semiconcave-reference autoresearch procedure
(``run_semiconcave_model_comparison.py``) but trains on the *sampled* pendulum
datasets instead of an analytic target:

  * PMP backward-sampler grid (infinite-horizon true value), and/or
  * transient reduced-gradient BFGS grid (finite-horizon T=3).

Key pendulum-specific adaptations:

  * Normalization. The raw value runs to O(100) over theta in [-2pi, 2pi],
    omega in [-6, 6]. Because the log-penalty grows at most ~linearly in the
    outer weights while the data loss grows quadratically in the value scale,
    leaving the data unscaled makes alpha=1e-5 effectively negligible. We map
    x -> [-1, 1]^2 and V -> ~[0, 1] and rescale dV consistently
    (dv_norm_i = (s_x_i / s_v) * dv_i) so the semiconcave hyperparameters
    transfer unchanged. Relative errors are reported in normalized space; the
    physical HJB residual un-normalizes first.

  * Train/eval split. Each dataset is a fixed point set, so we hold out a
    seeded random 20% as eval and train on the remaining 80%.

  * Metrics. rel_value / rel_grad / rel_h1 / neuron count, plus the pendulum
    stationary-HJB residual (physical units) and a data-driven switching-set
    metric (near/far gradient error around grid cells where the true gradient
    jumps). For the finite-horizon transient dataset the stationary-HJB
    residual is diagnostic only -- it is not expected to vanish.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_discontinuous_activation_experiment import ACTIVATIONS, set_seed  # noqa: E402
from src.PDAP import PDAP  # noqa: E402
from src.experiment_logging import RunRecordWriter  # noqa: E402
from src.logging_config import configure_logging  # noqa: E402
from src.models.net import ShallowNetwork  # noqa: E402
from src.paths import DATA_DIR  # noqa: E402

logger = logging.getLogger(__name__)


def relative_norm(error: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(error.reshape(-1)) / max(np.linalg.norm(reference.reshape(-1)), 1e-12))


def predict_signed_network(result, x, activation=torch.relu):
    """Rebuild a pure signed shallow network from a PDAP result and predict."""
    best_iteration = int(result["best_iteration"])
    inner = result["inner_weights"][best_iteration]
    weights, bias = inner["weight"], inner["bias"]
    outer = result["outer_weights"][best_iteration]
    n_neurons = int(weights.shape[0])
    if n_neurons == 0:
        return np.zeros((x.shape[0], 1), dtype=np.float64), np.zeros_like(x), 0
    net = ShallowNetwork(
        [x.shape[1], n_neurons, 1], activation=activation, p=float(result["power"]),
        inner_weights=weights, inner_bias=bias, outer_weights=outer,
    )
    net.eval()
    xt = torch.as_tensor(x, dtype=torch.float64).requires_grad_(True)
    value = net(xt)
    grad = torch.autograd.grad(value.sum(), xt, create_graph=False)[0]
    return value.detach().cpu().numpy(), grad.detach().cpu().numpy(), n_neurons


DEFAULT_GAMMAS = (0.0, 1e-2, 1e-1, 1.0, 10.0)
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "autoresearch"
    / "SemiconcaveFittingComparison"
    / "data: pendulum"
    / "extended_semiconcave_runs"
)
# Datasets are read from the central data directory (see src/paths.py).

# Physical pendulum constants, identical for both generators
# (PendulumPmpParameters and PendulumSwingUpProblem defaults).
PENDULUM_PHYS = {
    "control_gain": 1.0,   # c = 1 / (m l^2)
    "damping_gain": 0.1,   # d = b / (m l^2)
    "gravity_gain": 9.8,   # gamma = g / l
    "q1": 1.0,
    "q2": 1.0,
    "R": 1.0,
}

DATASETS = {
    "pmp": "PENDULUM_pmp_openloop_train_grid_80x80_512_pendulum_train_v1.npy",
    "transient": "PENDULUM_transient_openloop_real_31x31_T3_tol1e-5_workers8.npy",
}
# Whether the stationary-HJB residual is meaningful (infinite-horizon) or just
# diagnostic (finite-horizon).
HJB_IS_STATIONARY = {"pmp": True, "transient": False}
RUN_RECORD = RunRecordWriter(
    DEFAULT_OUTPUT_DIR,
    name="pendulum_model_comparison",
    id_fields=("model", "activation", "seed"),
    config_fields=(
        "model",
        "dataset",
        "activation",
        "seed",
        "num_iterations",
        "num_insertion",
        "power",
        "alpha",
    ),
    metric_field="per_gamma",
    metric_step_field="gamma",
)


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


# --------------------------------------------------------------------------- #
# Data loading, normalization, switching detection
# --------------------------------------------------------------------------- #
def load_pendulum_dataset(path: Path) -> dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    return {
        "x": np.asarray(raw["x"], dtype=np.float64).reshape(-1, 2),
        "v": np.asarray(raw["v"], dtype=np.float64).reshape(-1, 1),
        "dv": np.asarray(raw["dv"], dtype=np.float64).reshape(-1, 2),
    }


def compute_scales(x: np.ndarray, v: np.ndarray) -> dict[str, np.ndarray | float]:
    """Symmetric input scale per axis and a value scale from the train split."""
    s_x = np.maximum(np.max(np.abs(x), axis=0), 1e-12)
    s_v = float(np.maximum(np.max(np.abs(v)), 1e-12))
    return {"s_x": s_x, "s_v": s_v}


def normalize(data: dict[str, np.ndarray], scales: dict) -> dict[str, np.ndarray]:
    s_x = scales["s_x"]
    s_v = scales["s_v"]
    return {
        "x": data["x"] / s_x,
        "v": data["v"] / s_v,
        # dV transforms by chain rule: V_norm = V / s_v, x = s_x * x_norm
        # => dV_norm_i = (s_x_i / s_v) * dV_i
        "dv": data["dv"] * (s_x / s_v),
    }


def switching_distance(x_full: np.ndarray, dv_full: np.ndarray, x_query: np.ndarray,
                       k: int = 6, jump_quantile: float = 0.8) -> np.ndarray:
    """Data-driven distance to the gradient-switching set (normalized x-space).

    A point is flagged as "on the switching set" if the spread of the true
    gradient among its k nearest neighbors is in the top ``1 - jump_quantile``
    fraction. The returned array is, for each query point, the Euclidean
    distance to the nearest flagged point.
    """
    tree = cKDTree(x_full)
    kk = min(k + 1, x_full.shape[0])
    _, idx = tree.query(x_full, k=kk)
    roughness = np.empty(x_full.shape[0], dtype=np.float64)
    for i in range(x_full.shape[0]):
        neigh = idx[i][1:]  # drop self
        if neigh.size == 0:
            roughness[i] = 0.0
            continue
        roughness[i] = float(np.max(np.linalg.norm(dv_full[neigh] - dv_full[i], axis=1)))
    cut = float(np.quantile(roughness, jump_quantile))
    flagged = x_full[roughness >= cut]
    if flagged.shape[0] == 0:
        return np.full(x_query.shape[0], np.inf, dtype=np.float64)
    flagged_tree = cKDTree(flagged)
    dist, _ = flagged_tree.query(x_query, k=1)
    return np.asarray(dist, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def make_hjb_evaluator(x_phys: np.ndarray):
    """Build a stationary-HJB residual evaluator in physical units.

    0 = q1(2-2cos th) + q2 om^2 + V_th om + V_om(-d om + gamma sin th)
        - c^2 V_om^2 / (4R)
    """
    p = PENDULUM_PHYS
    theta = x_phys[:, 0]
    omega = x_phys[:, 1]
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    def residual(grad_phys: np.ndarray) -> np.ndarray:
        v_theta = grad_phys[:, 0]
        v_omega = grad_phys[:, 1]
        run = p["q1"] * (2.0 - 2.0 * cos_t) + p["q2"] * omega**2
        drift = v_theta * omega + v_omega * (-p["damping_gain"] * omega + p["gravity_gain"] * sin_t)
        control = (p["control_gain"] ** 2) * v_omega**2 / (4.0 * p["R"])
        return run + drift - control

    return residual


def pendulum_metrics(
    eval_norm: dict[str, np.ndarray],
    eval_x_phys: np.ndarray,
    scales: dict,
    sw_distance: np.ndarray,
    value_pred: np.ndarray,
    grad_pred: np.ndarray,
    hjb_residual_fn,
) -> dict[str, float]:
    value_true = eval_norm["v"]
    grad_true = eval_norm["dv"]
    s_x = scales["s_x"]
    s_v = scales["s_v"]

    out: dict[str, float] = {
        "rel_value": relative_norm(value_pred - value_true, value_true),
        "rel_grad": relative_norm(grad_pred - grad_true, grad_true),
        "rel_h1": relative_norm(
            np.concatenate([(value_pred - value_true).reshape(-1), (grad_pred - grad_true).reshape(-1)]),
            np.concatenate([value_true.reshape(-1), grad_true.reshape(-1)]),
        ),
    }

    # Physical HJB residual: un-normalize predicted gradient.
    grad_phys = grad_pred * (s_v / s_x)
    residual = np.abs(hjb_residual_fn(grad_phys))
    out["hjb_mean"] = float(np.mean(residual))
    out["hjb_max"] = float(np.max(residual))

    # Data-driven switching-set gradient error (normalized space).
    near_cut = float(np.quantile(sw_distance, 0.20))
    far_cut = float(np.quantile(sw_distance, 0.50))
    near = sw_distance <= near_cut
    far = sw_distance >= far_cut
    near_grad = np.linalg.norm((grad_pred - grad_true)[near], axis=1)
    far_grad = np.linalg.norm((grad_pred - grad_true)[far], axis=1)
    out["near_grad_mean"] = float(np.mean(near_grad)) if near_grad.size else float("nan")
    out["far_grad_mean"] = float(np.mean(far_grad)) if far_grad.size else float("nan")
    out["near_far_grad_ratio"] = out["near_grad_mean"] / max(out["far_grad_mean"], 1e-12)
    return out


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_semiconcave_profile(train_norm, eval_norm, eval_x_phys, scales, sw_distance, hjb_fn,
                              gamma, seed, args) -> dict[str, Any]:
    """Semiconcave model with profile-threshold insertion."""
    set_seed(seed)
    start = time.time()
    pdpa = PDAP(
        data=train_norm,
        alpha=args.alpha,
        gamma=gamma,
        power=args.power,
        model="semiconcave",
        insertion="profile",
        activation=args.activation_fn,
        loss_weights="h1",
        lr=args.pdap_lr,
        th=args.th,
        use_sphere=args.use_sphere,
        c_init=args.c_init,
        verbose=False,
    )
    result = pdpa.fit(
        num_iterations=args.num_iterations,
        num_insertion=args.num_insertion,
        threshold=args.threshold,
        max_insert=args.max_insert,
        verbose=False,
    )
    value_pred, grad_pred = pdpa.predict(eval_norm["x"])
    metrics = pendulum_metrics(eval_norm, eval_x_phys, scales, sw_distance,
                               value_pred, grad_pred, hjb_fn)
    neurons = int(result["final_neurons"])
    bi = int(result["best_iteration"])
    return {
        "model": "semiconcave_profile",
        "activation": args.activation_name,
        "seed": seed,
        "gamma": gamma,
        "status": "ok",
        "elapsed_s": round(time.time() - start, 3),
        "neurons": neurons,
        "score": metrics["rel_h1"] * max(neurons, 1),
        "C": float(result["C"]),
        "train_h1_final": float(result["err_h1_train"][bi]),
        "val_h1_final": float(result["err_h1_val"][bi]),
        **metrics,
    }


def train_signed_profile(train_norm, eval_norm, eval_x_phys, scales, sw_distance, hjb_fn,
                         gamma, seed, args) -> dict[str, Any]:
    set_seed(seed)
    start = time.time()
    pdpa = PDAP(
        data=train_norm,
        alpha=args.alpha,
        gamma=gamma,
        power=args.power,
        model="signed",
        insertion="profile",
        activation=args.activation_fn,
        loss_weights="h1",
        lr=args.pdap_lr,
        optimizer="SSN",
        use_sphere=args.use_sphere,
        verbose=False,
    )
    result = pdpa.fit(
        num_iterations=args.num_iterations,
        num_insertion=args.num_insertion,
        threshold=args.threshold,
        max_insert=args.max_insert,
        verbose=False,
    )
    value_pred, grad_pred, neurons = predict_signed_network(
        result, eval_norm["x"], activation=args.activation_fn
    )
    metrics = pendulum_metrics(eval_norm, eval_x_phys, scales, sw_distance,
                               value_pred, grad_pred, hjb_fn)
    best_iteration = int(result["best_iteration"])
    return {
        "model": "signed_profile",
        "activation": args.activation_name,
        "seed": seed,
        "gamma": gamma,
        "status": "ok",
        "elapsed_s": round(time.time() - start, 3),
        "neurons": int(neurons),
        "score": metrics["rel_h1"] * max(int(neurons), 1),
        "C": None,
        "train_h1_final": float(result["err_h1_train"][best_iteration]),
        "val_h1_final": float(result["err_h1_val"][best_iteration]),
        **metrics,
    }


_TRAIN_FNS = {
    "semiconcave_profile": train_semiconcave_profile,
    "signed_profile": train_signed_profile,
}


def run_model(model_name, seed, ctx, args) -> dict[str, Any]:
    rows = []
    errors = []
    start = time.time()
    for gamma in args.gammas:
        try:
            fn = _TRAIN_FNS[model_name]
            row = fn(ctx["train_norm"], ctx["eval_norm"], ctx["eval_x_phys"],
                     ctx["scales"], ctx["sw_distance"], ctx["hjb_fn"],
                     gamma, seed, args)
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            errors.append({"model": model_name, "seed": seed, "gamma": gamma, "error": repr(exc)})

    ok_rows = [r for r in rows if r["status"] == "ok" and np.isfinite(r["score"])]
    best = min(ok_rows, key=lambda r: r["score"]) if ok_rows else None
    out: dict[str, Any] = {
        "model": model_name,
        "dataset": args.dataset_name,
        "seed": seed,
        "alpha": args.alpha,
        "th": args.th,
        "power": args.power,
        "loss": "h1",
        "activation": args.activation_name,
        "use_sphere": args.use_sphere,
        "hjb_stationary": HJB_IS_STATIONARY[args.dataset_name],
        "num_iterations": args.num_iterations,
        "num_insertion": args.num_insertion,
        "threshold": args.threshold,
        "max_insert": args.max_insert,
        "elapsed_s": round(time.time() - start, 2),
        "per_gamma": rows,
        "errors": errors,
        "status": "ok" if best is not None and not errors else ("partial" if best is not None else "failed"),
    }
    if best is not None:
        out.update({
            "best_gamma": best["gamma"],
            "best_score": best["score"],
            "best_neurons": best["neurons"],
            "best_C": best["C"],
            "best_rel_value": best["rel_value"],
            "best_rel_grad": best["rel_grad"],
            "best_rel_h1": best["rel_h1"],
            "best_near_grad_mean": best["near_grad_mean"],
            "best_far_grad_mean": best["far_grad_mean"],
            "best_near_far_grad_ratio": best["near_far_grad_ratio"],
            "best_hjb_mean": best["hjb_mean"],
            "best_hjb_max": best["hjb_max"],
        })
    return out


def write_outputs(output_dir: Path, run: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    RUN_RECORD.write(run, output_dir=output_dir)


def write_summary(output_dir: Path, runs: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    header = [
        "model", "dataset", "activation", "seed", "status",
        "best_gamma", "best_score", "best_rel_h1", "best_rel_value",
        "best_rel_grad", "best_neurons", "best_C",
        "best_near_grad_mean", "best_far_grad_mean", "best_near_far_grad_ratio",
        "best_hjb_mean", "best_hjb_max", "hjb_stationary", "elapsed_s",
    ]
    with (output_dir / "results.tsv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=header, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in runs:
            writer.writerow(row)


def build_context(dataset_name: str, args: argparse.Namespace) -> dict[str, Any]:
    """Load, split (seeded 80/20), normalize, and precompute switching distance."""
    data = load_pendulum_dataset(DATA_DIR / DATASETS[dataset_name])
    n = data["x"].shape[0]
    rng = np.random.default_rng(args.split_seed)
    perm = rng.permutation(n)
    n_eval = max(1, int(round(n * args.eval_fraction)))
    eval_idx = perm[:n_eval]
    train_idx = perm[n_eval:]

    train_phys = {k: data[k][train_idx] for k in ("x", "v", "dv")}
    eval_phys = {k: data[k][eval_idx] for k in ("x", "v", "dv")}

    scales = compute_scales(train_phys["x"], train_phys["v"])
    train_norm = normalize(train_phys, scales)
    eval_norm = normalize(eval_phys, scales)

    # switching distance computed on the full normalized dataset for stability,
    # queried at the eval points.
    full_norm = normalize(data, scales)
    sw_distance = switching_distance(full_norm["x"], full_norm["dv"], eval_norm["x"])

    return {
        "train_norm": train_norm,
        "eval_norm": eval_norm,
        "eval_x_phys": eval_phys["x"],
        "scales": scales,
        "sw_distance": sw_distance,
        "hjb_fn": make_hjb_evaluator(eval_phys["x"]),
        "n_train": train_idx.size,
        "n_eval": eval_idx.size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="pmp,transient", help="comma-separated subset of pmp,transient")
    parser.add_argument("--models", default="semiconcave_profile,signed_profile",
                        help="comma-separated subset of semiconcave_profile,signed_profile")
    parser.add_argument("--activation", default="leaky_relu")
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--gammas", default=",".join(str(g) for g in DEFAULT_GAMMAS))
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--num-iterations", type=int, default=10)
    parser.add_argument("--num-insertion", type=int, default=50)
    parser.add_argument("--max-insert", type=int, default=15)
    parser.add_argument("--threshold", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=1e-5)
    parser.add_argument("--th", type=float, default=0.5)
    parser.add_argument("--power", type=float, default=1.0)
    parser.add_argument("--pdap-lr", type=float, default=1.0)
    parser.add_argument("--c-init", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    args.datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in args.datasets if d not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown_m = [m for m in args.models if m not in _TRAIN_FNS]
    if unknown_m:
        raise ValueError(f"Unknown models: {unknown_m}")
    if args.activation not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {args.activation}")
    args.seeds_list = parse_int_list(args.seeds)
    args.gammas = parse_float_list(args.gammas)
    return args


def main() -> int:
    configure_logging()
    args = parse_args()
    args.activation_name = args.activation
    args.activation_fn, args.use_sphere = ACTIVATIONS[args.activation]

    all_runs: list[dict[str, Any]] = []
    total = len(args.datasets) * len(args.models) * len(args.seeds_list)
    completed = 0
    for dataset_name in args.datasets:
        args.dataset_name = dataset_name
        ctx = build_context(dataset_name, args)
        logger.info(
            "dataset ready: dataset=%s train=%s eval=%s s_x=%s s_v=%.4g",
            dataset_name, ctx["n_train"], ctx["n_eval"], ctx["scales"]["s_x"], ctx["scales"]["s_v"],
        )
        out_dir = args.output_dir / dataset_name
        for seed in args.seeds_list:
            for model_name in args.models:
                run = run_model(model_name, seed, ctx, args)
                all_runs.append(run)
                completed += 1
                if not args.quiet:
                    print(json.dumps(run, default=str), flush=True)
                progress = (f"[{completed}/{total}] {dataset_name} {run['model']} "
                            f"seed={seed} status={run['status']}")
                if "best_rel_h1" in run:
                    progress += (f" best_h1={run['best_rel_h1']:.3e}"
                                 f" rel_grad={run['best_rel_grad']:.3e}"
                                 f" neurons={run['best_neurons']}"
                                 f" gamma={run['best_gamma']}"
                                 f" hjb_mean={run['best_hjb_mean']:.3e}")
                logger.info(progress)
                if not args.no_save:
                    write_outputs(out_dir, run)
        if not args.no_save:
            write_summary(out_dir, [r for r in all_runs if r["dataset"] == dataset_name])

    if not args.no_save:
        write_summary(args.output_dir, all_runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
