#!/usr/bin/env python3
"""Run one activation-search experiment on the analytic discontinuous-gradient data.

The training data matches Experiment 3 in notebook/pdpa_vdp.ipynb. Each gamma is
trained with PDPA_v2, then the selected network is evaluated against the exact
analytic value and gradient on a dense grid. The printed JSON line is consumed
by autoresearch/ActivationSearch/data:analytical/scripts/aggregate.py.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_activation_experiment import ACTIVATIONS as BASE_ACTIVATIONS  # noqa: E402
from src.PDAP import from_alias  # noqa: E402
from src.experiment_logging import ExperimentRun  # noqa: E402
from src.net import ShallowNetwork  # noqa: E402

GAMMAS = [0, 1e-2, 1e-1, 1, 10]
ALPHA = 1e-5
POWER = 1.0
LOSS_WEIGHTS = "h1"
NUM_ITERATIONS = 10
NUM_INSERTION = 50
PRUNING_THRESHOLD = 1e-5
C_JUMP = 1.0


def gaussian_notebook(z: torch.Tensor) -> torch.Tensor:
    return torch.exp(-z.pow(2) / 2.0)


ACTIVATIONS = dict(BASE_ACTIVATIONS)
ACTIVATIONS["gaussian"] = (gaussian_notebook, False)


def write_discontinuous_activation_run_record(output_dir: Path, summary: dict[str, Any]) -> Path:
    run = ExperimentRun(
        output_dir,
        name="activation_search_analytical",
        run_id=f"{summary['activation']}_seed{summary['seed']}",
        config={"activation": summary["activation"], "seed": summary["seed"]},
    )
    return run.finish(summary=summary)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def value_and_gradient(x: np.ndarray, c_jump: float = C_JUMP) -> tuple[np.ndarray, np.ndarray]:
    x1 = x[:, 0]
    x2 = x[:, 1]
    h = x1 + x2 * np.abs(x2) / 2.0
    sign_h = np.sign(h)
    sign_h[h == 0.0] = 1.0
    value = x1**2 + x2**2 + c_jump * np.abs(h)
    grad = np.column_stack(
        [
            2.0 * x1 + c_jump * sign_h,
            2.0 * x2 + c_jump * sign_h * np.abs(x2),
        ]
    )
    return value, grad


def make_grid(grid_size: int, c_jump: float = C_JUMP) -> tuple[dict[str, np.ndarray], np.ndarray]:
    x1 = np.linspace(-2.0, 2.0, grid_size)
    x2 = np.linspace(-2.0, 2.0, grid_size)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    x = np.column_stack([x1_grid.ravel(), x2_grid.ravel()])
    value, grad = value_and_gradient(x, c_jump=c_jump)
    data = {"x": x, "v": value, "dv": grad}
    h = x[:, 0] + x[:, 1] * np.abs(x[:, 1]) / 2.0
    dist_to_curve = np.abs(h) / np.sqrt(1.0 + x[:, 1] ** 2)
    return data, dist_to_curve


def as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def build_network(result: dict[str, Any], iteration: int, activation_fn: Any) -> ShallowNetwork:
    inner = result["inner_weights"][iteration]
    weights = as_numpy(inner["weight"])
    bias = as_numpy(inner["bias"])
    outer = as_numpy(result["outer_weights"][iteration])
    n_neurons = weights.shape[0]
    net = ShallowNetwork(
        [2, n_neurons, 1],
        activation=activation_fn,
        p=result.get("power", POWER),
        inner_weights=weights,
        inner_bias=bias,
        outer_weights=outer,
    )
    net.eval()
    return net


def evaluate_network(
    net: ShallowNetwork,
    eval_data: dict[str, np.ndarray],
    dist_to_curve: np.ndarray,
) -> dict[str, float]:
    x_tensor = torch.tensor(eval_data["x"], dtype=torch.float64)
    value_true = torch.tensor(eval_data["v"], dtype=torch.float64).reshape(-1, 1)
    grad_true = torch.tensor(eval_data["dv"], dtype=torch.float64)

    x_req = x_tensor.detach().requires_grad_(True)
    with torch.enable_grad():
        pred_value = net(x_req)
        pred_grad = torch.autograd.grad(pred_value.sum(), x_req, create_graph=False)[0]

    value_diff_sq = (pred_value.detach() - value_true).pow(2).sum()
    value_true_sq = value_true.pow(2).sum().clamp_min(1e-30)
    grad_diff_by_point = (pred_grad.detach() - grad_true).pow(2).sum(dim=1)
    grad_true_by_point = grad_true.pow(2).sum(dim=1)

    grad_diff_sq = grad_diff_by_point.sum()
    grad_true_sq = grad_true_by_point.sum().clamp_min(1e-30)

    l2 = torch.sqrt(value_diff_sq / value_true_sq)
    grad = torch.sqrt(grad_diff_sq / grad_true_sq)
    h1 = torch.sqrt((value_diff_sq + grad_diff_sq) / (value_true_sq + grad_true_sq))

    near_mask = torch.tensor(dist_to_curve < np.percentile(dist_to_curve, 20))
    far_mask = torch.tensor(dist_to_curve > np.percentile(dist_to_curve, 50))

    def regional_grad(mask: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(
            grad_diff_by_point[mask].sum()
            / grad_true_by_point[mask].sum().clamp_min(1e-30)
        )

    near_grad = regional_grad(near_mask)
    far_grad = regional_grad(far_mask)
    return {
        "eval_l2": float(l2),
        "eval_grad": float(grad),
        "eval_h1": float(h1),
        "near_grad": float(near_grad),
        "far_grad": float(far_grad),
        "near_far_ratio": float(near_grad / far_grad.clamp_min(1e-30)),
        "near_points": int(near_mask.sum()),
        "far_points": int(far_mask.sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", required=True, choices=sorted(ACTIVATIONS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-iterations", type=int, default=NUM_ITERATIONS)
    parser.add_argument("--num-insertion", type=int, default=NUM_INSERTION)
    parser.add_argument("--train-grid-size", type=int, default=30)
    parser.add_argument("--eval-grid-size", type=int, default=61)
    parser.add_argument("--c-jump", type=float, default=C_JUMP)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    activation_fn, use_sphere = ACTIVATIONS[args.activation]
    train_data, _ = make_grid(args.train_grid_size, c_jump=args.c_jump)
    eval_data, eval_dist = make_grid(args.eval_grid_size, c_jump=args.c_jump)
    run = None
    if args.output_dir is not None:
        run = ExperimentRun(
            args.output_dir,
            name="activation_search_analytical",
            run_id=f"{args.activation}_seed{args.seed}",
            config={
                "activation": args.activation,
                "seed": args.seed,
                "num_iterations": args.num_iterations,
                "num_insertion": args.num_insertion,
                "train_grid_size": args.train_grid_size,
                "eval_grid_size": args.eval_grid_size,
                "c_jump": args.c_jump,
            },
        )

    per_gamma = []
    start = time.time()
    for gamma in GAMMAS:
        set_seed(args.seed)
        pdpa = from_alias(
            "v2",
            data=train_data,
            alpha=ALPHA,
            gamma=gamma,
            power=POWER,
            activation=activation_fn,
            use_sphere=use_sphere,
            loss_weights=LOSS_WEIGHTS,
            verbose=False,
        )
        result = pdpa.fit(
            num_iterations=args.num_iterations,
            num_insertion=args.num_insertion,
            threshold=PRUNING_THRESHOLD,
            verbose=False,
        )
        best_iteration = int(result["best_iteration"])
        net = build_network(result, best_iteration, activation_fn)
        metrics = evaluate_network(net, eval_data, eval_dist)
        n_neurons = int(result["inner_weights"][best_iteration]["weight"].shape[0])
        train_h1 = float(result["err_h1_train"][best_iteration])
        val_h1 = float(result["err_h1_val"][best_iteration])
        score = metrics["eval_h1"] * n_neurons
        per_gamma.append(
            {
                "gamma": gamma,
                "score": score,
                "n": n_neurons,
                "best_iteration": best_iteration,
                "train_h1": train_h1,
                "val_h1": val_h1,
                **metrics,
            }
        )

    best = min(per_gamma, key=lambda row: row["score"])
    out = {
        "activation": args.activation,
        "seed": args.seed,
        "power": POWER,
        "loss": LOSS_WEIGHTS,
        "use_sphere": use_sphere,
        "c_jump": args.c_jump,
        "train_grid_size": args.train_grid_size,
        "eval_grid_size": args.eval_grid_size,
        "elapsed_s": round(time.time() - start, 2),
        "per_gamma": per_gamma,
        "best_gamma": best["gamma"],
        "best_score": best["score"],
        "best_eval_h1": best["eval_h1"],
        "best_eval_l2": best["eval_l2"],
        "best_eval_grad": best["eval_grad"],
        "best_near_grad": best["near_grad"],
        "best_far_grad": best["far_grad"],
        "best_near_far_ratio": best["near_far_ratio"],
        "best_train_h1": best["train_h1"],
        "best_val_h1": best["val_h1"],
        "best_n": best["n"],
    }
    if run is not None:
        path = run.finish(summary=out)
        print(f"saved run record: {path}", file=sys.stderr, flush=True)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
