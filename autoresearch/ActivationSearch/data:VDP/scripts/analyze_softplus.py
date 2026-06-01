#!/usr/bin/env python3
"""Analyze selected VDP activation reruns for softplus diagnostics."""
from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
ANALYSIS.mkdir(exist_ok=True)


def mean(values: list[float]) -> float:
    return float(np.mean(values))


def std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def family_name(name: str) -> str:
    if name.startswith("softplus_b"):
        return "Softplus"
    if name.startswith("gelu_b"):
        return "GELU"
    if name.startswith("mish_b"):
        return "Mish"
    if name.startswith("smoothy_relu"):
        return "SmoothReLU"
    if name == "relu":
        return "ReLU"
    return name


def variant(name: str) -> str:
    match = re.search(r"_b([0-9_]+)$", name)
    if match:
        return "beta=" + match.group(1).replace("_", ".")
    match = re.search(r"_w(.+)$", name)
    if match:
        return "w=" + match.group(1).replace("_", ".")
    return "baseline"


def load_runs() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(RUNS.glob("*_seed*.json")):
        try:
            row = json.loads(path.read_text().strip().splitlines()[-1])
        except Exception:
            continue
        if "per_gamma" in row:
            out[row["activation"]].append(row)
    return out


def write_family_tables(runs: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for act, act_runs in sorted(runs.items()):
        scores = [float(row["best_score"]) for row in act_runs]
        h1s = [float(row["best_h1"]) for row in act_runs]
        ns = [float(row["best_n"]) for row in act_runs]
        rows.append(
            {
                "activation": act,
                "family": family_name(act),
                "variant": variant(act),
                "seeds": ",".join(str(row["seed"]) for row in sorted(act_runs, key=lambda x: x["seed"])),
                "mean_score": mean(scores),
                "std_score": std(scores),
                "mean_h1": mean(h1s),
                "std_h1": std(h1s),
                "mean_neurons": mean(ns),
                "std_neurons": std(ns),
            }
        )

    with (ANALYSIS / "selected_activation_summary.tsv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    reps: dict[str, dict] = {}
    for row in rows:
        family = row["family"]
        if family not in reps or row["mean_score"] < reps[family]["mean_score"]:
            reps[family] = row
    family_rows = sorted(reps.values(), key=lambda row: row["mean_score"])
    with (ANALYSIS / "family_representatives.tsv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(family_rows[0]))
        writer.writeheader()
        writer.writerows(family_rows)
    return rows


def write_gamma_tables(runs: dict[str, list[dict]]) -> None:
    gamma_rows = []
    for act, act_runs in sorted(runs.items()):
        by_gamma: dict[float, list[dict]] = defaultdict(list)
        for run in act_runs:
            for row in run["per_gamma"]:
                by_gamma[float(row["gamma"])].append(row)
        for gamma, rows in sorted(by_gamma.items()):
            h1 = [float(row["h1"]) for row in rows]
            n = [float(row["n"]) for row in rows]
            score = [float(row["score"]) for row in rows]
            gamma_rows.append(
                {
                    "activation": act,
                    "family": family_name(act),
                    "variant": variant(act),
                    "gamma": gamma,
                    "mean_h1": mean(h1),
                    "std_h1": std(h1),
                    "mean_neurons": mean(n),
                    "std_neurons": std(n),
                    "mean_score": mean(score),
                    "std_score": std(score),
                }
            )
    with (ANALYSIS / "gamma_sweep.tsv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(gamma_rows[0]))
        writer.writeheader()
        writer.writerows(gamma_rows)


def activation_fn(name: str):
    if name.startswith("softplus_b"):
        beta = float(name.removeprefix("softplus_b").replace("_", "."))
        return lambda x: torch.nn.functional.softplus(x, beta=beta)
    if name.startswith("gelu_b"):
        beta = float(name.removeprefix("gelu_b").replace("_", "."))
        return lambda x: 0.5 * x * (1.0 + torch.erf(beta * x / math.sqrt(2.0)))
    if name.startswith("mish_b"):
        beta = float(name.removeprefix("mish_b").replace("_", "."))
        return lambda x: x * torch.tanh(torch.nn.functional.softplus(x, beta=beta))
    if name == "relu":
        return torch.relu
    if name == "smoothy_relu_w0_25":
        return lambda x: torch.where(
            x >= 0.25,
            x - 0.125,
            torch.where(x <= -0.25, torch.zeros_like(x), 2.0 * (x + 0.25).pow(2)),
        )
    raise KeyError(name)


def write_property_table(activations: list[str]) -> None:
    z = torch.linspace(-6.0, 6.0, 4001, dtype=torch.float64, requires_grad=True)
    rows = []
    for act in activations:
        y = activation_fn(act)(z)
        d1 = torch.autograd.grad(y.sum(), z, create_graph=True)[0]
        d2 = torch.autograd.grad(d1.sum(), z, create_graph=False)[0]
        d1_np = d1.detach().numpy()
        d2_np = d2.detach().numpy()
        max_d2 = max(float(np.max(np.abs(d2_np))), 1e-30)
        rows.append(
            {
                "activation": act,
                "family": family_name(act),
                "variant": variant(act),
                "d1_min": float(np.min(d1_np)),
                "d1_max": float(np.max(d1_np)),
                "d1_mean": float(np.mean(d1_np)),
                "d1_std": float(np.std(d1_np)),
                "frac_negative_slope": float(np.mean(d1_np < -1e-8)),
                "frac_near_zero_slope": float(np.mean(np.abs(d1_np) < 1e-3)),
                "curvature_l1": float(np.mean(np.abs(d2_np))),
                "curvature_peak": max_d2,
                "curvature_width_10pct": float(np.mean(np.abs(d2_np) > 0.1 * max_d2)),
            }
        )
    with (ANALYSIS / "activation_shape_properties.tsv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_family(rows: list[dict]) -> None:
    reps: dict[str, dict] = {}
    for row in rows:
        family = row["family"]
        if family not in reps or row["mean_score"] < reps[family]["mean_score"]:
            reps[family] = row
    points = sorted(reps.values(), key=lambda row: row["mean_h1"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.1, 1.2]})
    ax = axes[0]
    ax.scatter(
        [row["mean_neurons"] for row in points],
        [row["mean_h1"] for row in points],
        s=80,
        c=[row["mean_score"] for row in points],
        cmap="viridis_r",
        edgecolors="black",
    )
    for row in points:
        ax.annotate(
            row["family"],
            (row["mean_neurons"], row["mean_h1"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("mean neurons")
    ax.set_ylabel("mean H1")
    ax.set_title("Selected family representatives")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    top = sorted(points, key=lambda row: row["mean_score"])
    labels = [f"{row['family']} ({row['variant']})" for row in top]
    ypos = list(range(len(top)))
    ax2.barh(
        ypos,
        [row["mean_score"] for row in top],
        xerr=[row["std_score"] for row in top],
        color="lightblue",
        edgecolor="black",
        capsize=3,
    )
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("score = H1 x neurons")
    ax2.set_title("Consolidated selected families")
    ax2.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROOT / "pareto.png", dpi=130, bbox_inches="tight")
    plt.close()


def main() -> int:
    runs = load_runs()
    rows = write_family_tables(runs)
    write_gamma_tables(runs)
    shape_acts = [
        "softplus_b0_15",
        "softplus_b0_25",
        "softplus_b0_5",
        "gelu_b0_25",
        "mish_b0_15",
        "relu",
        "smoothy_relu_w0_25",
    ]
    write_property_table([act for act in shape_acts if act in runs or act in {"relu", "smoothy_relu_w0_25"}])
    plot_family(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
