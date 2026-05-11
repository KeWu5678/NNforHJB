#!/usr/bin/env python3
"""Quantify where gamma has the largest effect in activation-search runs."""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

EXPERIMENTS = {
    "vdp_hjb": {
        "runs": ROOT / "vdp_hjb_summary" / "runs",
        "out": ROOT / "vdp_hjb_summary" / "analysis" / "gamma_effect.tsv",
        "metrics": {
            "h1": lambda row: float(row["h1"]),
            "neurons": lambda row: float(row["n"]),
            "score": lambda row: float(row["score"]),
        },
    },
    "discontinuous_gradient": {
        "runs": ROOT / "discontinuous_gradient" / "runs",
        "out": ROOT / "discontinuous_gradient" / "gamma_effect.tsv",
        "metrics": {
            "near_grad": lambda row: float(row["near_grad"]),
            "far_grad": lambda row: float(row["far_grad"]),
            "near_far_ratio": lambda row: float(row["near_far_ratio"]),
            "neurons": lambda row: float(row["n"]),
            "near_score": lambda row: float(row["near_grad"]) * float(row["n"]),
            "eval_h1": lambda row: float(row["eval_h1"]),
        },
    },
}


def mean(values: list[float]) -> float:
    return statistics.mean(values)


def load_runs(runs_dir: Path) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(runs_dir.glob("*_seed*.json")):
        try:
            row = json.loads(path.read_text().strip().splitlines()[-1])
        except Exception:
            continue
        if "per_gamma" in row:
            runs[row["activation"]].append(row)
    return runs


def analyze_experiment(name: str, config: dict) -> list[dict]:
    runs = load_runs(config["runs"])
    out_rows = []
    for activation, activation_runs in sorted(runs.items()):
        seeds = sorted(int(row["seed"]) for row in activation_runs)
        by_gamma: dict[float, list[dict]] = defaultdict(list)
        for run in activation_runs:
            for row in run["per_gamma"]:
                by_gamma[float(row["gamma"])].append(row)

        for metric, getter in config["metrics"].items():
            gamma_values = []
            for gamma, rows in sorted(by_gamma.items()):
                values = [getter(row) for row in rows]
                gamma_values.append((gamma, mean(values)))
            if len(gamma_values) < 2:
                continue

            min_gamma, min_value = min(gamma_values, key=lambda item: item[1])
            max_gamma, max_value = max(gamma_values, key=lambda item: item[1])
            abs_range = max_value - min_value
            rel_range = abs_range / max(abs(min_value), 1e-30)

            adjacent = []
            for left, right in zip(gamma_values, gamma_values[1:]):
                adjacent.append((left[0], right[0], abs(right[1] - left[1])))
            adj_left, adj_right, adj_delta = max(adjacent, key=lambda item: item[2])

            out_rows.append(
                {
                    "experiment": name,
                    "activation": activation,
                    "seeds": ",".join(str(seed) for seed in seeds),
                    "num_seeds": len(seeds),
                    "metric": metric,
                    "min_gamma": min_gamma,
                    "min_value": min_value,
                    "max_gamma": max_gamma,
                    "max_value": max_value,
                    "abs_range": abs_range,
                    "rel_range_pct": 100.0 * rel_range,
                    "largest_adjacent_gamma_pair": f"{adj_left:g}->{adj_right:g}",
                    "largest_adjacent_delta": adj_delta,
                    "gamma_values": "; ".join(
                        f"{gamma:g}:{value:.6g}" for gamma, value in gamma_values
                    ),
                }
            )

    config["out"].parent.mkdir(parents=True, exist_ok=True)
    with config["out"].open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    return out_rows


def write_summary(all_rows: list[dict]) -> None:
    summary = ROOT / "gamma_effect_summary.md"
    lines = [
        "# Gamma Effect Summary",
        "",
        "Gamma effect is measured as the range of the mean metric across the fixed",
        "gamma sweep `[0, 0.01, 0.1, 1, 10]`, averaged over available seeds.",
        "",
        "## Main Interpretation",
        "",
        "The largest gamma effects are not the same as the most useful gamma effects.",
        "In both experiments, gamma often moves weak or high-neuron activations more",
        "than it moves the activation families that actually win the sparsity/accuracy",
        "tradeoff.",
        "",
        "For the smooth VDP HJB experiment, gamma matters most for `smoothy_relu_w0_25`",
        "in accuracy/score and for `relu` in sparsity. The best sparse softplus variant",
        "(`softplus_b0_25`) changes only modestly: H1 range `0.0179`, neuron range",
        "`4.8`, and score range `1.50`.",
        "",
        "For the discontinuous-gradient experiment, gamma has the largest absolute",
        "near-gradient effect on weak smooth/saturating activations such as `asinh`,",
        "`gelu_b0_2`, and `smoothy_relu_w0_05`. The leading squared-ReLU family is",
        "stable: `leaky_relu2_a0_02_sphere` has near-gradient range `0.0051`, neuron",
        "range `5.5`, and near-score range `0.64` across the full gamma sweep.",
        "",
    ]
    for experiment in ["vdp_hjb", "discontinuous_gradient"]:
        rows = [row for row in all_rows if row["experiment"] == experiment]
        lines.extend([f"## {experiment}", ""])
        for metric in sorted({row["metric"] for row in rows}):
            metric_rows = sorted(
                [row for row in rows if row["metric"] == metric],
                key=lambda row: float(row["abs_range"]),
                reverse=True,
            )[:8]
            lines.extend(
                [
                    f"### Largest absolute effect on `{metric}`",
                    "",
                    "| activation | min gamma/value | max gamma/value | range | largest adjacent |",
                    "|:-----------|:----------------|:----------------|------:|:-----------------|",
                ]
            )
            for row in metric_rows:
                lines.append(
                    "| {activation} | {min_gamma:g} / {min_value:.4g} | "
                    "{max_gamma:g} / {max_value:.4g} | {abs_range:.4g} | "
                    "{largest_adjacent_gamma_pair} |".format(**row)
                )
            lines.append("")
    summary.write_text("\n".join(lines) + "\n")


def main() -> int:
    all_rows = []
    for name, config in EXPERIMENTS.items():
        all_rows.extend(analyze_experiment(name, config))
    write_summary(all_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
