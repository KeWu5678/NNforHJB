#!/usr/bin/env python3
"""Rank activation runs by behavior near the known gradient discontinuity.

This re-aggregates existing per-seed JSON files. For each activation and seed,
it selects the gamma with the smallest near-discontinuity gradient error, then
reports activation-level means. This is the primary view for Experiment 3.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
OUT = ROOT / "results_near.tsv"

HEADER = [
    "activation",
    "seeds",
    "mean_near_grad",
    "std_near_grad",
    "mean_far_grad",
    "near_far_ratio",
    "mean_eval_h1",
    "mean_eval_grad",
    "mean_neurons",
    "mean_near_score",
    "std_near_score",
    "best_gamma_mode",
    "status",
]


def mean(values: list[float]) -> float:
    return statistics.mean(values)


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def load_runs() -> dict[str, list[tuple[int, dict]]]:
    runs: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for path in sorted(RUNS.glob("*_seed*.json")):
        stem = path.stem
        if "_seed" not in stem:
            continue
        activation, seed_text = stem.rsplit("_seed", 1)
        try:
            seed = int(seed_text)
            line = path.read_text().strip().splitlines()[-1]
            row = json.loads(line)
        except Exception as exc:
            print(f"skip {path.name}: {exc}", file=sys.stderr)
            continue
        if "per_gamma" not in row:
            print(f"skip {path.name}: missing per_gamma", file=sys.stderr)
            continue
        runs[activation].append((seed, row))
    return runs


def main() -> int:
    runs = load_runs()
    rows = []
    for activation, seed_rows in sorted(runs.items()):
        selected = []
        seeds = []
        for seed, row in sorted(seed_rows):
            per_gamma = row["per_gamma"]
            best = min(per_gamma, key=lambda item: float(item["near_grad"]))
            selected.append(best)
            seeds.append(seed)
        if not selected:
            continue

        near = [float(row["near_grad"]) for row in selected]
        far = [float(row["far_grad"]) for row in selected]
        h1 = [float(row["eval_h1"]) for row in selected]
        grad = [float(row["eval_grad"]) for row in selected]
        neurons = [float(row["n"]) for row in selected]
        near_score = [float(row["near_grad"]) * float(row["n"]) for row in selected]
        gammas = [row["gamma"] for row in selected]
        baseline = [42, 43, 44, 45, 46]
        status = "done" if all(seed in seeds for seed in baseline) else "partial"
        rows.append(
            [
                activation,
                ",".join(str(seed) for seed in seeds),
                f"{mean(near):.6f}",
                f"{stdev(near):.6f}",
                f"{mean(far):.6f}",
                f"{mean(near) / max(mean(far), 1e-30):.4f}",
                f"{mean(h1):.6f}",
                f"{mean(grad):.6f}",
                f"{mean(neurons):.2f}",
                f"{mean(near_score):.4f}",
                f"{stdev(near_score):.4f}",
                str(Counter(gammas).most_common(1)[0][0]),
                status,
            ]
        )

    rows.sort(key=lambda row: float(row[2]))
    OUT.write_text(
        "\t".join(HEADER)
        + "\n"
        + "\n".join("\t".join(row) for row in rows)
        + "\n"
    )
    print(f"wrote {OUT} ({len(rows)} activations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
