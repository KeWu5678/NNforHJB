#!/usr/bin/env python3
"""Aggregate per-seed JSON files for the discontinuous-gradient search."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
TSV = ROOT / "results.tsv"

HEADER = [
    "commit",
    "activation",
    "power",
    "loss",
    "seeds",
    "mean_score",
    "std_score",
    "mean_eval_h1",
    "mean_eval_grad",
    "mean_neurons",
    "mean_near_grad",
    "mean_far_grad",
    "near_far_ratio",
    "best_gamma_mode",
    "status",
    "description",
]


def mean(values: list[float]) -> float:
    return statistics.mean(values)


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def ensure_header() -> None:
    if not TSV.exists() or TSV.stat().st_size == 0:
        TSV.write_text("\t".join(HEADER) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", required=True)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--description", default="")
    args = parser.parse_args()

    seeds = [int(seed) for seed in args.seeds.split(",") if seed]
    rows = []
    used = []
    for seed in seeds:
        path = RUNS / f"{args.activation}_seed{seed}.json"
        if not path.exists():
            continue
        try:
            line = path.read_text().strip().splitlines()[-1]
            rows.append(json.loads(line))
            used.append(seed)
        except Exception as exc:
            print(f"skip {path.name}: {exc}", file=sys.stderr)

    if not rows:
        print(f"no valid runs for {args.activation}", file=sys.stderr)
        return 1

    scores = [float(row["best_score"]) for row in rows]
    eval_h1 = [float(row["best_eval_h1"]) for row in rows]
    eval_grad = [float(row["best_eval_grad"]) for row in rows]
    neurons = [float(row["best_n"]) for row in rows]
    near_grad = [float(row["best_near_grad"]) for row in rows]
    far_grad = [float(row["best_far_grad"]) for row in rows]
    gammas = [row["best_gamma"] for row in rows]
    mode_gamma = Counter(gammas).most_common(1)[0][0]
    ratio = mean(near_grad) / max(mean(far_grad), 1e-30)

    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT.parents[1]
    ).decode().strip()

    status = "done" if len(rows) == len(seeds) else "partial"
    out = [
        commit,
        args.activation,
        f"{rows[0]['power']}",
        rows[0]["loss"],
        ",".join(str(seed) for seed in used),
        f"{mean(scores):.4f}",
        f"{stdev(scores):.4f}",
        f"{mean(eval_h1):.6f}",
        f"{mean(eval_grad):.6f}",
        f"{mean(neurons):.2f}",
        f"{mean(near_grad):.6f}",
        f"{mean(far_grad):.6f}",
        f"{ratio:.4f}",
        str(mode_gamma),
        status,
        args.description,
    ]

    ensure_header()
    row = "\t".join(out)
    with TSV.open("a") as file:
        file.write(row + "\n")
    print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
