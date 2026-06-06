#!/usr/bin/env python3
"""Analyse the activationsearch experiment.

Reads the per-run JSON records written by ``scripts/train.py`` under the Hydra
multirun directory (``multirun/activationsearch/<n>/``) and produces a Markdown
results table and the sparsity/accuracy tradeoff figure under
``experiments/activationsearch/``. Selection rule: the best (lowest-score) gamma
per (activation, loss, seed). Reproduce the runs with ``make activationsearch``.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metric import format_table
from src.plots import plot_score_tradeoff

EXPERIMENT = "activationsearch"
MULTIRUN_DIR = REPO_ROOT / "rawdata" / "logs" / "multirun" / EXPERIMENT
OUTPUT_DIR = REPO_ROOT / "experiments" / EXPERIMENT

# Readable label for the swept loss_weights pair.
_LOSS_LABEL = {(1.0, 0.0): "l2", (1.0, 1.0): "h1"}


def load_rows() -> list[dict[str, Any]]:
    """Read each multirun run-record: axes from its config, metrics from its log."""
    records = sorted(MULTIRUN_DIR.glob("*/*.json"))
    if not records:
        raise FileNotFoundError(
            f"no run records under {MULTIRUN_DIR} — run `make activationsearch` first"
        )
    rows = []
    for path in records:
        record = json.loads(path.read_text(encoding="utf-8"))
        model = record["config"]["model"]
        metrics = record["metrics"][0]["values"]
        neurons = int(metrics["best_neurons"])
        val_h1 = float(metrics["rel_h1_val"])
        loss = _LOSS_LABEL.get(tuple(model["loss_weights"]), str(model["loss_weights"]))
        rows.append({
            "activation": model["activation"], "loss": loss,
            "gamma": float(model["gamma"]), "seed": int(record["config"]["env"]["seed"]),
            "neurons": neurons, "val_l2": float(metrics["rel_l2_val"]),
            "val_h1": val_h1, "score": val_h1 * max(neurons, 1),
        })
    return rows


def best_per_cell(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick the lowest-score row for each (activation, loss, seed)."""
    def cell(row: dict[str, Any]) -> tuple:
        return (row["activation"], row["loss"], row["seed"])

    best = []
    for _, group in itertools.groupby(sorted(rows, key=cell), key=cell):
        best.append(min(group, key=lambda row: row["score"]))
    return best


def main() -> int:
    best = best_per_cell(load_rows())

    table = format_table(
        best,
        ["activation", "loss", "seed", "gamma", "neurons", "val_h1", "score"],
        headers={"val_h1": "Val H1"},
        formats={"gamma": "{:g}", "val_h1": "{:.2e}", "score": "{:.2e}"},
        title="Best gamma per activation/loss/seed",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "results.md").write_text(
        f"# {EXPERIMENT} Results\n\n{table}\n", encoding="utf-8")

    plot_score_tradeoff(
        best,
        x="neurons", y="val_h1", label="activation", color="loss",
        title="activationsearch: sparsity/accuracy tradeoff",
        xlabel="Neurons", ylabel="Validation H1",
        save_path=OUTPUT_DIR / "figures" / "tradeoff.png",
    )
    print(f"wrote {OUTPUT_DIR / 'results.md'} and figures/tradeoff.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
