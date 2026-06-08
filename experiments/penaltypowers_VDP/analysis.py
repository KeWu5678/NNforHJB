#!/usr/bin/env python3
"""Analyse the penaltypowers_VDP experiment.

Reads the per-run JSON records written by ``scripts/train.py`` under the Hydra
multirun directory and produces a Markdown results table and figure under
``experiments/penaltypowers_VDP/``. Selection rule: the best (lowest-score)
gamma per (power, loss, seed).
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

EXPERIMENT = "penaltypowers_VDP"
MULTIRUN_DIR = REPO_ROOT / "rawdata" / "logs" / "multirun" / EXPERIMENT
OUTPUT_DIR = REPO_ROOT / "experiments" / EXPERIMENT

_LOSS_LABEL = {(1.0, 0.0): "l2", (1.0, 1.0): "h1"}


def load_rows() -> list[dict[str, Any]]:
    """Read Hydra multirun records: axes from config, metrics from the run record."""
    records = sorted(MULTIRUN_DIR.glob("*/*.json"))
    if not records:
        raise FileNotFoundError(
            f"no run records under {MULTIRUN_DIR} - run `make penaltypowers_VDP` first"
        )
    rows = []
    for path in records:
        record = json.loads(path.read_text(encoding="utf-8"))
        model = record["config"]["model"]
        metrics = record["metrics"][0]["values"]
        loss = _LOSS_LABEL.get(tuple(model["loss_weights"]), str(model["loss_weights"]))
        neurons = int(metrics["best_neurons"])
        val_h1 = float(metrics["rel_h1_val"])
        rows.append({
            "power": float(model["power"]), "loss": loss,
            "gamma": float(model["gamma"]), "seed": int(record["config"]["env"]["seed"]),
            "neurons": neurons, "val_l2": float(metrics["rel_l2_val"]),
            "val_h1": val_h1, "score": val_h1 * max(neurons, 1),
        })
    return rows


def best_per_cell(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick the lowest-score row for each (power, loss, seed)."""
    def cell(row: dict[str, Any]) -> tuple:
        return (row["power"], row["loss"], row["seed"])

    best = []
    for _, group in itertools.groupby(sorted(rows, key=cell), key=cell):
        best.append(min(group, key=lambda row: row["score"]))
    return best


def main() -> int:
    best = best_per_cell(load_rows())

    table = format_table(
        best,
        ["power", "loss", "seed", "gamma", "neurons", "val_h1", "score"],
        headers={"val_h1": "Val H1"},
        formats={"power": "{:g}", "gamma": "{:g}", "val_h1": "{:.2e}", "score": "{:.2e}"},
        title="Best gamma per power/loss/seed",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "results.md").write_text(
        f"# {EXPERIMENT} Results\n\n{table}\n", encoding="utf-8")

    plot_score_tradeoff(
        best,
        x="neurons", y="val_h1", label="power", color="loss",
        title="penaltypowers_VDP: sparsity/accuracy tradeoff",
        xlabel="Neurons", ylabel="Validation H1",
        save_path=OUTPUT_DIR / "figures" / "tradeoff.png",
    )
    print(f"wrote {OUTPUT_DIR / 'results.md'} and figures/tradeoff.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
