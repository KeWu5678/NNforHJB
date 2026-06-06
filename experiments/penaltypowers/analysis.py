#!/usr/bin/env python3
"""Analyse the penaltypowers experiment.

Reads the ``runs.json`` index written by ``run.py`` and produces a Markdown
results table and the sparsity/accuracy tradeoff figure under
``experiments/penaltypowers/``. Selection rule: the best (lowest-score)
gamma per (power, loss, seed).
"""

from __future__ import annotations

import itertools
import json
import pickle
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metric import format_table
from src.paths import DATA_DIR
from src.plots import plot_score_tradeoff

EXPERIMENT = "penaltypowers"
RUNS_INDEX = DATA_DIR / EXPERIMENT / "runs.json"
OUTPUT_DIR = REPO_ROOT / "experiments" / EXPERIMENT


def load_rows() -> list[dict[str, Any]]:
    """Read the run index and pull metrics from each pickled fit result.

    The index holds only the sweep axes + artifact path; the metrics
    (neurons, errors, score) live in the pickled result and are read here.
    """
    index = json.loads(RUNS_INDEX.read_text(encoding="utf-8"))
    rows = []
    for entry in index:
        with (REPO_ROOT / entry["artifact"]).open("rb") as file:
            result = pickle.load(file)
        bi = int(result["best_iteration"])
        neurons = int(result["best_neurons"])
        val_h1 = float(result["err_h1_val"][bi])
        rows.append({
            "power": entry["power"], "loss": entry["loss"],
            "gamma": entry["gamma"], "seed": entry["seed"],
            "neurons": neurons, "val_l2": float(result["err_l2_val"][bi]),
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
        title="penaltypowers: sparsity/accuracy tradeoff",
        xlabel="Neurons", ylabel="Validation H1",
        save_path=OUTPUT_DIR / "figures" / "tradeoff.png",
    )
    print(f"wrote {OUTPUT_DIR / 'results.md'} and figures/tradeoff.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
