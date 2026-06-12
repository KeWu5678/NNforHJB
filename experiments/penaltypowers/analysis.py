#!/usr/bin/env python3
"""Analyse the penaltypowers experiment.

Reads the per-run JSON records written by ``scripts/train.py`` under the Hydra
multirun directory and produces a Markdown results table and figure under
``experiments/penaltypowers/``. Selection rule: the best (lowest-score)
gamma per (data, activation, power, loss, seed).
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

from src.config.activations import ACTIVATIONS
from src.metric import format_table
from src.plots import plot_score_tradeoff

EXPERIMENT = "penaltypowers"
MULTIRUN_DIR = REPO_ROOT / "rawdata" / "logs" / "multirun" / EXPERIMENT
OUTPUT_DIR = REPO_ROOT / "experiments" / EXPERIMENT

_LOSS_LABEL = {(1.0, 0.0): "l2", (1.0, 1.0): "h1"}


def _dataset_label(path: str) -> str:
    p = path.lower()
    if "pendulum" in p:
        return "pendulum"
    if "vdp" in p or "van_der_pol" in p:
        return "vdp"
    return Path(path).stem


def homogeneous_activation_names() -> list[str]:
    return [name for name, (_, use_sphere) in ACTIVATIONS.items() if use_sphere]


def load_rows() -> list[dict[str, Any]]:
    """Read Hydra multirun records: axes from config, metrics from the run record."""
    records = sorted(MULTIRUN_DIR.glob("*/*.json"))
    if not records:
        raise FileNotFoundError(
            f"no run records under {MULTIRUN_DIR} - run `make penaltypowers` first"
        )
    rows = []
    for path in records:
        record = json.loads(path.read_text(encoding="utf-8"))
        cfg = record["config"]
        model = cfg["model"]
        metrics = record["metrics"][0]["values"]
        loss = _LOSS_LABEL.get(tuple(model["loss_weights"]), str(model["loss_weights"]))
        neurons = int(metrics["best_neurons"])
        val_h1 = float(metrics["rel_h1_val"])
        rows.append({
            "data": _dataset_label(cfg["data"]["path"]),
            "activation": model["activation"],
            "power": float(model["power"]), "loss": loss,
            "gamma": float(model["gamma"]), "seed": int(record["config"]["env"]["seed"]),
            "neurons": neurons, "val_l2": float(metrics["rel_l2_val"]),
            "val_h1": val_h1, "score": val_h1 * max(neurons, 1),
        })
    return rows


def best_per_cell(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick the lowest-score row for each (data, activation, power, loss, seed)."""
    def cell(row: dict[str, Any]) -> tuple:
        return (row["data"], row["activation"], row["power"], row["loss"], row["seed"])

    best = []
    for _, group in itertools.groupby(sorted(rows, key=cell), key=cell):
        best.append(min(group, key=lambda row: row["score"]))
    return best


def coverage_summary(rows: list[dict[str, Any]]) -> str:
    expected_data = ["vdp", "pendulum"]
    expected_activations = homogeneous_activation_names()
    observed_data = sorted({row["data"] for row in rows})
    observed_activations = sorted({row["activation"] for row in rows})
    missing_data = [value for value in expected_data if value not in observed_data]
    missing_activations = [
        value for value in expected_activations if value not in observed_activations
    ]

    def line(label: str, values: list[str]) -> str:
        return f"- {label}: {', '.join(values) if values else 'none'}"

    return "\n".join(
        [
            "## Sweep coverage",
            "",
            line("Configured data", expected_data),
            line("Observed data", observed_data),
            line("Missing configured data", missing_data),
            line("Configured homogeneous activations", expected_activations),
            line("Observed activations", observed_activations),
            line("Missing configured activations", missing_activations),
        ]
    )


def main() -> int:
    rows = load_rows()
    best = best_per_cell(rows)
    best.sort(key=lambda row: (
        row["data"], row["activation"], row["power"], row["loss"], row["score"],
    ))

    table = format_table(
        best,
        ["data", "activation", "power", "loss", "seed",
         "gamma", "neurons", "val_h1", "score"],
        headers={"val_h1": "Val H1"},
        formats={"power": "{:g}", "gamma": "{:g}", "val_h1": "{:.2e}", "score": "{:.2e}"},
        title="Best gamma per data/activation/power/loss/seed",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "results.md").write_text(
        f"# {EXPERIMENT} Results\n\n{coverage_summary(rows)}\n\n{table}\n",
        encoding="utf-8",
    )

    plot_score_tradeoff(
        [dict(row, label=f"{row['activation']} p={row['power']:g}") for row in best],
        x="neurons", y="val_h1", label="label", color="data",
        title="penaltypowers: sparsity/accuracy tradeoff",
        xlabel="Neurons", ylabel="Validation H1",
        save_path=OUTPUT_DIR / "figures" / "tradeoff.png",
    )
    print(f"wrote {OUTPUT_DIR / 'results.md'} and figures/tradeoff.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
