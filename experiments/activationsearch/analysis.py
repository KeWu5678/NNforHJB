#!/usr/bin/env python3
"""Analyse the activationsearch experiment.

Reads per-run JSON records from ``rawdata/logs/multirun/activationsearch/``
and writes a Markdown summary table to ``experiments/activationsearch/results.md``.
Selection rule: best (lowest score) gamma per (data, kind, insertion, activation, loss, seed).
Reproduce with ``make activationsearch``.
"""

from __future__ import annotations

import itertools
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metric import format_table

EXPERIMENT = "activationsearch"
MAKEFILE = REPO_ROOT / "Makefile"
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


def _makefile_override_values(name: str) -> list[str]:
    """Return comma-separated values for a simple Makefile CLI override."""
    text = MAKEFILE.read_text(encoding="utf-8")
    match = re.search(rf"(?m)\b{re.escape(name)}=([^\s\\]+)", text)
    if not match:
        return []
    return [value for value in match.group(1).split(",") if value]


def load_rows() -> list[dict[str, Any]]:
    records = sorted(MULTIRUN_DIR.glob("*/*.json"))
    if not records:
        raise FileNotFoundError(
            f"no run records under {MULTIRUN_DIR} — run `make activationsearch` first"
        )
    rows = []
    for path in records:
        record = json.loads(path.read_text(encoding="utf-8"))
        cfg = record["config"]
        model = cfg["model"]
        metrics = record["metrics"][0]["values"]
        neurons = int(metrics["best_neurons"])
        val_h1 = float(metrics["rel_h1_val"])
        loss = _LOSS_LABEL.get(tuple(model["loss_weights"]), str(model["loss_weights"]))
        rows.append({
            "data": _dataset_label(cfg["data"]["path"]),
            "kind": model["kind"],
            "insertion": model["insertion"],
            "activation": model["activation"],
            "loss": loss,
            "gamma": float(model["gamma"]),
            "seed": int(cfg["env"]["seed"]),
            "neurons": neurons,
            "val_l2": float(metrics["rel_l2_val"]),
            "val_h1": val_h1,
            "score": val_h1 * max(neurons, 1),
        })
    return rows


def best_per_cell(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick the lowest-score row for each (data, kind, insertion, activation, loss, seed)."""
    def cell(row: dict[str, Any]) -> tuple:
        return (row["data"], row["kind"], row["insertion"],
                row["activation"], row["loss"], row["seed"])

    best = []
    for _, group in itertools.groupby(sorted(rows, key=cell), key=cell):
        best.append(min(group, key=lambda r: r["score"]))
    return best


def coverage_summary(rows: list[dict[str, Any]]) -> str:
    expected_data = _makefile_override_values("data")
    expected_activations = _makefile_override_values("model.activation")
    observed_data = sorted({row["data"] for row in rows})
    observed_activations = sorted({row["activation"] for row in rows})

    def line(label: str, values: list[str]) -> str:
        return f"- {label}: {', '.join(values) if values else 'none'}"

    missing_data = [value for value in expected_data if value not in observed_data]
    missing_activations = [
        value for value in expected_activations if value not in observed_activations
    ]

    return "\n".join(
        [
            "## Sweep coverage",
            "",
            line("Configured data", expected_data),
            line("Observed data", observed_data),
            line("Missing configured data", missing_data),
            line("Configured activations", expected_activations),
            line("Observed activations", observed_activations),
            line("Missing configured activations", missing_activations),
        ]
    )


def main() -> int:
    rows = load_rows()
    best = best_per_cell(rows)
    best.sort(key=lambda r: (r["data"], r["kind"], r["insertion"],
                              r["activation"], r["loss"], r["score"]))

    table = format_table(
        best,
        ["data", "kind", "insertion", "activation", "loss", "seed",
         "gamma", "neurons", "val_l2", "val_h1", "score"],
        headers={"val_l2": "Val L2", "val_h1": "Val H1"},
        formats={"gamma": "{:g}", "val_l2": "{:.2e}", "val_h1": "{:.2e}", "score": "{:.2e}"},
        title="Best gamma per data/kind/insertion/activation/loss/seed",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "results.md"
    out.write_text(
        f"# {EXPERIMENT} Results\n\n{coverage_summary(rows)}\n\n{table}\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
