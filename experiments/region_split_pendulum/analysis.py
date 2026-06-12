#!/usr/bin/env python3
"""Analyse the region_split_pendulum study.

Reads per-run JSON records from ``rawdata/logs/multirun/region_split_pendulum/``.
The region metrics are computed by the training hook on the live as-fit model over
the **full dataset** (see ``scripts/train.py``); this just aggregates them into
``results.md`` (two tables + the error-vs-distance plot). `near` = lowest 10% of
samples by distance to the switching set; `far` = the rest. See ``README.md`` for
the error-metric rationale. Reproduce with ``make region_split_pendulum``.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metric import format_table  # noqa: E402
from src.paths import DATA_DIR  # noqa: E402

EXPERIMENT = "region_split_pendulum"
MULTIRUN_DIR = REPO_ROOT / "rawdata" / "logs" / "multirun" / EXPERIMENT
OUTPUT_DIR = REPO_ROOT / "experiments" / EXPERIMENT

_LOSS_LABEL = {(1.0, 0.0): "l2", (1.0, 1.0): "h1"}
_N_BINS = 8


def load_rows() -> tuple[list[dict[str, Any]], str | None]:
    records = sorted(MULTIRUN_DIR.glob("*/*.json"))
    if not records:
        raise FileNotFoundError(
            f"no run records under {MULTIRUN_DIR} — run `make region_split_pendulum` first"
        )
    rows, cache = [], None
    for path in records:
        record = json.loads(path.read_text(encoding="utf-8"))
        cfg = record["config"]
        model = cfg["model"]
        m = record["metrics"][0]["values"]
        if "near_l1_h1" not in m:
            continue
        cache = cache or cfg.get("eval", {}).get("distance_cache")
        loss = _LOSS_LABEL.get(tuple(model["loss_weights"]), str(model["loss_weights"]))
        rows.append({
            "kind": model["kind"],
            "insertion": model["insertion"],
            "activation": model["activation"],
            "loss": loss,
            "gamma": float(model["gamma"]),
            "neurons": int(m["best_neurons"]),
            "near_l1": float(m["near_l1_h1"]),
            "far_l1": float(m["far_l1_h1"]),
            "l1_near/far": float(m["near_l1_h1"]) / float(m["far_l1_h1"]) if m["far_l1_h1"] else float("inf"),
            "near_h1": float(m["near_h1"]),
            "far_h1": float(m["far_h1"]),
            "rel_near/far": float(m["near_h1"]) / float(m["far_h1"]) if m["far_h1"] else float("inf"),
            "bins": [m.get(f"distbin{i + 1}_ratio", float("nan")) for i in range(_N_BINS)],
        })
    return rows, cache


def best_per_cell(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lowest far-region L1 per (kind, insertion, activation, loss) — gamma selection."""
    def cell(row: dict[str, Any]) -> tuple:
        return (row["kind"], row["insertion"], row["activation"], row["loss"])

    best = []
    for _, group in itertools.groupby(sorted(rows, key=cell), key=cell):
        best.append(min(group, key=lambda r: r["far_l1"]))
    return best


def _bin_centers(cache: str | None) -> np.ndarray | None:
    if not cache:
        return None
    d = np.load(DATA_DIR / cache)["distance"]
    edges = np.quantile(d, np.linspace(0.0, 1.0, _N_BINS + 1))
    return 0.5 * (edges[:-1] + edges[1:])


def _plot_error_vs_distance(best: list[dict[str, Any]], centers: np.ndarray) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = sorted({r["kind"] for r in best})
    fig, axes = plt.subplots(1, len(kinds), figsize=(6 * len(kinds), 5), squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        for r in sorted(best, key=lambda r: r["activation"]):
            if r["kind"] == kind:
                ax.plot(centers, r["bins"], "-o", ms=3, label=r["activation"])
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_title(f"{kind} / profile")
        ax.set_xlabel("distance to switching set")
        ax.set_ylabel("per-sample abs error / model mean")
        ax.legend(fontsize=8)
    fig.suptitle("Error vs distance to switching set (near → far); >1 = worse than model average")
    fig.tight_layout()
    out = OUTPUT_DIR / "figures" / "error_vs_distance.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    return out


def main() -> int:
    rows, cache = load_rows()
    best = best_per_cell(rows)

    l1 = sorted(best, key=lambda r: (r["kind"], r["insertion"], r["loss"], r["l1_near/far"]))
    l1_table = format_table(
        l1,
        ["kind", "insertion", "activation", "loss", "gamma", "neurons",
         "near_l1", "far_l1", "l1_near/far"],
        headers={"near_l1": "near L1", "far_l1": "far L1", "l1_near/far": "near/far"},
        formats={"gamma": "{:g}", "near_l1": "{:.2e}", "far_l1": "{:.2e}",
                 "l1_near/far": "{:.2f}"},
        title="Mean per-sample L1 over the full dataset — count-fair, robust to V→0",
    )
    rel = sorted(best, key=lambda r: (r["kind"], r["insertion"], r["loss"], r["rel_near/far"]))
    rel_table = format_table(
        rel,
        ["kind", "insertion", "activation", "loss", "gamma", "neurons",
         "near_h1", "far_h1", "rel_near/far"],
        headers={"near_h1": "near H1", "far_h1": "far H1", "rel_near/far": "near/far"},
        formats={"gamma": "{:g}", "near_h1": "{:.2e}", "far_h1": "{:.2e}",
                 "rel_near/far": "{:.2f}"},
        title="Relative H1 (kept for continuity — confounded by the V→0 interior)",
    )

    centers = _bin_centers(cache)
    fig_line = ""
    if centers is not None:
        fig = _plot_error_vs_distance(best, centers)
        fig_line = f"\n![error vs distance](figures/{fig.name})\n"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "results.md"
    out.write_text(
        f"# {EXPERIMENT} Results\n\n"
        "Region split scored over the **full dataset** on the live as-fit model. "
        "`near` = lowest 10% of samples by distance to the switching set; `far` = the "
        "rest. See `README.md` for the error-metric rationale.\n\n"
        "## Mean per-sample L1 (primary)\n\n"
        "`near/far` > 1 ⇒ worse at the switching set. Region mean per-sample L1 "
        "(absolute) error / global mean ‖true‖ — count-fair and robust to the V→0 "
        "interior.\n\n"
        f"{l1_table}\n\n"
        "## Error vs distance to switching set (diagnostic)\n"
        f"{fig_line}\n"
        "## Relative H1 (kept for continuity — confounded)\n\n"
        f"{rel_table}\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
