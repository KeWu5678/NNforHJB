"""Centralized output locations for the SparseNNforHJB project.

Single source of truth for where generated artifacts go, so scripts and
modules don't each re-derive ``REPO_ROOT`` and hard-code ``rawdata/...``.
Import the directory you need instead of building the path inline::

    from src.paths import PLOTS_DIR, LOGS_DIR, DATA_DIR

    fig.savefig(PLOTS_DIR / "value_surface.png")

  PLOTS_DIR : all generated figures (.png/.pdf)
  LOGS_DIR  : all run logs (diagnostic logs, experiment run records)
  DATA_DIR  : all generated data outputs (open-loop datasets, model pickles, ...)
"""

from __future__ import annotations

from pathlib import Path

#: Repository root (parent of ``src/``).
REPO_ROOT = Path(__file__).resolve().parents[1]

PLOTS_DIR = REPO_ROOT / "rawdata" / "plots"
LOGS_DIR = REPO_ROOT / "rawdata" / "logs"
DATA_DIR = REPO_ROOT / "rawdata" / "data"

__all__ = ["REPO_ROOT", "PLOTS_DIR", "LOGS_DIR", "DATA_DIR"]
