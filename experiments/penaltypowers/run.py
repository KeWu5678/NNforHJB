#!/usr/bin/env python3
"""Run the penaltypowers experiment: sweep (seed, power, loss, gamma).

Fits one PDAP model per grid point, pickles each fit result under
``rawdata/data/penaltypowers/``, and writes a ``runs.json`` index of summary
rows. Turn that index into tables/figures with ``analysis.py``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.PDAP import PDAP
from src.config.schema import ExperimentConfig
from src.paths import DATA_DIR

EXPERIMENT = "penaltypowers"
DEFAULT_CONFIG = REPO_ROOT / "conf" / "experiment" / f"{EXPERIMENT}.yaml"
ARTIFACT_DIR = DATA_DIR / EXPERIMENT

# The `losses` sweep axis is a categorical label; map each to (value, gradient) weights.
_LOSS_WEIGHTS = {"l2": (1.0, 0.0), "h1": (1.0, 1.0)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, default=None, help="run only the first N grid points")
    args = parser.parse_args()

    raw = OmegaConf.load(args.config)
    base = OmegaConf.structured(ExperimentConfig)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    grid = itertools.product(raw.seeds, raw.powers, raw.losses, raw.gammas)
    for index, (seed, power, loss, gamma) in enumerate(grid):
        if args.limit is not None and index >= args.limit:
            break
        set_seed(seed)
        cfg = OmegaConf.merge(base, {
            "name": EXPERIMENT,
            "model": {**raw.model, "power": float(power),
                      "loss_weights": list(_LOSS_WEIGHTS[loss]), "gamma": float(gamma)},
            "training": raw.training,
            "data": dict(raw.data),
            "env": {"seed": int(seed), "verbose": False},
        })
        result = PDAP.from_config(cfg).fit_from_config(cfg.training, verbose=False)

        rid = f"power{power:g}_{loss}_gamma{gamma:g}_seed{seed}".replace(".", "p").replace("-", "m")
        artifact = ARTIFACT_DIR / f"{rid}.pkl"
        with artifact.open("wb") as file:
            pickle.dump(result, file)

        # Index records only the sweep axes + artifact path (the axes are not in
        # the pickled result); metrics are read back from the artifact in analysis.py.
        rows.append({
            "power": float(power), "loss": loss, "gamma": float(gamma), "seed": int(seed),
            "artifact": str(artifact.relative_to(REPO_ROOT)),
        })
        print(f"{rid}: done")

    index_path = ARTIFACT_DIR / "runs.json"
    index_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} runs -> {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
