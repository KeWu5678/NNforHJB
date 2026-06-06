#!/usr/bin/env python3
"""Run one (activation, seed) experiment for the activation-search study.

Sweeps the fixed gamma list with signed-profile PDAP (power=1, loss=h1), then prints a
single JSON line on stdout summarizing per-gamma results and the best score
score := err_h1_val[best_iteration] * best_neurons.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.PDAP import PDAP
from src.config.activations import ACTIVATIONS
from src.config.schema import EnvConfig, ExperimentConfig, ModelConfig, TrainingConfig
from src.experiment_logging import RunRecordWriter
from src.logging_config import configure_logging
from src.paths import DATA_DIR

logger = logging.getLogger(__name__)


GAMMAS = [0, 1e-2, 1e-1, 1, 10]
ALPHA = 1e-5
POWER = 1.0
LOSS_WEIGHTS = (1.0, 1.0)
NUM_ITERATIONS = 10
NUM_INSERTION = 50
DATA_PATH = DATA_DIR / "VDP_beta_0.1_grid_30x30.npy"

RUN_RECORD = RunRecordWriter(
    REPO_ROOT,
    name="activation_search",
    id_fields=("activation", "seed"),
    config_fields=(
        "activation",
        "seed",
        "num_iterations",
        "num_insertion",
        "power",
        "loss",
        "use_sphere",
    ),
    metric_field="per_gamma",
    metric_step_field="gamma",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data() -> dict:
    raw = np.load(DATA_PATH)
    return {
        "x":  np.asarray(raw["x"],  dtype=np.float64),
        "v":  np.asarray(raw["v"],  dtype=np.float64),
        "dv": np.asarray(raw["dv"], dtype=np.float64),
    }


def main() -> int:
    configure_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--activation", required=True, choices=sorted(ACTIVATIONS))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--num-iterations", type=int, default=NUM_ITERATIONS)
    p.add_argument("--num-insertion",  type=int, default=NUM_INSERTION)
    p.add_argument("--use-sphere", action="store_true",
                   help="sample candidate directions on S^d (only for positively-homogeneous activations)")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    data = load_data()

    per_gamma = []
    t0 = time.time()
    for gamma in GAMMAS:
        set_seed(args.seed)
        cfg = ExperimentConfig(
            model=ModelConfig(
                kind="signed", insertion="profile", activation=args.activation,
                power=POWER, loss_weights=tuple(LOSS_WEIGHTS), use_sphere=args.use_sphere,
                alpha=ALPHA, gamma=gamma,
            ),
            training=TrainingConfig(num_iterations=args.num_iterations, num_insertion=args.num_insertion),
            env=EnvConfig(verbose=False),
        )
        pdpa = PDAP(cfg, data)
        result = pdpa.fit(
            num_iterations=args.num_iterations,
            num_insertion=args.num_insertion,
            verbose=False,
        )
        bi = result["best_iteration"]
        h1 = float(result["err_h1_val"][bi])
        n  = int(result["best_neurons"])
        per_gamma.append({
            "gamma": gamma, "h1": h1, "n": n, "score": h1 * n,
            "best_iteration": bi,
        })

    best = min(per_gamma, key=lambda r: r["score"])
    out = {
        "activation": args.activation,
        "seed":       args.seed,
        "num_iterations": args.num_iterations,
        "num_insertion": args.num_insertion,
        "power":      POWER,
        "loss":       LOSS_WEIGHTS,
        "use_sphere": args.use_sphere,
        "elapsed_s":  round(time.time() - t0, 2),
        "per_gamma":  per_gamma,
        "best_gamma": best["gamma"],
        "best_score": best["score"],
        "best_h1":    best["h1"],
        "best_n":     best["n"],
    }
    if args.output_dir is not None:
        path = RUN_RECORD.write(out, output_dir=args.output_dir)
        logger.info("saved run record: path=%s", path)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
