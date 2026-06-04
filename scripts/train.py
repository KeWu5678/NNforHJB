#!/usr/bin/env python3
"""Generic Hydra entry point: train a PDAP model on any value/gradient dataset.

A run = pick a registered model + a data source, override the rest::

    python scripts/train.py model=semiconcave model.gamma=10
    python scripts/train.py -m model.gamma=0,1e-2,1e-1,1,10 env.seed=42,43,44

This entry is domain-agnostic — it loads a ``.npy`` with keys ``x, v, dv`` and
fits the PDAP model described by the config. The default ``data=vdp`` reproduces a
single VDP signed-profile run.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

# Compatibility shim: Python 3.14 added ArgumentParser._check_help, which rejects
# Hydra 1.3's --shell-completion help (it contains a literal '%'). The guard is a
# pure help-lint added in 3.14; neutralizing it restores pre-3.14 behavior so
# @hydra.main can build its CLI parser. Remove once Hydra ships a 3.14-safe release.
if hasattr(argparse.ArgumentParser, "_check_help"):
    argparse.ArgumentParser._check_help = lambda self, action: None

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import src.config.store  # noqa: F401  — registers `config_schema` with Hydra's ConfigStore
from src.PDAP import PDAP
from src.experiment_logging import ExperimentRun
from src.logging_config import configure_logging
from src.paths import DATA_DIR

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(path: Path) -> dict:
    """Load a value/gradient dataset (any ``.npy`` with keys ``x, v, dv``)."""
    raw = np.load(path, allow_pickle=True)
    return {
        "x": np.asarray(raw["x"], dtype=np.float64),
        "v": np.asarray(raw["v"], dtype=np.float64),
        "dv": np.asarray(raw["dv"], dtype=np.float64),
    }


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    configure_logging(verbose=cfg.env.verbose, level=cfg.env.log_level, log_file=cfg.env.log_file)
    set_seed(cfg.env.seed)

    data_path = Path(cfg.data.path)
    if not data_path.is_absolute():
        data_path = DATA_DIR / data_path
    data = load_dataset(data_path)
    logger.info("loaded %s  (N=%d, d=%d)", data_path.name, data["x"].shape[0], data["x"].shape[1])

    pdap = PDAP.from_config(cfg, data)
    result = pdap.fit_from_config(cfg.training, verbose=cfg.env.verbose)

    # Report the relative errors + sparsity metrics at the selected best iteration.
    bi = int(result["best_iteration"])
    metrics = {
        "rel_l2_train": float(result["err_l2_train"][bi]),
        "rel_l2_val": float(result["err_l2_val"][bi]),
        "rel_grad_train": float(result["err_grad_train"][bi]),
        "rel_grad_val": float(result["err_grad_val"][bi]),
        "rel_h1_train": float(result["err_h1_train"][bi]),
        "rel_h1_val": float(result["err_h1_val"][bi]),
        "best_iteration": bi,
        "best_neurons": int(result["best_neurons"]),
        "final_neurons": int(result["final_neurons"]),
    }
    logger.info(
        "best iter %d | neurons %d | rel-L2 %.3e | rel-semiH1 %.3e | rel-H1 %.3e (val)",
        bi, metrics["best_neurons"], metrics["rel_l2_val"], metrics["rel_grad_val"], metrics["rel_h1_val"],
    )

    # Persist a run record into Hydra's per-run output dir (Hydra also writes
    # .hydra/config.yaml). Experiment-tracking config is deferred.
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run = ExperimentRun(
        output_dir=run_dir,
        name=cfg.name,
        run_id=f"{cfg.model.kind}_{cfg.model.insertion}_seed{cfg.env.seed}",
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    run.log_metrics(metrics)
    record = run.finish(status="completed")
    logger.info("run record: %s", record)


if __name__ == "__main__":
    main()
