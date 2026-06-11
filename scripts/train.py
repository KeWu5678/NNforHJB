#!/usr/bin/env python3
"""Generic Hydra entry point: train a PDAP model on any value/gradient dataset.

A run = pick a registered model + a data source, override the rest::

    python scripts/train.py model=semiconcave model.gamma=10
    python scripts/train.py -m model.gamma=0,1e-2,1e-1,1,10 env.seed=42,43,44

This entry is domain-agnostic — it loads a ``.npy``/``.npz`` with keys ``x, v, dv`` and
fits the PDAP model described by the config. The default ``data=vdp`` reproduces a
single VDP signed-profile run.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import random
import re
import secrets
import sys
from datetime import datetime
from pathlib import Path

# Compatibility shim: Python 3.14 added ArgumentParser._check_help, which rejects
# Hydra 1.3's --shell-completion help (it contains a literal '%'). The guard is a
# pure help-lint added in 3.14; neutralizing it restores pre-3.14 behavior so
# @hydra.main can build its CLI parser. Remove once Hydra ships a 3.14-safe release.
if hasattr(argparse.ArgumentParser, "_check_help"):
    argparse.ArgumentParser._check_help = lambda _self, _action: None

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import src.config.store  # noqa: F401  — registers `config_schema` with Hydra's ConfigStore
from src.data import load_value_samples, normalize_value_samples, split_value_samples
from src.models import build_model
from src.PDAP import PDAP
from src.experiment_logging import ExperimentRun
from src.logging_config import configure_logging

logger = logging.getLogger(__name__)


def _slug(value: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "", str(value).lower())
    return slug or "run"


def _to_container(value):
    if isinstance(value, DictConfig | ListConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _select(hydra_cfg, path: str):
    return _to_container(OmegaConf.select(hydra_cfg, path))


def hydra_metadata(hydra_cfg) -> dict:
    return {
        "output_dir": str(_select(hydra_cfg, "runtime.output_dir")),
        "job": {
            "name": _select(hydra_cfg, "job.name"),
            "id": _select(hydra_cfg, "job.id"),
            "num": _select(hydra_cfg, "job.num"),
        },
        "runtime": {
            "choices": _select(hydra_cfg, "runtime.choices") or {},
        },
        "overrides": {
            "task": _select(hydra_cfg, "overrides.task") or [],
        },
    }


def run_id_from_config(
    cfg: DictConfig,
    *,
    hydra_cfg=None,
    today: str | None = None,
    suffix: str | None = None,
) -> str:
    choices = _select(hydra_cfg, "runtime.choices") if hydra_cfg is not None else {}
    data_choice = (choices or {}).get("data") or Path(str(cfg.data.path)).stem
    run_date = today or datetime.now().strftime("%Y%m%d")
    run_suffix = suffix or secrets.token_hex(2)
    return "_".join([_slug(cfg.name), _slug(data_choice), run_date, run_suffix])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Every run logs to its own file in the Hydra output dir, so parallel sweep
    # workers (joblib) don't interleave their progress tables on the shared
    # console. `env.verbose` still controls console streaming; `env.log_file`
    # overrides the per-run default when set.
    hydra_cfg = HydraConfig.get()
    run_dir = Path(hydra_cfg.runtime.output_dir)
    log_file = cfg.env.log_file or (run_dir / "run.log")
    configure_logging(verbose=cfg.env.verbose, level=cfg.env.log_level, log_file=log_file)
    set_seed(cfg.env.seed)

    # Create the run record before model construction/training so elapsed_s covers
    # the actual run, not only JSON serialization.
    run = ExperimentRun(
        output_dir=run_dir,
        name=cfg.name,
        run_id=run_id_from_config(cfg, hydra_cfg=hydra_cfg),
        config=OmegaConf.to_container(cfg, resolve=True),
        hydra=hydra_metadata(hydra_cfg),
    )

    # Data preprocessing lives in the script: load, normalize, split.  The model
    # is built by build_model; the trainer holds only config and returns a History.
    data = load_value_samples(cfg.data.path)
    if cfg.data.normalize:
        data, _ = normalize_value_samples(data)
    input_dim = data["x"].shape[1]
    logger.info("loaded %s  (d=%d)", cfg.data.path, input_dim)
    train_data, valid_data = split_value_samples(data, cfg.data.train_fraction)
    model = build_model(cfg, input_dim)

    history = PDAP(cfg).fit(
        model, train_data, valid_data,
        num_iterations=cfg.training.num_iterations,
        num_insertion=cfg.training.num_insertion,
        max_insert=cfg.training.max_insert,
        amp_tol=cfg.training.prune_amp_tol,
        # Always emit the progress tables to the logger so every run's log file is
        # complete. `env.verbose` only controls whether they also stream to the
        # console (configure_logging above) — which interleaves under parallel
        # sweeps, so leave it off and read the per-run run.log instead.
        verbose=True,
    )

    metrics = history.summary_metrics()
    logger.info(
        "best iter %d | neurons %d | rel-L2 %.3e | rel-semiH1 %.3e | rel-H1 %.3e (val)",
        metrics["best_iteration"],
        metrics["best_neurons"],
        metrics["rel_l2_val"],
        metrics["rel_grad_val"],
        metrics["rel_h1_val"],
    )

    artifact = run_dir / f"result_{run.run_id}.pkl"
    with artifact.open("wb") as file:
        pickle.dump(history, file)
    run.add_artifact("fit_history", artifact)

    # Persist the run record into Hydra's per-run output dir (Hydra also writes
    # .hydra/config.yaml). MLflow can be added as a backend behind this interface.
    run.log_metrics(metrics)
    record = run.finish(status="completed")
    logger.info("run record: %s", record)


if __name__ == "__main__":
    main()
