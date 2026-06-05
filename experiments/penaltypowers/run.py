#!/usr/bin/env python3
"""Run the penaltypowers experiment."""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.PDAP import PDAP
from src.data import load_value_samples, normalize_value_samples
from src.experiment_logging import ExperimentRun
from src.metric import format_table
from src.paths import DATA_DIR, LOGS_DIR


EXPERIMENT = "penaltypowers"
DEFAULT_CONFIG = REPO_ROOT / "conf" / "experiment" / f"{EXPERIMENT}.yaml"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / EXPERIMENT


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def number_label(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def run_id(*, power: float, loss: str, gamma: float, seed: int) -> str:
    return f"power{number_label(power)}_{loss}_gamma{number_label(gamma)}_seed{seed}"


def make_pdap_cfg(raw: dict[str, Any], *, power: float, loss: str, gamma: float, seed: int):
    model = raw["model"]
    training = raw["training"]
    return OmegaConf.create({
        "name": EXPERIMENT,
        "model": {
            "kind": model["kind"],
            "insertion": model["insertion"],
            "activation": model["activation"],
            "power": float(power),
            "loss_weights": loss,
            "use_sphere": None,
            "c_init": 1.0,
            "alpha": float(model["alpha"]),
            "gamma": float(gamma),
            "th": 0.5,
        },
        "training": dict(training),
        "data": dict(raw["data"]),
        "env": {
            "seed": int(seed),
            "verbose": False,
            "log_level": "INFO",
            "log_file": None,
        },
    })


def run_point(experiment_name: str, raw: dict[str, Any], data: dict[str, np.ndarray],
              normalizer: dict[str, Any] | None,
              *, power: float, loss: str, gamma: float, seed: int,
              record_dir: Path, artifact_dir: Path) -> dict[str, Any]:
    set_seed(seed)
    cfg = make_pdap_cfg(raw, power=power, loss=loss, gamma=gamma, seed=seed)
    pdap = PDAP.from_config(cfg, data)
    started = time.time()
    result = pdap.fit_from_config(cfg.training, verbose=False)
    elapsed_s = round(time.time() - started, 3)

    rid = run_id(power=power, loss=loss, gamma=gamma, seed=seed)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_path = artifact_dir / f"{rid}_result.pkl"
    with result_path.open("wb") as file:
        pickle.dump(result, file)

    bi = int(result["best_iteration"])
    row = {
        "experiment": experiment_name,
        "power": float(power),
        "loss": loss,
        "gamma": float(gamma),
        "seed": int(seed),
        "best_iteration": bi + 1,
        "neurons": int(result["best_neurons"]),
        "train_l2": float(result["err_l2_train"][bi]),
        "train_h1": float(result["err_h1_train"][bi]),
        "val_l2": float(result["err_l2_val"][bi]),
        "val_h1": float(result["err_h1_val"][bi]),
        "score": float(result["err_h1_val"][bi]) * max(int(result["best_neurons"]), 1),
        "elapsed_s": elapsed_s,
        "result_artifact": str(result_path.relative_to(REPO_ROOT)),
    }

    run = ExperimentRun(
        output_dir=record_dir,
        name=experiment_name,
        run_id=rid,
        config={
            "power": float(power),
            "loss": loss,
            "gamma": float(gamma),
            "seed": int(seed),
            "model": dict(cfg.model),
            "training": dict(cfg.training),
            "data": dict(raw["data"]),
            "normalizer": normalizer,
        },
    )
    run.log_metrics({key: row[key] for key in ("train_l2", "train_h1", "val_l2", "val_h1", "score", "neurons")})
    run.add_artifact("fit_result", result_path)
    row["run_record"] = str(run.finish(status="completed", summary=row).relative_to(REPO_ROOT))
    return row


def write_outputs(rows: list[dict[str, Any]]) -> None:
    from src.plots import plot_score_tradeoff

    experiment_name = str(rows[0]["experiment"])
    experiment_dir = REPO_ROOT / "experiments" / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    results_path = experiment_dir / "results.md"
    best_rows = []
    for key, group in itertools.groupby(
        sorted(rows, key=lambda r: (r["power"], r["loss"], r["seed"], r["score"])),
        key=lambda r: (r["power"], r["loss"], r["seed"]),
    ):
        best_rows.append(min(list(group), key=lambda r: r["score"]))

    table = format_table(
        best_rows,
        ["power", "loss", "seed", "gamma", "neurons", "val_h1", "score"],
        headers={"val_h1": "Val H1"},
        formats={"power": "{:g}", "gamma": "{:g}", "val_h1": "{:.2e}", "score": "{:.2e}"},
        title="Best gamma per power/loss/seed",
    )
    results_path.write_text(
        f"# {experiment_name} Results\n\n"
        f"{table}\n\n"
        "Generated from Run Records and fit-result artifacts.\n",
        encoding="utf-8",
    )
    plot_score_tradeoff(
        best_rows,
        x="neurons",
        y="val_h1",
        label="power",
        color="loss",
        title="penaltypowers: sparsity/accuracy tradeoff",
        xlabel="Neurons",
        ylabel="Validation H1",
        save_path=experiment_dir / "figures" / "tradeoff.png",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, default=None, help="run only the first N config points")
    parser.add_argument("--output-name", default=EXPERIMENT, help="experiment output directory/name")
    parser.add_argument("--no-normalize", action="store_true", help="disable data normalization for this run")
    args = parser.parse_args()

    raw = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if args.no_normalize:
        raw["data"]["normalize"] = False
    data_path = Path(raw["data"]["path"])
    if not data_path.is_absolute():
        data_path = DATA_DIR / data_path
    data = load_value_samples(data_path)
    normalizer_info = None
    if raw["data"].get("normalize", True):
        data, normalizer = normalize_value_samples(data)
        normalizer_info = normalizer.to_dict()

    record_dir = LOGS_DIR / args.output_name
    artifact_dir = DATA_DIR / args.output_name
    rows = []
    points = itertools.product(raw["seeds"], raw["powers"], raw["losses"], raw["gammas"])
    for index, (seed, power, loss, gamma) in enumerate(points):
        if args.limit is not None and index >= args.limit:
            break
        row = run_point(
            args.output_name, raw, data, normalizer_info,
            power=power, loss=loss, gamma=gamma, seed=seed,
            record_dir=record_dir, artifact_dir=artifact_dir,
        )
        rows.append(row)
        print(json.dumps(row))
    if rows:
        write_outputs(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
