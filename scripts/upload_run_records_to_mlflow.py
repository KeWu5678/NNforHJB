#!/usr/bin/env python3
"""Upload existing local Run Records to an MLflow tracking server.

Examples:
    MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
      uv run python scripts/upload_run_records_to_mlflow.py

    uv run python scripts/upload_run_records_to_mlflow.py \
      rawdata/logs/multirun/activationsearch/1069/*.json \
      --tracking-uri http://127.0.0.1:5000

    # Upload only the latest FULL sweep (by its multirun.yaml launch marker):
    uv run python scripts/upload_run_records_to_mlflow.py \
      rawdata/logs/multirun/activationsearch --latest-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.experiment_logging import publish_record_to_mlflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[REPO_ROOT / "rawdata" / "logs" / "multirun"],
        help="Run Record JSON files or directories to scan recursively.",
    )
    parser.add_argument(
        "--tracking-uri",
        help="MLflow tracking URI. Defaults to MLFLOW_TRACKING_URI.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List records that would be uploaded without calling MLflow.",
    )
    parser.add_argument(
        "--latest-run",
        action="store_true",
        help=(
            "For each directory input, upload only the latest FULL sweep: the "
            "records written at/after the sweep's multirun.yaml launch marker, "
            "excluding stale job dirs left over from earlier sweeps."
        ),
    )
    return parser.parse_args()


def discover_records(paths: list[Path], *, latest_run: bool = False) -> list[Path]:
    records: list[Path] = []
    for path in paths:
        path = path.expanduser()
        if path.is_dir():
            records.extend(latest_run_records(path) if latest_run else records_under(path))
        elif path.is_file() and path.suffix == ".json":
            records.append(path)
        else:
            raise FileNotFoundError(f"not a JSON file or directory: {path}")
    return sorted(set(records))


def records_under(path: Path) -> list[Path]:
    return [
        candidate
        for candidate in path.rglob("*.json")
        if ".hydra" not in candidate.parts
    ]


def latest_run_records(path: Path) -> list[Path]:
    """Records from the latest full sweep rooted at (or under) ``path``.

    Hydra rewrites ``multirun.yaml`` at the sweep root every time a sweep
    launches, so its mtime marks when the latest sweep started. Job records from
    that sweep are written afterwards; leftover dirs from earlier sweeps that the
    new sweep did not overwrite keep older mtimes. We take the newest
    ``multirun.yaml`` under ``path`` and return its sweep root's records with
    mtime >= that launch marker — the full latest run, minus stale leftovers.
    """
    markers = sorted(path.rglob("multirun.yaml"), key=lambda p: p.stat().st_mtime)
    if not markers:
        return records_under(path)
    launch_marker = markers[-1]
    cutoff = launch_marker.stat().st_mtime
    sweep_root = launch_marker.parent
    return [
        record
        for record in records_under(sweep_root)
        if record.stat().st_mtime >= cutoff
    ]


def load_record(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    record.setdefault("hydra", hydra_metadata_from_run_dir(path.parent))
    return record


def hydra_metadata_from_run_dir(run_dir: Path) -> dict[str, Any]:
    hydra_dir = run_dir / ".hydra"
    task_overrides = _load_yaml_list(hydra_dir / "overrides.yaml")
    return {
        "output_dir": str(run_dir),
        "runtime": {
            "choices": choices_from_overrides(task_overrides),
        },
        "overrides": {
            "task": task_overrides,
        },
    }


def _load_yaml_list(path: Path) -> list[Any]:
    if not path.exists():
        return []
    value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return value if isinstance(value, list) else []


def choices_from_overrides(overrides: list[Any]) -> dict[str, str]:
    choices: dict[str, str] = {}
    for override in overrides:
        if not isinstance(override, str) or "=" not in override:
            continue
        key, value = override.split("=", 1)
        key = key.removeprefix("+")
        if "." not in key:
            choices[key] = value
    return choices


def main() -> None:
    args = parse_args()
    if not args.dry_run and not (args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")):
        raise SystemExit("Set MLFLOW_TRACKING_URI or pass --tracking-uri.")
    records = discover_records(args.paths, latest_run=args.latest_run)
    if not records:
        print("No Run Record JSON files found.")
        return

    for path in records:
        if args.dry_run:
            print(path)
            continue
        record = load_record(path)
        publish_record_to_mlflow(record, path, tracking_uri=args.tracking_uri)
        print(f"uploaded {path}")


if __name__ == "__main__":
    main()
