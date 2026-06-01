"""Shared helpers for OpenLoop sample-set generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def grid_initial_states(
    n_first: int,
    n_second: int,
    first_range: tuple[float, float],
    second_range: tuple[float, float],
) -> np.ndarray:
    first_values = np.linspace(first_range[0], first_range[1], n_first)
    second_values = np.linspace(second_range[0], second_range[1], n_second)
    first_grid, second_grid = np.meshgrid(first_values, second_values)
    return np.column_stack((first_grid.ravel(), second_grid.ravel()))


def random_initial_states(
    n_samples: int,
    first_range: tuple[float, float],
    second_range: tuple[float, float],
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    first_values = rng.uniform(first_range[0], first_range[1], n_samples)
    second_values = rng.uniform(second_range[0], second_range[1], n_samples)
    return np.column_stack((first_values, second_values))


def save_dataset_bundle(
    output_dir: str | Path,
    arrays: dict[str, np.ndarray],
    json_files: dict[str, Any] | None = None,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}
    for filename, array in arrays.items():
        path = output_path / filename
        np.save(path, array)
        saved[filename] = path

    for filename, payload in (json_files or {}).items():
        path = output_path / filename
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        saved[filename] = path

    return saved


__all__ = ["grid_initial_states", "random_initial_states", "save_dataset_bundle"]
