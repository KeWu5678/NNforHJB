"""VDP-local initial-state sampling helpers."""

from __future__ import annotations

import numpy as np


def grid_initial_states(
    n_first: int,
    n_second: int,
    first_range: tuple[float, float] = (-3.0, 3.0),
    second_range: tuple[float, float] = (-3.0, 3.0),
) -> np.ndarray:
    first_values = np.linspace(first_range[0], first_range[1], n_first)
    second_values = np.linspace(second_range[0], second_range[1], n_second)
    first_grid, second_grid = np.meshgrid(first_values, second_values)
    return np.column_stack((first_grid.ravel(), second_grid.ravel())).astype(np.float64)


def random_initial_states(
    n_samples: int,
    first_range: tuple[float, float] = (-3.0, 3.0),
    second_range: tuple[float, float] = (-3.0, 3.0),
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    first_values = rng.uniform(first_range[0], first_range[1], n_samples)
    second_values = rng.uniform(second_range[0], second_range[1], n_samples)
    return np.column_stack((first_values, second_values)).astype(np.float64)


__all__ = ["grid_initial_states", "random_initial_states"]
