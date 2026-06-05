"""Nonsmooth-curve detection and branch restriction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPoint, Point

from src.OpenLoop.pendulum.trajectories import PmpTrajectory


@dataclass(frozen=True)
class NonsmoothCurve:
    """Detected equal-value intersections between periodic pendulum branches."""

    points: np.ndarray
    value_levels: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        value_levels = np.asarray(self.value_levels, dtype=np.float64)
        if points.size == 0:
            points = np.empty((0, 2), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape (n, 2)")
        if value_levels.shape != (points.shape[0],):
            raise ValueError("value_levels must have shape (n,)")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "value_levels", value_levels)

    @property
    def is_empty(self) -> bool:
        return self.points.shape[0] == 0

    def as_linestring(self) -> LineString | None:
        if self.points.shape[0] < 2:
            return None
        return LineString(self.points)


def compute_nonsmooth_curve(
    trajectories: tuple[PmpTrajectory, ...],
    value_delta: float,
    value_max: float | None = None,
    angle_period: float = 2.0 * np.pi,
) -> NonsmoothCurve:
    """Detect the paper's shifted equal-value contour intersections.

    Each value level forms one polyline by taking one point from every raw PMP
    trajectory at that same cost-to-go value. Intersections with a shifted copy
    identify periodic branches assigning equal value to the same physical state.
    """
    if value_delta <= 0.0:
        raise ValueError("value_delta must be positive")
    if not trajectories:
        return NonsmoothCurve(np.empty((0, 2)), np.empty((0,)))

    usable_max = min(float(np.max(t.value)) for t in trajectories if t.value.size > 0)
    if value_max is not None:
        usable_max = min(usable_max, float(value_max))
    first_level = max(float(np.max([t.value[0] for t in trajectories])), value_delta)
    levels = np.arange(first_level, usable_max + 0.5 * value_delta, value_delta)

    points: list[np.ndarray] = []
    point_levels: list[float] = []
    for level in levels:
        contour = equal_value_contour(trajectories, level)
        if contour.shape[0] < 2:
            continue
        line = LineString(contour)
        shifted = LineString(contour + np.array([angle_period, 0.0]))
        for point in _extract_points(line.intersection(shifted)):
            points.append(np.array([point.x, point.y], dtype=np.float64))
            point_levels.append(float(level))

    if not points:
        return NonsmoothCurve(np.empty((0, 2)), np.empty((0,)))

    raw_points = np.vstack(points)
    raw_levels = np.asarray(point_levels, dtype=np.float64)
    unique_points, unique_levels = _deduplicate_points(raw_points, raw_levels)
    order = np.argsort(unique_levels)
    return NonsmoothCurve(unique_points[order], unique_levels[order])


def equal_value_contour(
    trajectories: tuple[PmpTrajectory, ...],
    value_level: float,
) -> np.ndarray:
    """Return one interpolated state per trajectory on a value contour."""
    states = [
        _state_at_value(trajectory, value_level)
        for trajectory in trajectories
        if trajectory.value.size > 1
        and trajectory.value[0] <= value_level <= trajectory.value[-1]
    ]
    if not states:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(states, dtype=np.float64)


def restrict_trajectory_to_curve(
    trajectory: PmpTrajectory,
    curve: NonsmoothCurve,
) -> tuple[PmpTrajectory, int]:
    """Approximate branch restriction by cutting at the first curve crossing."""
    line = curve.as_linestring()
    if line is None or trajectory.state.shape[0] < 2:
        return trajectory, 0

    trajectory_line = LineString(trajectory.state)
    intersections = _extract_points(trajectory_line.intersection(line))
    if not intersections:
        return trajectory, 0

    intersection_array = np.asarray([[p.x, p.y] for p in intersections], dtype=np.float64)
    distances = np.linalg.norm(
        trajectory.state[:, None, :] - intersection_array[None, :, :],
        axis=2,
    )
    cut_index = int(np.min(np.argmin(distances, axis=0)))
    if cut_index <= 0:
        cut_index = 1

    discarded = int(trajectory.state.shape[0] - cut_index)
    restricted = PmpTrajectory(
        boundary_angle=trajectory.boundary_angle,
        tau=trajectory.tau[:cut_index].copy(),
        state=trajectory.state[:cut_index].copy(),
        costate=trajectory.costate[:cut_index].copy(),
        value=trajectory.value[:cut_index].copy(),
        control=trajectory.control[:cut_index].copy(),
        hamiltonian=trajectory.hamiltonian[:cut_index].copy(),
        trajectory_id=trajectory.trajectory_id,
        success=trajectory.success,
        hit_value_event=trajectory.hit_value_event,
        message=trajectory.message,
    )
    return restricted, discarded


def _state_at_value(trajectory: PmpTrajectory, value: float) -> np.ndarray:
    values = trajectory.value
    states = trajectory.state
    upper = int(np.searchsorted(values, value, side="left"))
    if upper <= 0:
        return states[0].copy()
    if upper >= values.size:
        return states[-1].copy()
    lower = upper - 1
    span = values[upper] - values[lower]
    if span <= 0.0:
        return states[lower].copy()
    weight = (value - values[lower]) / span
    return (1.0 - weight) * states[lower] + weight * states[upper]


def _extract_points(geometry) -> list[Point]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, Point):
        return [geometry]
    if isinstance(geometry, MultiPoint):
        return list(geometry.geoms)
    if isinstance(geometry, LineString):
        coords = list(geometry.coords)
        return [Point(coords[0]), Point(coords[-1])] if coords else []
    if isinstance(geometry, MultiLineString):
        points: list[Point] = []
        for line in geometry.geoms:
            points.extend(_extract_points(line))
        return points
    if isinstance(geometry, GeometryCollection):
        points: list[Point] = []
        for item in geometry.geoms:
            points.extend(_extract_points(item))
        return points
    return []


def _deduplicate_points(
    points: np.ndarray,
    value_levels: np.ndarray,
    decimals: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    rounded = np.round(points, decimals=decimals)
    _unique, indices = np.unique(rounded, axis=0, return_index=True)
    indices = np.sort(indices)
    return points[indices], value_levels[indices]


__all__ = [
    "NonsmoothCurve",
    "compute_nonsmooth_curve",
    "equal_value_contour",
    "restrict_trajectory_to_curve",
]
