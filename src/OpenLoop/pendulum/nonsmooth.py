"""Nonsmooth-curve detection and branch restriction.

The optimal value function of pendulum swing-up is nonsmooth across a *switching
set* where the clockwise and counter-clockwise swing-up branches assign equal
value to the same physical state (Han & Yang, arXiv:2312.17467). This module:

  1. detects that switching set as ordered *spiral arms* — per value level, the
     equal-value contour crosses its own 2π-shifted copy at exactly 4 vertices
     (the paper's ``SID==0`` points); the 4 vertices are tracked across levels by
     optimal assignment into 4 arms;
  2. assembles the closed *smooth basin* of the upright equilibrium from those
     arms via the problem's reflection symmetry ``(θ,ω)→(2π−θ,−ω)`` plus a −2π
     copy (the reference's ``bsp`` construction);
  3. restricts each raw PMP trajectory to that basin by point-in-polygon
     containment, truncating at the first exit (the reference's ``inpolygon`` cut).

This replaces an earlier buggy approach that connected the switching points into
one value-ordered polyline and cut on ``LineString`` intersection — whose
cross-arm chords swept through the origin and truncated every trajectory ~25×
too early (GitHub issue #18; see experiments/pendulum_branch_restriction/).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.affinity import translate
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point, Polygon
from shapely.prepared import prep

from src.OpenLoop.pendulum.trajectories import PmpTrajectory

TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class NonsmoothCurve:
    """The pendulum switching set (``points``) and the upright smooth basin.

    ``points`` / ``value_levels`` are the ordered switching-set (spiral-arm)
    samples with their cost-to-go levels — the curve the value function is
    nonsmooth across, used downstream for distance-to-switching-set diagnostics.
    ``basin`` is the closed smooth-basin boundary (``(m, 2)`` exterior ring of the
    upright equilibrium's region) used to restrict trajectories.
    """

    points: np.ndarray
    value_levels: np.ndarray
    basin: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        value_levels = np.asarray(self.value_levels, dtype=np.float64)
        if points.size == 0:
            points = np.empty((0, 2), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape (n, 2)")
        if value_levels.shape != (points.shape[0],):
            raise ValueError("value_levels must have shape (n,)")
        basin = np.empty((0, 2)) if self.basin is None else np.asarray(self.basin, dtype=np.float64)
        if basin.size and (basin.ndim != 2 or basin.shape[1] != 2):
            raise ValueError("basin must have shape (m, 2)")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "value_levels", value_levels)
        object.__setattr__(self, "basin", basin)

    @property
    def is_empty(self) -> bool:
        return self.points.shape[0] == 0

    def as_linestring(self) -> LineString | MultiLineString | None:
        if self.points.shape[0] < 2:
            return None
        arm_len = self.points.shape[0] // 4
        if (
            arm_len >= 2
            and self.points.shape[0] == 4 * arm_len
            and np.allclose(self.value_levels[:arm_len], self.value_levels[arm_len : 2 * arm_len])
            and np.allclose(self.value_levels[:arm_len], self.value_levels[2 * arm_len : 3 * arm_len])
            and np.allclose(self.value_levels[:arm_len], self.value_levels[3 * arm_len :])
        ):
            return MultiLineString(
                [self.points[i * arm_len : (i + 1) * arm_len] for i in range(4)]
            )
        return LineString(self.points)

    def basin_polygon(self) -> Polygon | None:
        if self.basin.shape[0] < 3:
            return None
        poly = Polygon(self.basin)
        return poly if poly.is_valid else poly.buffer(0)

    def save_npz(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            points=self.points,
            value_levels=self.value_levels,
            basin=self.basin,
        )
        return output_path

    @classmethod
    def load_npz(cls, path: str | Path) -> "NonsmoothCurve":
        with np.load(path) as data:
            basin = data["basin"] if "basin" in data.files else None
            return cls(
                points=data["points"],
                value_levels=data["value_levels"],
                basin=basin,
            )


def compute_nonsmooth_curve(
    trajectories: tuple[PmpTrajectory, ...],
    value_delta: float,
    value_max: float | None = None,
    *,
    basin_value_max: float = 35.0,
    angle_period: float = TWO_PI,
) -> NonsmoothCurve:
    """Detect the switching-set arms and assemble the upright smooth basin.

    ``value_delta`` is the value-level step for arm tracking. ``basin_value_max``
    caps the value extent of the arms used to build the basin: the upright smooth
    region only reaches the first branch collision (value ≈ 26–31 for the paper's
    parameters), so tracking arms further produces leaky basins. It mirrors the
    reference's hand-tuned ``boundary_spiral(1:1252)`` trim (≈ value 31) and may
    need adjusting if the physical parameters change.
    """
    if value_delta <= 0.0:
        raise ValueError("value_delta must be positive")
    if not trajectories:
        return NonsmoothCurve(np.empty((0, 2)), np.empty((0,)))

    usable_max = min(float(np.max(t.value)) for t in trajectories if t.value.size > 0)
    if value_max is not None:
        usable_max = min(usable_max, float(value_max))
    cap = min(basin_value_max, usable_max)
    # Arm tracking needs a fine value step to resolve the spiral arms (a coarse
    # step gives too few arm points to assemble a non-degenerate basin); cap the
    # caller's value_delta at 0.25.
    step = min(value_delta, 0.25)
    levels = np.arange(step, cap + 0.5 * step, step)

    tracked = _track_arms(trajectories, levels, angle_period)
    if tracked is None:
        return NonsmoothCurve(np.empty((0, 2)), np.empty((0,)))
    arms, used_levels = tracked

    points = np.vstack(arms)
    # value level per arm point: all arms share the same sequence of contributing
    # levels (one point per tracked level), so tile those levels across the arms.
    value_levels = np.tile(used_levels, len(arms))

    basin = _build_basin(arms, trajectories, basin_value_max, angle_period)
    return NonsmoothCurve(points, value_levels, basin)


def _polygon_crossings(contour: np.ndarray, angle_period: float) -> np.ndarray:
    """The 4 vertices where the contour polygon boundary meets its +period shift."""
    poly = Polygon(contour)
    if not poly.is_valid:
        poly = poly.buffer(0)
    shifted = translate(poly, xoff=angle_period)
    crossing = poly.boundary.intersection(shifted.boundary)
    return np.asarray([[p.x, p.y] for p in _crossing_points(crossing)], dtype=np.float64)


def _track_arms(
    trajectories: tuple[PmpTrajectory, ...],
    levels: np.ndarray,
    angle_period: float,
) -> tuple[list[np.ndarray], np.ndarray] | None:
    """Track the 4 switching-set vertices across value levels into ordered arms.

    Trajectories are ordered by boundary angle so the equal-value contour is a
    proper ring. Across levels the 4 vertices are matched by optimal (Hungarian)
    assignment — robust to the arm-swaps a greedy nearest-neighbour match makes
    where arms approach each other near the (π,0) collision. Returns the 4 arms
    plus the value levels that actually contributed (levels with <4 crossings,
    e.g. below the first collision, are skipped).
    """
    ordered = tuple(sorted(trajectories, key=lambda t: t.boundary_angle))
    arms: np.ndarray | None = None
    collected: list[list[np.ndarray]] = [[], [], [], []]
    used_levels: list[float] = []
    for level in levels:
        contour = equal_value_contour(ordered, float(level))
        if contour.shape[0] < 3:
            continue
        crossings = _polygon_crossings(contour, angle_period)
        if crossings.shape[0] < 4:
            continue
        # Keep the 4 crossings nearest the (π,0) collision; drop higher-order wraps.
        near = np.argsort(np.hypot(crossings[:, 0] - 0.5 * angle_period, crossings[:, 1]))
        crossings = crossings[near[:4]]
        if arms is None:
            order = np.argsort(np.arctan2(crossings[:, 1], crossings[:, 0] - 0.5 * angle_period))
            arms = crossings[order].copy()
        else:
            cost = np.linalg.norm(arms[:, None, :] - crossings[None, :, :], axis=2)
            rows, cols = linear_sum_assignment(cost)
            updated = arms.copy()
            for r, c in zip(rows, cols):
                updated[r] = crossings[c]
            arms = updated
        for i in range(4):
            collected[i].append(arms[i])
        used_levels.append(float(level))
    if arms is None or len(collected[0]) < 2:
        return None
    return [np.asarray(a, dtype=np.float64) for a in collected], np.asarray(used_levels)


def _reflect(points: np.ndarray, angle_period: float) -> np.ndarray:
    """Reflect through the bottom point (period/2, 0): (θ,ω) → (period−θ, −ω)."""
    return np.column_stack([angle_period - points[:, 0], -points[:, 1]])


def _assemble_bsp(a_arm: np.ndarray, c_arm: np.ndarray, angle_period: float) -> Polygon:
    """Reference bsp: join a reflected arm to another, close by reflection + −period."""
    part1 = np.vstack([_reflect(c_arm, angle_period)[::-1], a_arm])
    part2 = np.vstack([part1, _reflect(part1, angle_period)[::-1]])
    ring = np.vstack([part2, np.column_stack([part2[:, 0] - angle_period, part2[:, 1]])[::-1]])
    poly = Polygon(ring)
    return poly if poly.is_valid else poly.buffer(0)


def _build_basin(
    arms: list[np.ndarray],
    trajectories: tuple[PmpTrajectory, ...],
    basin_value_max: float,
    angle_period: float,
) -> np.ndarray:
    """Pick the arm pair whose bsp is the upright smooth basin; return its ring.

    Selection (empirical, parameter-robust — see issue #18 discussion): among arm
    pairs whose bsp is a valid polygon containing the origin, keep those that bound
    the trajectories tightly — graded by the *first-exit cut* itself (not mere
    containment: trajectories spiral and re-enter the basin, so contained points
    include high-value re-entrant points) — requiring retained value_max ≤
    basin_value_max + 5, and take the largest-area such basin. The reference
    instead hard-codes ``a_bsp``/``c_bsp`` index picks, which are parameter-specific.
    """
    probe = trajectories[::8]  # subsample of trajectories for fast grading
    origin = Point(0.0, 0.0)
    best_ring: np.ndarray | None = None
    best_area = -1.0
    for ai in range(len(arms)):
        for ci in range(len(arms)):
            if ai == ci:
                continue
            poly = _assemble_bsp(arms[ai], arms[ci], angle_period)
            if poly.is_empty or not poly.is_valid or not poly.contains(origin):
                continue
            component = _origin_component(poly, origin)
            if component is None or component.area <= best_area:
                continue
            retained_max = _first_exit_value_max(probe, prep(component))
            if retained_max is None or retained_max > basin_value_max + 5.0:
                continue  # empty cut, or leaky basin: trajectories escape the arms
            best_area = component.area
            best_ring = np.asarray(component.exterior.coords, dtype=np.float64)
    return best_ring if best_ring is not None else np.empty((0, 2))


def _first_exit_value_max(trajectories, prepared_basin) -> float | None:
    """Max retained value when each trajectory is cut at its first basin exit."""
    retained = []
    for t in trajectories:
        inside = np.fromiter(
            (prepared_basin.covers(Point(p)) for p in t.state), bool, t.state.shape[0]
        )
        outside = np.flatnonzero(~inside)
        cut = int(outside[0]) if outside.size else t.state.shape[0]
        if cut >= 1:
            retained.append(float(t.value[:cut].max()))
    return max(retained) if retained else None


def _origin_component(poly: Polygon, origin: Point) -> Polygon | None:
    """The single polygon component containing the origin (drop ±period copies)."""
    if poly.geom_type == "Polygon":
        return poly
    for part in poly.geoms:
        if part.contains(origin):
            return part
    return None


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
    """Restrict a trajectory to the smooth basin, truncating at the first exit.

    Mirrors the reference ``inpolygon`` cut: keep the leading run of states inside
    the basin polygon; discard from the first state that leaves it. With no basin
    available the trajectory is returned unchanged.
    """
    basin = curve.basin_polygon()
    if basin is None or trajectory.state.shape[0] < 2:
        return trajectory, 0

    prepared = prep(basin)
    inside = np.fromiter(
        (prepared.covers(Point(p)) for p in trajectory.state), bool, trajectory.state.shape[0]
    )
    outside = np.flatnonzero(~inside)
    if outside.size == 0:
        return trajectory, 0
    cut_index = int(outside[0])
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


def _crossing_points(geometry) -> list[Point]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, Point):
        return [geometry]
    if isinstance(geometry, MultiPoint):
        return list(geometry.geoms)
    points: list[Point] = []
    for item in getattr(geometry, "geoms", []):
        points.extend(_crossing_points(item))
    return points


__all__ = [
    "NonsmoothCurve",
    "compute_nonsmooth_curve",
    "equal_value_contour",
    "restrict_trajectory_to_curve",
]
