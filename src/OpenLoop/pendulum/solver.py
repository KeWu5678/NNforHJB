"""Paper-aligned pendulum PMP value-sample solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.OpenLoop.pendulum.nonsmooth import (
    NonsmoothCurve,
    compute_nonsmooth_curve,
    restrict_trajectory_to_curve,
)
from src.OpenLoop.pendulum.problem import PendulumSwingUpProblem
from src.OpenLoop.value_samples import ValueSamples
from src.OpenLoop.pendulum.trajectories import (
    PmpTrajectory,
    adaptive_boundary_angles,
    integrate_pmp_trajectory,
    uniform_boundary_angles,
)


@dataclass(frozen=True)
class PendulumPmpSolverConfig:
    """Numerical configuration for one pendulum value-sample run."""

    epsilon: float = 2e-4
    value_max: float = 100.0
    t_final: float = 50.0
    max_step: float = 0.005
    rtol: float = 1e-10
    atol: float = 1e-12
    num_trajectories: int = 256
    adaptive_sampling: bool = True
    reference_value: float | None = None
    boundary_distance_power: float = 0.8
    contour_delta: float = 1.0
    periodic_copies: int = 0

    def __post_init__(self) -> None:
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if self.value_max <= self.epsilon:
            raise ValueError("value_max must be larger than epsilon")
        if self.t_final <= 0.0:
            raise ValueError("t_final must be positive")
        if self.max_step <= 0.0:
            raise ValueError("max_step must be positive")
        if self.rtol <= 0.0 or self.atol <= 0.0:
            raise ValueError("rtol and atol must be positive")
        if self.num_trajectories <= 0:
            raise ValueError("num_trajectories must be positive")
        if self.boundary_distance_power <= 0.0:
            raise ValueError("boundary_distance_power must be positive")
        if self.contour_delta <= 0.0:
            raise ValueError("contour_delta must be positive")
        if self.periodic_copies < 0:
            raise ValueError("periodic_copies must be nonnegative")


@dataclass(frozen=True)
class SolverDiagnostics:
    requested_trajectories: int
    integrated_trajectories: int
    failed_trajectories: tuple[dict[str, object], ...]
    nonsmooth_points: int
    discarded_points: int
    retained_points: int


@dataclass(frozen=True)
class PendulumValueSolution:
    value_samples: ValueSamples
    raw_trajectories: tuple[PmpTrajectory, ...]
    restricted_trajectories: tuple[PmpTrajectory, ...]
    nonsmooth_curve: NonsmoothCurve
    diagnostics: SolverDiagnostics


class PendulumPmpSolver:
    """Run the paper's backward-PMP construction and emit `ValueSamples`."""

    def __init__(
        self,
        problem: PendulumSwingUpProblem | None = None,
        config: PendulumPmpSolverConfig | None = None,
    ) -> None:
        self.problem = problem or PendulumSwingUpProblem()
        self.config = config or PendulumPmpSolverConfig()

    def solve(
        self,
        progress: Callable[[int, int], None] | None = None,
    ) -> PendulumValueSolution:
        raw, failures = self.generate_raw_trajectories(progress=progress)
        curve = self.compute_nonsmooth_curve(raw)
        samples, restricted, discarded = self.build_value_samples(raw, curve)
        diagnostics = SolverDiagnostics(
            requested_trajectories=self.config.num_trajectories,
            integrated_trajectories=len(raw),
            failed_trajectories=tuple(failures),
            nonsmooth_points=int(curve.points.shape[0]),
            discarded_points=int(discarded),
            retained_points=int(samples.size),
        )
        return PendulumValueSolution(
            value_samples=samples,
            raw_trajectories=raw,
            restricted_trajectories=restricted,
            nonsmooth_curve=curve,
            diagnostics=diagnostics,
        )

    def generate_raw_trajectories(
        self,
        angles: np.ndarray | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> tuple[tuple[PmpTrajectory, ...], tuple[dict[str, object], ...]]:
        angle_values = self.sample_boundary_angles(progress=progress) if angles is None else angles
        if len(angle_values) != self.config.num_trajectories:
            raise ValueError("angles must have length num_trajectories")

        trajectories: list[PmpTrajectory] = []
        failures: list[dict[str, object]] = []
        for idx, angle in enumerate(np.asarray(angle_values, dtype=np.float64)):
            if progress is not None:
                progress(idx, len(angle_values))
            try:
                trajectory = self._integrate_angle(float(angle), trajectory_id=idx)
            except Exception as exc:
                failures.append(
                    {
                        "trajectory_id": idx,
                        "boundary_angle": float(angle),
                        "message": str(exc),
                    }
                )
                continue
            if trajectory.success:
                trajectories.append(trajectory)
            else:
                failures.append(
                    {
                        "trajectory_id": idx,
                        "boundary_angle": float(angle),
                        "message": trajectory.message,
                    }
                )
        if progress is not None:
            progress(len(angle_values), len(angle_values))
        return tuple(trajectories), tuple(failures)

    def sample_boundary_angles(
        self,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        if not self.config.adaptive_sampling:
            return uniform_boundary_angles(self.config.num_trajectories)

        # The adaptive sampler uses a reference value contour to spread raw PMP
        # trajectories in the state plane instead of evenly spacing boundary angles.
        reference_value = (
            min(20.0, self.config.value_max)
            if self.config.reference_value is None
            else float(self.config.reference_value)
        )
        return adaptive_boundary_angles(
            lambda angle: self._integrate_angle(angle, trajectory_id=-1),
            self.problem,
            self.config.num_trajectories,
            self.config.epsilon,
            reference_value,
            boundary_distance_power=self.config.boundary_distance_power,
            progress=progress,
        )

    def compute_nonsmooth_curve(
        self,
        trajectories: tuple[PmpTrajectory, ...],
    ) -> NonsmoothCurve:
        return compute_nonsmooth_curve(
            trajectories,
            value_delta=self.config.contour_delta,
            value_max=self.config.value_max,
        )

    def build_value_samples(
        self,
        trajectories: tuple[PmpTrajectory, ...],
        curve: NonsmoothCurve,
    ) -> tuple[ValueSamples, tuple[PmpTrajectory, ...], int]:
        restricted: list[PmpTrajectory] = []
        discarded = 0
        for trajectory in trajectories:
            cut, count = restrict_trajectory_to_curve(trajectory, curve)
            restricted.append(cut)
            discarded += count

        samples = self._trajectories_to_value_samples(tuple(restricted))
        return samples, tuple(restricted), discarded

    def _integrate_angle(self, angle: float, trajectory_id: int) -> PmpTrajectory:
        return integrate_pmp_trajectory(
            self.problem,
            angle=angle,
            epsilon=self.config.epsilon,
            value_max=self.config.value_max,
            t_final=self.config.t_final,
            max_step=self.config.max_step,
            rtol=self.config.rtol,
            atol=self.config.atol,
            trajectory_id=trajectory_id,
        )

    def _trajectories_to_value_samples(
        self,
        trajectories: tuple[PmpTrajectory, ...],
    ) -> ValueSamples:
        chunks: list[ValueSamples] = []
        for trajectory in trajectories:
            for copy_index in range(-self.config.periodic_copies, self.config.periodic_copies + 1):
                states = trajectory.state.copy()
                states[:, 0] += 2.0 * np.pi * copy_index
                chunks.append(
                    ValueSamples(
                        x=states,
                        v=trajectory.value.copy(),
                        dv=trajectory.costate.copy(),
                    )
                )
        return ValueSamples.concatenate(chunks)


__all__ = [
    "PendulumPmpSolver",
    "PendulumPmpSolverConfig",
    "PendulumValueSolution",
    "SolverDiagnostics",
]
