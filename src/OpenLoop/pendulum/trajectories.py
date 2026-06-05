"""Backward-PMP trajectory integration for pendulum value samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from src.OpenLoop.pendulum.problem import PendulumSwingUpProblem


@dataclass(frozen=True)
class PmpTrajectory:
    """One raw backward-PMP trajectory before branch restriction."""

    boundary_angle: float
    tau: np.ndarray
    state: np.ndarray
    costate: np.ndarray
    value: np.ndarray
    control: np.ndarray
    hamiltonian: np.ndarray
    trajectory_id: int = -1
    success: bool = True
    hit_value_event: bool = False
    message: str = ""


def backward_pmp_rhs(
    problem: PendulumSwingUpProblem,
    _tau: float,
    z: np.ndarray,
) -> np.ndarray:
    """PMP characteristic ODE integrated away from the local LQR region."""
    state = z[0:2]
    costate = z[2:4]
    control = float(problem.minimizing_control(costate))
    state_rhs = problem.dynamics(state, control)
    costate_rhs = problem.costate_rhs(state, costate)
    value_rhs = float(problem.running_cost(state, control))
    return np.array(
        [
            -state_rhs[0],
            -state_rhs[1],
            -costate_rhs[0],
            -costate_rhs[1],
            value_rhs,
        ],
        dtype=np.float64,
    )


def integrate_pmp_trajectory(
    problem: PendulumSwingUpProblem,
    angle: float,
    epsilon: float,
    value_max: float,
    t_final: float,
    max_step: float,
    rtol: float,
    atol: float,
    trajectory_id: int = -1,
) -> PmpTrajectory:
    """Integrate one raw trajectory from a local-LQR boundary point."""
    if value_max <= epsilon:
        raise ValueError("value_max must be larger than epsilon")
    if t_final <= 0.0:
        raise ValueError("t_final must be positive")
    if max_step <= 0.0:
        raise ValueError("max_step must be positive")
    if rtol <= 0.0 or atol <= 0.0:
        raise ValueError("rtol and atol must be positive")

    state0 = problem.boundary_state(angle, epsilon)
    costate0 = problem.boundary_costate(state0)
    value0 = problem.local_lqr_value(state0)
    z0 = np.array([state0[0], state0[1], costate0[0], costate0[1], value0])

    def value_event(_tau: float, z: np.ndarray) -> float:
        return float(z[4] - value_max)

    value_event.terminal = True
    value_event.direction = 1

    solution = solve_ivp(
        lambda tau, z: backward_pmp_rhs(problem, tau, z),
        (0.0, t_final),
        z0,
        method="DOP853",
        events=value_event,
        max_step=max_step,
        rtol=rtol,
        atol=atol,
    )

    states = solution.y[0:2, :].T
    costates = solution.y[2:4, :].T
    controls = problem.minimizing_control(costates)
    values = solution.y[4, :].copy()
    return PmpTrajectory(
        boundary_angle=float(angle),
        tau=solution.t.copy(),
        state=np.asarray(states, dtype=np.float64),
        costate=np.asarray(costates, dtype=np.float64),
        value=np.asarray(values, dtype=np.float64),
        control=np.asarray(controls, dtype=np.float64),
        hamiltonian=np.asarray(problem.hjb_residual(states, costates), dtype=np.float64),
        trajectory_id=int(trajectory_id),
        success=bool(solution.success),
        hit_value_event=bool(solution.t_events[0].size > 0),
        message=str(solution.message),
    )


def uniform_boundary_angles(num_trajectories: int) -> np.ndarray:
    if num_trajectories <= 0:
        raise ValueError("num_trajectories must be positive")
    return np.linspace(0.0, 2.0 * np.pi, num_trajectories, endpoint=False)


def _circular_midpoint(first: float, second: float) -> float:
    gap = (second - first) % (2.0 * np.pi)
    return (first + 0.5 * gap) % (2.0 * np.pi)


def adaptive_boundary_angles(
    integrate_angle: Callable[[float], PmpTrajectory],
    problem: PendulumSwingUpProblem,
    num_trajectories: int,
    epsilon: float,
    reference_value: float,
    boundary_distance_power: float = 0.8,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Choose boundary angles by refining gaps on a reference value contour."""
    if num_trajectories <= 0:
        raise ValueError("num_trajectories must be positive")
    if num_trajectories <= 4:
        return np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])[
            :num_trajectories
        ]
    if reference_value <= epsilon:
        raise ValueError("reference_value must be larger than epsilon")
    if boundary_distance_power <= 0.0:
        raise ValueError("boundary_distance_power must be positive")

    angles: list[float] = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
    boundary_states = [problem.boundary_state(angle, epsilon) for angle in angles]
    reference_states = [
        _state_at_value(integrate_angle(angle), reference_value) for angle in angles
    ]

    while len(angles) < num_trajectories:
        if progress is not None:
            progress(len(angles), num_trajectories)

        boundary = np.asarray(boundary_states)
        reference = np.asarray(reference_states)
        boundary_next = np.roll(boundary, -1, axis=0)
        reference_next = np.roll(reference, -1, axis=0)
        reference_gap = np.sum(np.abs(reference - reference_next), axis=1)
        boundary_gap = np.sum(
            np.abs(boundary - boundary_next) ** boundary_distance_power,
            axis=1,
        )
        insert_after = int(np.argmax(reference_gap * boundary_gap))
        insert_before = (insert_after + 1) % len(angles)
        new_angle = _circular_midpoint(angles[insert_after], angles[insert_before])

        insert_at = insert_after + 1
        angles.insert(insert_at, new_angle)
        boundary_states.insert(insert_at, problem.boundary_state(new_angle, epsilon))
        reference_states.insert(insert_at, _state_at_value(integrate_angle(new_angle), reference_value))

    if progress is not None:
        progress(num_trajectories, num_trajectories)
    return np.asarray(angles, dtype=np.float64)


def _state_at_value(trajectory: PmpTrajectory, value: float) -> np.ndarray:
    """Interpolate a trajectory point on a requested value contour."""
    values = trajectory.value
    states = trajectory.state
    if value <= values[0]:
        return states[0].copy()
    if value >= values[-1]:
        return states[-1].copy()

    upper = int(np.searchsorted(values, value, side="left"))
    lower = max(0, upper - 1)
    span = values[upper] - values[lower]
    if span <= 0.0:
        return states[lower].copy()
    weight = (value - values[lower]) / span
    return (1.0 - weight) * states[lower] + weight * states[upper]


__all__ = [
    "PmpTrajectory",
    "adaptive_boundary_angles",
    "backward_pmp_rhs",
    "integrate_pmp_trajectory",
    "uniform_boundary_angles",
]
