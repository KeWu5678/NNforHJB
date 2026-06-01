"""Pendulum swing-up optimal-control problem from Han and Yang's HJB example.

The ODE-constrained control problem is:

    minimize integral_0^infty L(x(t), u(t)) dt

    subject to theta_dot = omega
               omega_dot = -b/(m l^2) omega + g/l sin(theta) + u/(m l^2)
               x(0) = x0
               x(t) -> (0, 0)

The paper computes the nonsmooth infinite-horizon value by starting on a small
local-LQR level set around the upright equilibrium and integrating the
Pontryagin system backward.  Every point reached this way is an open-loop
sample: state x, value V(x), and costate p = dV(x).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are


PENDULUM_DATA_DTYPE = np.dtype(
    [
        ("x", "2float64"),
        ("dv", "2float64"),
        ("v", "float64"),
        ("u", "float64"),
        ("tau", "float64"),
        ("trajectory_id", "int64"),
        ("boundary_angle", "float64"),
        ("periodic_copy", "int64"),
        ("hamiltonian", "float64"),
    ]
)


@dataclass(frozen=True)
class PendulumPmpParameters:
    """Physical and numerical parameters for the paper's pendulum example."""

    mass: float = 1.0
    length: float = 1.0
    damping: float = 0.1
    gravity: float = 9.8
    state_weights: tuple[float, float] = (1.0, 1.0)
    control_weight: float = 1.0
    epsilon: float = 2e-4
    value_max: float = 100.0
    t_final: float = 50.0
    max_step: float = 0.005
    rtol: float = 1e-10
    atol: float = 1e-12
    control_limit: float | None = None


@dataclass
class PendulumPmpTrajectory:
    """One backward-PMP trajectory from the local LQR boundary."""

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


class PendulumPmpSampler:
    """ODE-constrained pendulum OCP and backward-PMP sample generator."""

    def __init__(self, parameters: PendulumPmpParameters | None = None) -> None:
        self.parameters = parameters or PendulumPmpParameters()
        self._validate_parameters()

        p = self.parameters
        self.control_gain = 1.0 / (p.mass * p.length**2)
        self.damping_gain = p.damping / (p.mass * p.length**2)
        self.gravity_gain = p.gravity / p.length
        self.state_weights = np.asarray(p.state_weights, dtype=float)
        self.local_lqr_matrix = self._compute_local_lqr_matrix()
        self.local_lqr_eigvecs = np.linalg.eigh(self.local_lqr_matrix)[1]

    def _validate_parameters(self) -> None:
        p = self.parameters
        if p.mass <= 0.0:
            raise ValueError("mass must be positive")
        if p.length <= 0.0:
            raise ValueError("length must be positive")
        if p.control_weight <= 0.0:
            raise ValueError("control_weight must be positive")
        if p.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if p.value_max <= p.epsilon:
            raise ValueError("value_max must be larger than epsilon")
        if p.t_final <= 0.0:
            raise ValueError("t_final must be positive")
        if p.max_step <= 0.0:
            raise ValueError("max_step must be positive")
        if p.rtol <= 0.0 or p.atol <= 0.0:
            raise ValueError("rtol and atol must be positive")
        if p.control_limit is not None and p.control_limit <= 0.0:
            raise ValueError("control_limit must be positive when provided")
        state_weights = np.asarray(p.state_weights, dtype=float)
        if state_weights.shape != (2,):
            raise ValueError("state_weights must contain two values")
        if np.any(state_weights < 0.0):
            raise ValueError("state_weights must be nonnegative")

    def _compute_local_lqr_matrix(self) -> np.ndarray:
        p = self.parameters
        a = np.array(
            [
                [0.0, 1.0],
                [self.gravity_gain, -self.damping_gain],
            ],
            dtype=float,
        )
        b = np.array([[0.0], [self.control_gain]], dtype=float)
        q = np.diag(np.asarray(p.state_weights, dtype=float))
        r = np.array([[p.control_weight]], dtype=float)
        return solve_continuous_are(a, b, q, r)

    def boundary_state(self, angle: float, epsilon: float | None = None) -> np.ndarray:
        """Return one point on the local LQR level set e.T P e = epsilon."""
        eps = self.parameters.epsilon if epsilon is None else float(epsilon)
        direction = (
            self.local_lqr_eigvecs[:, 0] * np.cos(angle)
            + self.local_lqr_eigvecs[:, 1] * np.sin(angle)
        )
        scale = np.sqrt(eps / float(direction @ self.local_lqr_matrix @ direction))
        return scale * direction

    def boundary_costate(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        return 2.0 * self.local_lqr_matrix @ state

    def local_lqr_value(self, state: np.ndarray) -> float:
        state = np.asarray(state, dtype=float)
        return float(state @ self.local_lqr_matrix @ state)

    def minimizing_control(self, costate: np.ndarray) -> np.ndarray:
        p2 = np.asarray(costate, dtype=float)[..., 1]
        u = -self.control_gain * p2 / (2.0 * self.parameters.control_weight)
        if self.parameters.control_limit is not None:
            u = np.clip(u, -self.parameters.control_limit, self.parameters.control_limit)
        return np.asarray(u, dtype=float)

    def optimal_control(self, costate: np.ndarray) -> np.ndarray:
        """Return the HJB/PMP minimizing control for a value gradient."""
        return self.minimizing_control(costate)

    def stationarity_residual(
        self,
        costate: np.ndarray,
        control: np.ndarray | float | None = None,
    ) -> np.ndarray:
        """Derivative of the unconstrained Hamiltonian with respect to control."""
        if control is None:
            control = self.minimizing_control(costate)
        p2 = np.asarray(costate, dtype=float)[..., 1]
        u = np.asarray(control, dtype=float)
        return self.control_gain * p2 + 2.0 * self.parameters.control_weight * u

    def forward_dynamics(self, state: np.ndarray, control: np.ndarray | float) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        theta = state[..., 0]
        omega = state[..., 1]
        u = np.asarray(control, dtype=float)
        return np.stack(
            [
                omega,
                -self.damping_gain * omega
                + self.gravity_gain * np.sin(theta)
                + self.control_gain * u,
            ],
            axis=-1,
        )

    def forward_costate_rhs(self, state: np.ndarray, costate: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        costate = np.asarray(costate, dtype=float)
        theta = state[..., 0]
        omega = state[..., 1]
        p1 = costate[..., 0]
        p2 = costate[..., 1]
        q1, q2 = self.state_weights
        return np.stack(
            [
                -2.0 * q1 * np.sin(theta) - self.gravity_gain * np.cos(theta) * p2,
                -2.0 * q2 * omega - p1 + self.damping_gain * p2,
            ],
            axis=-1,
        )

    def running_cost(self, state: np.ndarray, control: np.ndarray | float) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        theta = state[..., 0]
        omega = state[..., 1]
        u = np.asarray(control, dtype=float)
        q1, q2 = self.state_weights
        return (
            q1 * (2.0 - 2.0 * np.cos(theta))
            + q2 * omega * omega
            + self.parameters.control_weight * u * u
        )

    def trajectory_cost(
        self,
        time: np.ndarray,
        state: np.ndarray,
        control: np.ndarray,
        terminal_value: float = 0.0,
    ) -> float:
        """Evaluate integral L(x(t), u(t)) dt plus an optional terminal value."""
        time = np.asarray(time, dtype=float)
        state = np.asarray(state, dtype=float)
        control = np.asarray(control, dtype=float)
        if time.ndim != 1:
            raise ValueError("time must be a one-dimensional array")
        if state.shape != (time.size, 2):
            raise ValueError("state must have shape (len(time), 2)")
        if control.shape != (time.size,):
            raise ValueError("control must have shape (len(time),)")
        running = self.running_cost(state, control)
        return float(np.trapezoid(running, time) + terminal_value)

    def hjb_residual(
        self,
        state: np.ndarray,
        costate: np.ndarray,
        control: np.ndarray | float | None = None,
    ) -> np.ndarray:
        """Stationary HJB residual L(x,u) + dV(x).f(x,u)."""
        if control is None:
            control = self.minimizing_control(costate)
        dynamics = self.forward_dynamics(state, control)
        running = self.running_cost(state, control)
        return running + np.sum(np.asarray(costate, dtype=float) * dynamics, axis=-1)

    def backward_pmp_rhs(self, _tau: float, z: np.ndarray) -> np.ndarray:
        state = z[0:2]
        costate = z[2:4]
        control = float(self.minimizing_control(costate))
        state_rhs = self.forward_dynamics(state, control)
        costate_rhs = self.forward_costate_rhs(state, costate)
        value_rhs = float(self.running_cost(state, control))
        return np.array(
            [
                -state_rhs[0],
                -state_rhs[1],
                -costate_rhs[0],
                -costate_rhs[1],
                value_rhs,
            ],
            dtype=float,
        )

    def integrate_angle(
        self,
        angle: float,
        trajectory_id: int = -1,
        value_max: float | None = None,
        t_final: float | None = None,
    ) -> PendulumPmpTrajectory:
        """Integrate one backward-PMP trajectory from a boundary angle."""
        state0 = self.boundary_state(angle)
        costate0 = self.boundary_costate(state0)
        value0 = self.local_lqr_value(state0)
        z0 = np.array([state0[0], state0[1], costate0[0], costate0[1], value0])

        stop_value = self.parameters.value_max if value_max is None else float(value_max)
        end_time = self.parameters.t_final if t_final is None else float(t_final)

        def value_event(_tau: float, z: np.ndarray) -> float:
            return float(z[4] - stop_value)

        value_event.terminal = True
        value_event.direction = 1

        solution = solve_ivp(
            self.backward_pmp_rhs,
            (0.0, end_time),
            z0,
            method="DOP853",
            events=value_event,
            max_step=self.parameters.max_step,
            rtol=self.parameters.rtol,
            atol=self.parameters.atol,
        )

        states = solution.y[0:2, :].T
        costates = solution.y[2:4, :].T
        controls = self.minimizing_control(costates)
        values = solution.y[4, :].copy()
        return PendulumPmpTrajectory(
            boundary_angle=float(angle),
            tau=solution.t.copy(),
            state=states,
            costate=costates,
            value=values,
            control=controls,
            hamiltonian=self.hjb_residual(states, costates),
            trajectory_id=int(trajectory_id),
            success=bool(solution.success),
            hit_value_event=bool(solution.t_events[0].size > 0),
            message=str(solution.message),
        )

    @staticmethod
    def uniform_angles(num_trajectories: int) -> np.ndarray:
        if num_trajectories <= 0:
            raise ValueError("num_trajectories must be positive")
        return np.linspace(0.0, 2.0 * np.pi, num_trajectories, endpoint=False)

    @staticmethod
    def _circular_midpoint(first: float, second: float) -> float:
        gap = (second - first) % (2.0 * np.pi)
        return (first + 0.5 * gap) % (2.0 * np.pi)

    def adaptive_angles(
        self,
        num_trajectories: int,
        reference_value: float | None = None,
        boundary_distance_power: float = 0.8,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Choose boundary angles by refining large gaps on a value contour.

        This mirrors the released MATLAB scripts: start from four cardinal
        angles, integrate each candidate to a reference value level, then insert
        the midpoint of the largest combined boundary/level-set gap.
        """
        if num_trajectories <= 0:
            raise ValueError("num_trajectories must be positive")
        if num_trajectories <= 4:
            return np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])[
                :num_trajectories
            ]

        target_value = (
            min(20.0, self.parameters.value_max)
            if reference_value is None
            else float(reference_value)
        )
        if target_value <= self.parameters.epsilon:
            raise ValueError("reference_value must be larger than epsilon")

        angles: list[float] = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
        boundary_states: list[np.ndarray] = []
        reference_states: list[np.ndarray] = []

        def evaluate_angle(angle: float) -> tuple[np.ndarray, np.ndarray]:
            boundary = self.boundary_state(angle)
            trajectory = self.integrate_angle(
                angle,
                value_max=target_value,
                t_final=self.parameters.t_final,
            )
            return boundary, trajectory.state[-1]

        for angle in angles:
            boundary, reference = evaluate_angle(angle)
            boundary_states.append(boundary)
            reference_states.append(reference)

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
            new_angle = self._circular_midpoint(
                angles[insert_after],
                angles[insert_before],
            )
            new_boundary, new_reference = evaluate_angle(new_angle)

            insert_at = insert_after + 1
            angles.insert(insert_at, new_angle)
            boundary_states.insert(insert_at, new_boundary)
            reference_states.insert(insert_at, new_reference)

        if progress is not None:
            progress(num_trajectories, num_trajectories)
        return np.asarray(angles, dtype=float)

    def sample_trajectories(
        self,
        num_trajectories: int,
        angles: np.ndarray | None = None,
        progress: Callable[[int, int], None] | None = None,
        skip_failures: bool = False,
    ) -> tuple[list[PendulumPmpTrajectory], list[dict[str, object]]]:
        """Sample trajectories around the local LQR boundary."""
        trajectories: list[PendulumPmpTrajectory] = []
        failures: list[dict[str, object]] = []
        angle_values = (
            self.uniform_angles(num_trajectories)
            if angles is None
            else np.asarray(angles, dtype=float)
        )
        if len(angle_values) != num_trajectories:
            raise ValueError("angles must have length num_trajectories")
        for idx, angle in enumerate(angle_values):
            if progress is not None:
                progress(idx, len(angle_values))
            try:
                trajectory = self.integrate_angle(angle, trajectory_id=idx)
            except Exception as exc:
                if not skip_failures:
                    raise
                failures.append(
                    {
                        "trajectory_id": idx,
                        "boundary_angle": float(angle),
                        "message": str(exc),
                    }
                )
                continue
            if not trajectory.success:
                failures.append(
                    {
                        "trajectory_id": idx,
                        "boundary_angle": float(angle),
                        "message": trajectory.message,
                    }
                )
                if not skip_failures:
                    raise RuntimeError(trajectory.message)
            trajectories.append(trajectory)
        if progress is not None:
            progress(len(angle_values), len(angle_values))
        return trajectories, failures

    def trajectory_to_dataset(
        self,
        trajectory: PendulumPmpTrajectory,
        periodic_copy: int = 0,
    ) -> np.ndarray:
        data = np.zeros(trajectory.tau.size, dtype=PENDULUM_DATA_DTYPE)
        states = trajectory.state.copy()
        states[:, 0] += 2.0 * np.pi * periodic_copy
        data["x"] = states
        data["dv"] = trajectory.costate
        data["v"] = trajectory.value
        data["u"] = trajectory.control
        data["tau"] = trajectory.tau
        data["trajectory_id"] = trajectory.trajectory_id
        data["boundary_angle"] = trajectory.boundary_angle
        data["periodic_copy"] = int(periodic_copy)
        data["hamiltonian"] = trajectory.hamiltonian
        return data

    def trajectories_to_dataset(
        self,
        trajectories: Iterable[PendulumPmpTrajectory],
        periodic_copies: int = 0,
        theta_range: tuple[float, float] | None = None,
        omega_range: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Convert trajectories into the structured x/dv/v dataset format."""
        if periodic_copies < 0:
            raise ValueError("periodic_copies must be nonnegative")

        chunks: list[np.ndarray] = []
        for trajectory in trajectories:
            for copy_index in range(-periodic_copies, periodic_copies + 1):
                chunks.append(self.trajectory_to_dataset(trajectory, copy_index))
        if not chunks:
            return np.zeros(0, dtype=PENDULUM_DATA_DTYPE)

        data = np.concatenate(chunks)
        mask = np.ones(data.shape[0], dtype=bool)
        if theta_range is not None:
            lo, hi = theta_range
            mask &= (data["x"][:, 0] >= lo) & (data["x"][:, 0] <= hi)
        if omega_range is not None:
            lo, hi = omega_range
            mask &= (data["x"][:, 1] >= lo) & (data["x"][:, 1] <= hi)
        return data[mask]
