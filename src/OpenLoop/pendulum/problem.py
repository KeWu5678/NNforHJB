"""Mathematical pendulum swing-up optimal-control problem."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_continuous_are


@dataclass(frozen=True)
class PendulumSwingUpProblem:
    """Infinite-horizon pendulum swing-up OCP from Han and Yang."""

    mass: float = 1.0
    length: float = 1.0
    damping: float = 0.1
    gravity: float = 9.8
    state_weights: tuple[float, float] = (1.0, 1.0)
    control_weight: float = 1.0
    control_limit: float | None = None
    control_gain: float = field(init=False)
    damping_gain: float = field(init=False)
    gravity_gain: float = field(init=False)
    local_lqr_matrix: np.ndarray = field(init=False, repr=False)
    local_lqr_eigvecs: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mass = float(self.mass)
        length = float(self.length)
        damping = float(self.damping)
        gravity = float(self.gravity)
        weights = np.asarray(self.state_weights, dtype=np.float64)
        control_weight = float(self.control_weight)
        control_limit = None if self.control_limit is None else float(self.control_limit)

        if mass <= 0.0:
            raise ValueError("mass must be positive")
        if length <= 0.0:
            raise ValueError("length must be positive")
        if weights.shape != (2,):
            raise ValueError("state_weights must contain two values")
        if np.any(weights < 0.0):
            raise ValueError("state_weights must be nonnegative")
        if control_weight <= 0.0:
            raise ValueError("control_weight must be positive")
        if control_limit is not None and control_limit <= 0.0:
            raise ValueError("control_limit must be positive when provided")

        control_gain = 1.0 / (mass * length**2)
        damping_gain = damping / (mass * length**2)
        gravity_gain = gravity / length
        lqr = self._compute_local_lqr_matrix(
            control_gain,
            damping_gain,
            gravity_gain,
            weights,
            control_weight,
        )

        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "length", length)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "gravity", gravity)
        object.__setattr__(self, "state_weights", (float(weights[0]), float(weights[1])))
        object.__setattr__(self, "control_weight", control_weight)
        object.__setattr__(self, "control_limit", control_limit)
        object.__setattr__(self, "control_gain", control_gain)
        object.__setattr__(self, "damping_gain", damping_gain)
        object.__setattr__(self, "gravity_gain", gravity_gain)
        object.__setattr__(self, "local_lqr_matrix", lqr)
        object.__setattr__(self, "local_lqr_eigvecs", np.linalg.eigh(lqr)[1])

    @staticmethod
    def wrap_angle(theta: np.ndarray | float) -> np.ndarray | float:
        return (np.asarray(theta) + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _compute_local_lqr_matrix(
        control_gain: float,
        damping_gain: float,
        gravity_gain: float,
        state_weights: np.ndarray,
        control_weight: float,
    ) -> np.ndarray:
        linearized_dynamics = np.array(
            [[0.0, 1.0], [gravity_gain, -damping_gain]],
            dtype=np.float64,
        )
        control_matrix = np.array([[0.0], [control_gain]], dtype=np.float64)
        q = np.diag(state_weights)
        r = np.array([[control_weight]], dtype=np.float64)
        return solve_continuous_are(linearized_dynamics, control_matrix, q, r)

    def bounded_control(self, control: float | np.ndarray) -> float | np.ndarray:
        if self.control_limit is None:
            return control
        return np.clip(control, -self.control_limit, self.control_limit)

    def dynamics(self, state: np.ndarray, control: np.ndarray | float) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        theta = state[..., 0]
        omega = state[..., 1]
        control_value = self.bounded_control(np.asarray(control, dtype=np.float64))
        return np.stack(
            [
                omega,
                -self.damping_gain * omega
                + self.gravity_gain * np.sin(theta)
                + self.control_gain * control_value,
            ],
            axis=-1,
        )

    def costate_rhs(self, state: np.ndarray, costate: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        costate = np.asarray(costate, dtype=np.float64)
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
        state = np.asarray(state, dtype=np.float64)
        theta = state[..., 0]
        omega = state[..., 1]
        control_value = self.bounded_control(np.asarray(control, dtype=np.float64))
        q1, q2 = self.state_weights
        return (
            q1 * (2.0 - 2.0 * np.cos(theta))
            + q2 * omega * omega
            + self.control_weight * control_value * control_value
        )

    def stationarity_residual(
        self,
        costate: np.ndarray,
        control: np.ndarray | float,
    ) -> np.ndarray:
        p2 = np.asarray(costate, dtype=np.float64)[..., 1]
        control_value = self.bounded_control(np.asarray(control, dtype=np.float64))
        return self.control_gain * p2 + 2.0 * self.control_weight * control_value

    def minimizing_control(self, costate: np.ndarray) -> np.ndarray:
        p2 = np.asarray(costate, dtype=np.float64)[..., 1]
        control = -self.control_gain * p2 / (2.0 * self.control_weight)
        return np.asarray(self.bounded_control(control), dtype=np.float64)

    def local_lqr_value(self, state: np.ndarray, weight: float = 1.0) -> float:
        state = np.asarray(state, dtype=np.float64).copy()
        state[..., 0] = self.wrap_angle(state[..., 0])
        return float(weight * state @ self.local_lqr_matrix @ state)

    def local_lqr_gradient(self, state: np.ndarray, weight: float = 1.0) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64).copy()
        state[..., 0] = self.wrap_angle(state[..., 0])
        return 2.0 * weight * self.local_lqr_matrix @ state

    def boundary_state(self, angle: float, epsilon: float) -> np.ndarray:
        """Point on the local LQR level set used as PMP terminal data."""
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        direction = (
            self.local_lqr_eigvecs[:, 0] * np.cos(angle)
            + self.local_lqr_eigvecs[:, 1] * np.sin(angle)
        )
        scale = np.sqrt(epsilon / float(direction @ self.local_lqr_matrix @ direction))
        return np.asarray(scale * direction, dtype=np.float64)

    def boundary_costate(self, state: np.ndarray) -> np.ndarray:
        return self.local_lqr_gradient(state)

    def hjb_residual(
        self,
        state: np.ndarray,
        costate: np.ndarray,
        control: np.ndarray | float | None = None,
    ) -> np.ndarray:
        if control is None:
            control = self.minimizing_control(costate)
        dynamics = self.dynamics(state, control)
        running = self.running_cost(state, control)
        return running + np.sum(np.asarray(costate, dtype=np.float64) * dynamics, axis=-1)


__all__ = ["PendulumSwingUpProblem"]
