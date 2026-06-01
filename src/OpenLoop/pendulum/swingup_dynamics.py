"""Shared pendulum swing-up Hamiltonian equations."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are


class PendulumSwingUpDynamics:
    """Physics, cost, and Hamiltonian helpers for the pendulum swing-up problem."""

    def __init__(
        self,
        mass: float = 1.0,
        length: float = 1.0,
        damping: float = 0.1,
        gravity: float = 9.8,
        state_weights: tuple[float, float] = (1.0, 1.0),
        control_weight: float = 1.0,
        control_limit: float | None = None,
    ) -> None:
        self.mass = float(mass)
        self.length = float(length)
        self.damping = float(damping)
        self.gravity = float(gravity)
        self.state_weights = np.asarray(state_weights, dtype=float)
        self.control_weight = float(control_weight)
        self.control_limit = None if control_limit is None else float(control_limit)

        if self.mass <= 0.0:
            raise ValueError("mass must be positive")
        if self.length <= 0.0:
            raise ValueError("length must be positive")
        if self.state_weights.shape != (2,):
            raise ValueError("state_weights must contain two values")
        if np.any(self.state_weights < 0.0):
            raise ValueError("state_weights must be nonnegative")
        if self.control_weight <= 0.0:
            raise ValueError("control_weight must be positive")
        if self.control_limit is not None and self.control_limit <= 0.0:
            raise ValueError("control_limit must be positive when provided")

        self.control_gain = 1.0 / (self.mass * self.length**2)
        self.damping_gain = self.damping / (self.mass * self.length**2)
        self.gravity_gain = self.gravity / self.length
        self.local_lqr_matrix = self._compute_local_lqr_matrix()

    @staticmethod
    def wrap_angle(theta: np.ndarray | float) -> np.ndarray | float:
        return (np.asarray(theta) + np.pi) % (2.0 * np.pi) - np.pi

    def _compute_local_lqr_matrix(self) -> np.ndarray:
        a = np.array(
            [[0.0, 1.0], [self.gravity_gain, -self.damping_gain]],
            dtype=float,
        )
        b = np.array([[0.0], [self.control_gain]], dtype=float)
        q = np.diag(self.state_weights)
        r = np.array([[self.control_weight]], dtype=float)
        return solve_continuous_are(a, b, q, r)

    def bounded_control(self, control: float | np.ndarray) -> float | np.ndarray:
        if self.control_limit is None:
            return control
        return np.clip(control, -self.control_limit, self.control_limit)

    def forward_dynamics(
        self,
        state: np.ndarray,
        control: np.ndarray | float,
    ) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        theta = state[..., 0]
        omega = state[..., 1]
        control_value = self.bounded_control(np.asarray(control, dtype=float))
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

    def running_cost(
        self,
        state: np.ndarray,
        control: np.ndarray | float,
    ) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        theta = state[..., 0]
        omega = state[..., 1]
        control_value = self.bounded_control(np.asarray(control, dtype=float))
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
        p2 = np.asarray(costate, dtype=float)[..., 1]
        control_value = self.bounded_control(np.asarray(control, dtype=float))
        return self.control_gain * p2 + 2.0 * self.control_weight * control_value

    def minimizing_control(self, costate: np.ndarray) -> np.ndarray:
        p2 = np.asarray(costate, dtype=float)[..., 1]
        control = -self.control_gain * p2 / (2.0 * self.control_weight)
        return np.asarray(self.bounded_control(control), dtype=float)

    def local_lqr_value(self, state: np.ndarray, weight: float = 1.0) -> float:
        state = np.asarray(state, dtype=float).copy()
        state[..., 0] = self.wrap_angle(state[..., 0])
        return float(weight * state @ self.local_lqr_matrix @ state)

    def local_lqr_gradient(self, state: np.ndarray, weight: float = 1.0) -> np.ndarray:
        state = np.asarray(state, dtype=float).copy()
        state[..., 0] = self.wrap_angle(state[..., 0])
        return 2.0 * weight * self.local_lqr_matrix @ state


__all__ = ["PendulumSwingUpDynamics"]
