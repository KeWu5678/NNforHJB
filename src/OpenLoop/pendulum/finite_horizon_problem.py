"""Pendulum swing-up problem for forward-backward finite-horizon solves."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.linalg import solve_continuous_are

from src.OpenLoop.forward_backward_optimizer import (
    ForwardBackwardOpenLoopOptimizer,
    ForwardBackwardOpenLoopResult,
)


class PendulumSwingUpProblem:
    """Pendulum swing-up open-loop problem from Han and Yang (2024)."""

    def __init__(
        self,
        mass: float = 1.0,
        length: float = 1.0,
        damping: float = 0.1,
        gravity: float = 9.8,
        state_weights: tuple[float, float] = (1.0, 1.0),
        control_weight: float = 1.0,
        terminal_weight: float = 1.0,
        T_final: float = 6.0,
        control_limit: Optional[float] = None,
    ) -> None:
        self.mass = float(mass)
        self.length = float(length)
        self.damping = float(damping)
        self.gravity = float(gravity)
        self.state_weights = np.asarray(state_weights, dtype=float)
        self.control_weight = float(control_weight)
        self.terminal_weight = float(terminal_weight)
        self.T_final = float(T_final)
        self.control_limit = None if control_limit is None else float(control_limit)
        self.fixed_target_signs = (1,)

        if self.mass <= 0:
            raise ValueError("mass must be positive")
        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.state_weights.shape != (2,):
            raise ValueError("state_weights must contain two values")
        if np.any(self.state_weights < 0):
            raise ValueError("state_weights must be nonnegative")
        if self.control_weight <= 0:
            raise ValueError("control_weight must be positive")
        if self.terminal_weight < 0:
            raise ValueError("terminal_weight must be nonnegative")
        if self.control_limit is not None and self.control_limit <= 0:
            raise ValueError("control_limit must be positive when provided")

        self._control_gain = 1.0 / (self.mass * self.length**2)
        self._damping_gain = self.damping / (self.mass * self.length**2)
        self._gravity_gain = self.gravity / self.length
        self.local_lqr_matrix = self._compute_local_lqr_matrix()

    @staticmethod
    def wrap_angle(theta: np.ndarray | float) -> np.ndarray | float:
        """Map angles to [-pi, pi) for the local terminal LQR penalty."""
        return (np.asarray(theta) + np.pi) % (2.0 * np.pi) - np.pi

    def _compute_local_lqr_matrix(self) -> np.ndarray:
        A = np.array(
            [
                [0.0, 1.0],
                [self._gravity_gain, -self._damping_gain],
            ],
            dtype=float,
        )
        B = np.array([[0.0], [self._control_gain]], dtype=float)
        Q = np.diag(self.state_weights)
        R = np.array([[self.control_weight]], dtype=float)
        return solve_continuous_are(A, B, Q, R)

    def _bounded_control(self, u: float | np.ndarray) -> float | np.ndarray:
        if self.control_limit is None:
            return u
        return np.clip(u, -self.control_limit, self.control_limit)

    def fixed_target(self, fixed_target_sign: int) -> np.ndarray:
        if fixed_target_sign != 1:
            raise ValueError(
                "PendulumSwingUpProblem has one upright target; "
                "use fixed_target_sign=+1"
            )
        return np.zeros(2, dtype=float)

    def dynamics(self, _t: float, y: np.ndarray, u: float) -> np.ndarray:
        theta, omega = y
        u_eff = float(self._bounded_control(u))
        return np.array(
            [
                omega,
                -self._damping_gain * omega
                + self._gravity_gain * np.sin(theta)
                + self._control_gain * u_eff,
            ],
            dtype=float,
        )

    def adjoint_rhs(self, _t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        theta, omega = y
        p1, p2 = p
        q1, q2 = self.state_weights
        return np.array(
            [
                -2.0 * q1 * np.sin(theta) - self._gravity_gain * np.cos(theta) * p2,
                -2.0 * q2 * omega - p1 + self._damping_gain * p2,
            ],
            dtype=float,
        )

    def terminal_cost(self, y_terminal: np.ndarray, fixed_target_sign: int) -> float:
        if fixed_target_sign != 1:
            raise ValueError(
                "PendulumSwingUpProblem has one upright target; "
                "use fixed_target_sign=+1"
            )
        diff = np.asarray(y_terminal, dtype=float).copy()
        diff[0] = float(self.wrap_angle(diff[0]))
        return float(self.terminal_weight * diff @ self.local_lqr_matrix @ diff)

    def terminal_gradient(self, y_terminal: np.ndarray, fixed_target_sign: int) -> np.ndarray:
        if fixed_target_sign != 1:
            raise ValueError(
                "PendulumSwingUpProblem has one upright target; "
                "use fixed_target_sign=+1"
            )
        diff = np.asarray(y_terminal, dtype=float).copy()
        diff[0] = float(self.wrap_angle(diff[0]))
        return 2.0 * self.terminal_weight * self.local_lqr_matrix @ diff

    def running_cost(self, y_values: np.ndarray, u_values: np.ndarray) -> np.ndarray:
        theta = y_values[0, :]
        omega = y_values[1, :]
        u_eff = self._bounded_control(u_values)
        q1, q2 = self.state_weights
        return (
            q1 * (2.0 - 2.0 * np.cos(theta))
            + q2 * omega * omega
            + self.control_weight * u_eff * u_eff
        )

    def stationarity_residual(self, p_values: np.ndarray, u_values: np.ndarray) -> np.ndarray:
        if self.control_limit is not None:
            u_values = self._bounded_control(u_values)
        return self._control_gain * p_values[1, :] + 2.0 * self.control_weight * u_values

    def feedback_from_gradient(self, gradient: np.ndarray) -> float:
        gradient = np.asarray(gradient, dtype=float)
        u = -self._control_gain * gradient[1] / (2.0 * self.control_weight)
        return float(self._bounded_control(u))

    def transform_initial_state(self, initial_state: np.ndarray) -> np.ndarray:
        return -np.asarray(initial_state, dtype=float)

    def transform_gradient(self, gradient: np.ndarray) -> np.ndarray:
        return -np.asarray(gradient, dtype=float)

    def transform_control_coefficients(self, control_coefficients: np.ndarray) -> np.ndarray:
        return -np.asarray(control_coefficients, dtype=float)


def solve_pendulum_swingup_sample(
    initial_state: np.ndarray,
    fixed_target_sign: Optional[int] = None,
    **kwargs,
) -> ForwardBackwardOpenLoopResult:
    """Solve one pendulum swing-up sample with the forward-backward optimizer."""
    problem = kwargs.pop("problem", PendulumSwingUpProblem())
    optimizer = ForwardBackwardOpenLoopOptimizer(
        problem=problem,
        initial_state=initial_state,
        fixed_target_sign=fixed_target_sign,
        **kwargs,
    )
    return optimizer.optimize()


__all__ = ["PendulumSwingUpProblem", "solve_pendulum_swingup_sample"]
