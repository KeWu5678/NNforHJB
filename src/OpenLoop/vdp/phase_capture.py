"""Van der Pol phase-capture problem for finite-horizon open-loop solves."""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.OpenLoop.forward_backward_optimizer import (
    ForwardBackwardOpenLoopOptimizer,
    ForwardBackwardOpenLoopResult,
)


class VdpPhaseCaptureProblem:
    """Two-target Van der Pol phase-capture problem."""

    def __init__(
        self,
        beta: float = 0.1,
        mu: float = 1.0,
        state_weight: float = 1.0,
        terminal_weight: float = 1.0,
        terminal_kappa: float = 1.0,
        target: tuple[float, float] = (2.0, 0.0),
        T_final: float = 3.0,
    ) -> None:
        self.beta = float(beta)
        self.mu = float(mu)
        self.state_weight = float(state_weight)
        self.terminal_weight = float(terminal_weight)
        self.terminal_kappa = float(terminal_kappa)
        self.target = np.asarray(target, dtype=float)
        self.T_final = float(T_final)

        if self.target.shape != (2,):
            raise ValueError("target must be a two-dimensional vector")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if self.terminal_kappa <= 0:
            raise ValueError("terminal_kappa must be positive")
        self.fixed_target_signs = (-1, 1)

    def fixed_target(self, fixed_target_sign: int) -> np.ndarray:
        if fixed_target_sign not in (-1, 1):
            raise ValueError("fixed_target_sign must be either +1 or -1")
        return fixed_target_sign * self.target

    def dynamics(self, _t: float, y: np.ndarray, u: float) -> np.ndarray:
        y1, y2 = y
        return np.array(
            [
                y2,
                self.mu * (1.0 - y1**2) * y2 - y1 + u,
            ],
            dtype=float,
        )

    def adjoint_rhs(self, _t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        y1, y2 = y
        p1, p2 = p
        return np.array(
            [
                -2.0 * self.state_weight * y1
                + p2 * (1.0 + 2.0 * self.mu * y1 * y2),
                -2.0 * self.state_weight * y2
                - p1
                - self.mu * (1.0 - y1**2) * p2,
            ],
            dtype=float,
        )

    def terminal_cost(self, y_terminal: np.ndarray, fixed_target_sign: int) -> float:
        diff = np.asarray(y_terminal, dtype=float) - self.fixed_target(fixed_target_sign)
        r2 = float(diff @ diff)
        return float(self.terminal_weight * (1.0 - np.exp(-self.terminal_kappa * r2)))

    def terminal_gradient(self, y_terminal: np.ndarray, fixed_target_sign: int) -> np.ndarray:
        diff = np.asarray(y_terminal, dtype=float) - self.fixed_target(fixed_target_sign)
        r2 = float(diff @ diff)
        scale = (
            2.0
            * self.terminal_weight
            * self.terminal_kappa
            * np.exp(-self.terminal_kappa * r2)
        )
        return scale * diff

    def running_cost(self, y_values: np.ndarray, u_values: np.ndarray) -> np.ndarray:
        state_cost = self.state_weight * np.sum(y_values * y_values, axis=0)
        control_cost = self.beta * u_values * u_values
        return state_cost + control_cost

    def stationarity_residual(self, p_values: np.ndarray, u_values: np.ndarray) -> np.ndarray:
        return p_values[1, :] + 2.0 * self.beta * u_values

    def feedback_from_gradient(self, gradient: np.ndarray) -> float:
        gradient = np.asarray(gradient, dtype=float)
        return float(-gradient[1] / (2.0 * self.beta))

    def transform_initial_state(self, initial_state: np.ndarray) -> np.ndarray:
        return -np.asarray(initial_state, dtype=float)

    def transform_gradient(self, gradient: np.ndarray) -> np.ndarray:
        return -np.asarray(gradient, dtype=float)

    def transform_control_coefficients(self, control_coefficients: np.ndarray) -> np.ndarray:
        return -np.asarray(control_coefficients, dtype=float)


def solve_phase_capture_sample(
    initial_state: np.ndarray,
    fixed_target_sign: Optional[int] = None,
    **kwargs,
) -> ForwardBackwardOpenLoopResult:
    """Solve one Van der Pol phase-capture sample."""
    problem = kwargs.pop("problem", VdpPhaseCaptureProblem())
    optimizer = ForwardBackwardOpenLoopOptimizer(
        problem=problem,
        initial_state=initial_state,
        fixed_target_sign=fixed_target_sign,
        **kwargs,
    )
    return optimizer.optimize()


__all__ = [
    "VdpPhaseCaptureProblem",
    "solve_phase_capture_sample",
]
