"""Van der Pol finite-horizon optimal-control problem."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VdpOptimalControlProblem:
    """Smooth VDP benchmark used to generate PDAP value samples."""

    beta: float = 0.1
    mu: float = 1.0
    state_weight: float = 1.0
    T_final: float = 3.0

    def __post_init__(self) -> None:
        if self.beta <= 0.0:
            raise ValueError("beta must be positive")
        if self.mu <= 0.0:
            raise ValueError("mu must be positive")
        if self.state_weight <= 0.0:
            raise ValueError("state_weight must be positive")
        if self.T_final <= 0.0:
            raise ValueError("T_final must be positive")

    def dynamics(self, _t: float, y: np.ndarray, u: float) -> np.ndarray:
        y1, y2 = np.asarray(y, dtype=np.float64)
        return np.array(
            [
                y2,
                -y1 + self.mu * (1.0 - y1**2) * y2 + u,
            ],
            dtype=np.float64,
        )

    def adjoint_rhs(self, _t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        y1, y2 = np.asarray(y, dtype=np.float64)
        p1, p2 = np.asarray(p, dtype=np.float64)
        return np.array(
            [
                -2.0 * self.state_weight * y1
                + p2 * (1.0 + 2.0 * self.mu * y1 * y2),
                -2.0 * self.state_weight * y2
                - p1
                - self.mu * (1.0 - y1**2) * p2,
            ],
            dtype=np.float64,
        )

    def terminal_gradient(self, _y_terminal: np.ndarray) -> np.ndarray:
        return np.zeros(2, dtype=np.float64)

    def running_cost(self, y_values: np.ndarray, u_values: np.ndarray) -> np.ndarray:
        y_values = np.asarray(y_values, dtype=np.float64)
        u_values = np.asarray(u_values, dtype=np.float64)
        state_cost = self.state_weight * np.sum(y_values * y_values, axis=0)
        control_cost = self.beta * u_values * u_values
        return state_cost + control_cost

    def reduced_gradient(self, p_values: np.ndarray, u_values: np.ndarray) -> np.ndarray:
        p_values = np.asarray(p_values, dtype=np.float64)
        u_values = np.asarray(u_values, dtype=np.float64)
        return p_values[1, :] + 2.0 * self.beta * u_values

    def feedback_from_gradient(self, gradient: np.ndarray) -> float:
        gradient = np.asarray(gradient, dtype=np.float64)
        return float(-gradient[1] / (2.0 * self.beta))


__all__ = ["VdpOptimalControlProblem"]
