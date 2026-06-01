"""Pendulum swing-up problem for forward-backward finite-horizon solves."""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.OpenLoop.forward_backward_optimizer import (
    ForwardBackwardOpenLoopOptimizer,
    ForwardBackwardOpenLoopResult,
)
from src.OpenLoop.pendulum.swingup_dynamics import PendulumSwingUpDynamics


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
        self.dynamics_model = PendulumSwingUpDynamics(
            mass=mass,
            length=length,
            damping=damping,
            gravity=gravity,
            state_weights=state_weights,
            control_weight=control_weight,
            control_limit=control_limit,
        )
        self.mass = self.dynamics_model.mass
        self.length = self.dynamics_model.length
        self.damping = self.dynamics_model.damping
        self.gravity = self.dynamics_model.gravity
        self.state_weights = self.dynamics_model.state_weights
        self.control_weight = float(control_weight)
        self.terminal_weight = float(terminal_weight)
        self.T_final = float(T_final)
        self.control_limit = self.dynamics_model.control_limit
        self.fixed_target_signs = (1,)

        if self.terminal_weight < 0:
            raise ValueError("terminal_weight must be nonnegative")

        self._control_gain = self.dynamics_model.control_gain
        self._damping_gain = self.dynamics_model.damping_gain
        self._gravity_gain = self.dynamics_model.gravity_gain
        self.local_lqr_matrix = self.dynamics_model.local_lqr_matrix

    @staticmethod
    def wrap_angle(theta: np.ndarray | float) -> np.ndarray | float:
        """Map angles to [-pi, pi) for the local terminal LQR penalty."""
        return PendulumSwingUpDynamics.wrap_angle(theta)

    def fixed_target(self, fixed_target_sign: int) -> np.ndarray:
        if fixed_target_sign != 1:
            raise ValueError(
                "PendulumSwingUpProblem has one upright target; "
                "use fixed_target_sign=+1"
            )
        return np.zeros(2, dtype=float)

    def dynamics(self, _t: float, y: np.ndarray, u: float) -> np.ndarray:
        return self.dynamics_model.forward_dynamics(y, u)

    def adjoint_rhs(self, _t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return self.dynamics_model.costate_rhs(y, p)

    def terminal_cost(self, y_terminal: np.ndarray, fixed_target_sign: int) -> float:
        if fixed_target_sign != 1:
            raise ValueError(
                "PendulumSwingUpProblem has one upright target; "
                "use fixed_target_sign=+1"
            )
        return self.dynamics_model.local_lqr_value(
            y_terminal,
            weight=self.terminal_weight,
        )

    def terminal_gradient(self, y_terminal: np.ndarray, fixed_target_sign: int) -> np.ndarray:
        if fixed_target_sign != 1:
            raise ValueError(
                "PendulumSwingUpProblem has one upright target; "
                "use fixed_target_sign=+1"
            )
        return self.dynamics_model.local_lqr_gradient(
            y_terminal,
            weight=self.terminal_weight,
        )

    def running_cost(self, y_values: np.ndarray, u_values: np.ndarray) -> np.ndarray:
        return self.dynamics_model.running_cost(y_values.T, u_values)

    def stationarity_residual(self, p_values: np.ndarray, u_values: np.ndarray) -> np.ndarray:
        return self.dynamics_model.stationarity_residual(p_values.T, u_values)

    def feedback_from_gradient(self, gradient: np.ndarray) -> float:
        return float(self.dynamics_model.minimizing_control(gradient))

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
