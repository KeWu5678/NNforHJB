"""Forward-backward reduced-gradient optimizer for open-loop control data.

This optimizer treats the control coefficients as the finite-dimensional
variables and evaluates each objective/gradient by:

1. integrating the state forward,
2. integrating the adjoint backward,
3. projecting the stationarity residual onto the Legendre basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.polynomial.legendre import legval, legvander
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


@dataclass
class ForwardBackwardOpenLoopResult:
    """One optimized fixed-initial-state control sample."""

    initial_state: np.ndarray
    fixed_target_sign: int
    value: float
    gradient: np.ndarray
    control_coefficients: np.ndarray
    coefficient_gradient: np.ndarray
    coefficient_gradient_norm: float
    iterations: int
    converged: bool
    message: str


class ForwardBackwardOpenLoopOptimizer:
    """Finite-dimensional open-loop optimizer using forward-backward IVP solves."""

    def __init__(
        self,
        problem: object,
        initial_state: np.ndarray,
        fixed_target_sign: Optional[int] = None,
        num_basis: int = 30,
        time_grid: Optional[np.ndarray] = None,
        initial_coefficients: Optional[np.ndarray] = None,
        optimizer_method: str = "L-BFGS-B",
        max_iter: int = 500,
        coefficient_gradient_tol: float = 1e-6,
        optimizer_ftol: float = 1e-12,
        max_line_search: int = 50,
        ivp_rtol: float = 1e-7,
        ivp_atol: float = 1e-9,
    ) -> None:
        self.problem = problem
        self.initial_state = np.asarray(initial_state, dtype=float)
        self.fixed_target_sign = fixed_target_sign
        self.num_basis = int(num_basis)
        self.optimizer_method = optimizer_method
        self.max_iter = int(max_iter)
        self.coefficient_gradient_tol = float(coefficient_gradient_tol)
        self.optimizer_ftol = float(optimizer_ftol)
        self.max_line_search = int(max_line_search)
        self.ivp_rtol = float(ivp_rtol)
        self.ivp_atol = float(ivp_atol)

        if self.initial_state.shape != (2,):
            raise ValueError("initial_state must be a two-dimensional vector")
        allowed_signs = tuple(getattr(self.problem, "fixed_target_signs", (-1, 1)))
        if fixed_target_sign is not None and fixed_target_sign not in allowed_signs:
            raise ValueError(
                "fixed_target_sign must be one of "
                f"{allowed_signs}, or None"
            )
        if self.num_basis <= 0:
            raise ValueError("num_basis must be positive")

        if time_grid is None:
            time_grid = np.linspace(0.0, self.problem.T_final, 301)
        self.time_grid = np.asarray(time_grid, dtype=float)
        self.domain = (float(self.time_grid[0]), float(self.time_grid[-1]))

        if initial_coefficients is None:
            initial_coefficients = np.zeros(self.num_basis, dtype=float)
        self.initial_coefficients = np.asarray(initial_coefficients, dtype=float)
        if self.initial_coefficients.shape != (self.num_basis,):
            raise ValueError("initial_coefficients must have shape (num_basis,)")

        self._basis_matrix = self._basis(self.time_grid)
        self.last_result: Optional[ForwardBackwardOpenLoopResult] = None

    def _map_to_legendre_domain(self, t: np.ndarray) -> np.ndarray:
        a, b = self.domain
        return (2.0 * t - (a + b)) / (b - a)

    def _basis(self, t: np.ndarray) -> np.ndarray:
        mapped = self._map_to_legendre_domain(np.asarray(t, dtype=float))
        return legvander(mapped, self.num_basis - 1)

    def control_values(self, coefficients: np.ndarray, t: np.ndarray) -> np.ndarray:
        mapped = self._map_to_legendre_domain(np.asarray(t, dtype=float))
        return legval(mapped, coefficients)

    def _integrate_state(self, coefficients: np.ndarray):
        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            u = float(self.control_values(coefficients, np.array([t]))[0])
            return self.problem.dynamics(t, y, u)

        return solve_ivp(
            rhs,
            self.domain,
            self.initial_state,
            t_eval=self.time_grid,
            dense_output=True,
            rtol=self.ivp_rtol,
            atol=self.ivp_atol,
        )

    def _integrate_adjoint(self, state_solution, fixed_target_sign: int):
        terminal_state = state_solution.y[:, -1]
        terminal_adjoint = self.problem.terminal_gradient(
            terminal_state,
            fixed_target_sign,
        )

        def rhs(t: float, p: np.ndarray) -> np.ndarray:
            y = state_solution.sol(t)
            return self.problem.adjoint_rhs(t, y, p)

        return solve_ivp(
            rhs,
            (self.domain[1], self.domain[0]),
            terminal_adjoint,
            t_eval=self.time_grid[::-1],
            dense_output=False,
            rtol=self.ivp_rtol,
            atol=self.ivp_atol,
        )

    def evaluate_fixed_target(
        self,
        coefficients: np.ndarray,
        fixed_target_sign: int,
    ) -> tuple[float, np.ndarray, dict]:
        coefficients = np.asarray(coefficients, dtype=float)
        state_sol = self._integrate_state(coefficients)
        if not state_sol.success:
            raise RuntimeError(f"state integration failed: {state_sol.message}")

        adjoint_sol = self._integrate_adjoint(state_sol, fixed_target_sign)
        if not adjoint_sol.success:
            raise RuntimeError(f"adjoint integration failed: {adjoint_sol.message}")

        y_values = state_sol.y
        p_values = adjoint_sol.y[:, ::-1]
        u_values = self.control_values(coefficients, self.time_grid)

        running_cost = self.problem.running_cost(y_values, u_values)
        value = float(np.trapezoid(running_cost, self.time_grid))
        value += self.problem.terminal_cost(y_values[:, -1], fixed_target_sign)

        residual = self.problem.stationarity_residual(p_values, u_values)
        coefficient_gradient = np.array(
            [
                np.trapezoid(residual * self._basis_matrix[:, i], self.time_grid)
                for i in range(self.num_basis)
            ],
            dtype=float,
        )

        info = {
            "state": y_values,
            "adjoint": p_values,
            "control": u_values,
            "residual": residual,
        }
        return value, coefficient_gradient, info

    def optimize_fixed_target(self, fixed_target_sign: int) -> ForwardBackwardOpenLoopResult:
        allowed_signs = tuple(getattr(self.problem, "fixed_target_signs", (-1, 1)))
        if fixed_target_sign not in allowed_signs:
            raise ValueError(f"fixed_target_sign must be one of {allowed_signs}")

        # Cache the last evaluation so fun and jac share one forward+adjoint solve.
        _cache: dict = {}

        def _eval(coefficients: np.ndarray):
            key = coefficients.tobytes()
            if key not in _cache:
                try:
                    v, g, info = self.evaluate_fixed_target(coefficients, fixed_target_sign)
                except RuntimeError:
                    v, g, info = 1e100, np.zeros_like(coefficients), {}
                _cache.clear()
                _cache[key] = (coefficients.copy(), v, g, info)
            return _cache[key]

        def fun(coefficients: np.ndarray) -> float:
            return _eval(coefficients)[1]

        def jac(coefficients: np.ndarray) -> np.ndarray:
            return _eval(coefficients)[2]

        options = {
            "maxiter": self.max_iter,
            "gtol": self.coefficient_gradient_tol,
        }
        if self.optimizer_method.upper() == "L-BFGS-B":
            options.update(
                {
                    "ftol": self.optimizer_ftol,
                    "maxls": self.max_line_search,
                }
            )

        opt = minimize(
            fun,
            self.initial_coefficients,
            jac=jac,
            method=self.optimizer_method,
            options=options,
        )

        _, value, coefficient_gradient, info = _eval(opt.x)
        coefficient_gradient_norm = float(np.linalg.norm(coefficient_gradient))
        converged = bool(opt.success) and (
            coefficient_gradient_norm <= self.coefficient_gradient_tol
        )
        message = str(opt.message)
        if not converged:
            message = (
                f"{message}; coefficient_gradient_norm="
                f"{coefficient_gradient_norm:.3e} > "
                f"{self.coefficient_gradient_tol:.3e}"
            )

        return ForwardBackwardOpenLoopResult(
            initial_state=self.initial_state.copy(),
            fixed_target_sign=fixed_target_sign,
            value=float(value),
            gradient=info["adjoint"][:, 0].copy(),
            control_coefficients=opt.x.copy(),
            coefficient_gradient=coefficient_gradient.copy(),
            coefficient_gradient_norm=coefficient_gradient_norm,
            iterations=int(opt.nit),
            converged=converged,
            message=message,
        )

    def optimize(self, return_all_fixed_targets: bool = False):
        default_signs = getattr(self.problem, "fixed_target_signs", (-1, 1))
        signs = (
            tuple(default_signs)
            if self.fixed_target_sign is None
            else (self.fixed_target_sign,)
        )
        results = [self.optimize_fixed_target(sign) for sign in signs]
        best = min(results, key=lambda result: result.value)
        self.last_result = best
        if return_all_fixed_targets:
            return best, results
        return best

    @staticmethod
    def transform_positive_result_to_negative(result: ForwardBackwardOpenLoopResult):
        """Map a positive fixed-target solve at ``-x`` to a negative solve at ``x``."""
        if result.fixed_target_sign != 1:
            raise ValueError("result must be for fixed_target_sign=+1")
        return ForwardBackwardOpenLoopResult(
            initial_state=-result.initial_state,
            fixed_target_sign=-1,
            value=float(result.value),
            gradient=-result.gradient,
            control_coefficients=-result.control_coefficients,
            coefficient_gradient=-result.coefficient_gradient,
            coefficient_gradient_norm=float(result.coefficient_gradient_norm),
            iterations=int(result.iterations),
            converged=bool(result.converged),
            message=f"symmetry transform: {result.message}",
        )
