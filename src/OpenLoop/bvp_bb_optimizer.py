"""Barzilai-Borwein open-loop optimizer around a boundary-value solve.

The optimizer treats the control as Legendre coefficients.  Each trial control
is evaluated by solving the state-costate BVP, computing the stationarity
residual, and fitting that residual back into the same coefficient basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.integrate import solve_bvp

from src import utils


ControlFunction = Callable[[np.ndarray], np.ndarray]


@dataclass
class BvpBbOpenLoopResult:
    """Result of one open-loop optimization run."""

    gradient: np.ndarray | None
    value: float | None
    converged: bool
    message: str
    iterations: int
    gradient_norm: float | None
    control_coefficients: np.ndarray | None
    gradient_values: np.ndarray | None
    gradient_coefficients: np.ndarray | None
    solution: Any | None = None

    def as_legacy_tuple(self) -> tuple[np.ndarray | None, float | None]:
        """Return the historical ``(gradient, value)`` API."""
        if not self.converged:
            return None, None
        return self.gradient, self.value


@dataclass
class _ControlEvaluation:
    control: ControlFunction
    control_coefficients: np.ndarray
    solution: Any
    gradient_values: np.ndarray
    gradient_coefficients: np.ndarray
    gradient_norm: float
    value: float


class BvpBbOpenLoopOptimizer:
    """Optimize an open-loop control represented by Legendre coefficients."""

    def __init__(
        self,
        ODE,
        bc,
        V,
        gradient,
        grid,
        guess,
        tol,
        max_it,
        num_basis=50,
        domain=None,
        bvp_tol=1e-7,
        max_nodes=10000,
        initial_coefficients=None,
        initial_step_scale=0.1,
        alpha_min=1e-5,
        alpha_max=10.0,
        alpha_default=0.1,
        line_search_beta=0.5,
        line_search_max_iter=500,
        line_search_cost_tol=1e-2,
        verbose=True,
    ):
        self.dynamics = ODE
        self.boundary_conditions = bc
        self.objective = V
        self.gradient = gradient
        self.grid = np.asarray(grid, dtype=float)
        self.guess = np.asarray(guess, dtype=float)
        self.tol = float(tol)
        self.max_it = int(max_it)
        self.num_basis = int(num_basis)
        self.domain = (
            (float(self.grid[0]), float(self.grid[-1]))
            if domain is None
            else (float(domain[0]), float(domain[1]))
        )
        self.bvp_tol = float(bvp_tol)
        self.max_nodes = int(max_nodes)
        self.initial_coefficients = (
            None
            if initial_coefficients is None
            else np.asarray(initial_coefficients, dtype=float)
        )
        self.initial_step_scale = float(initial_step_scale)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.alpha_default = float(alpha_default)
        self.line_search_beta = float(line_search_beta)
        self.line_search_max_iter = int(line_search_max_iter)
        self.line_search_cost_tol = float(line_search_cost_tol)
        self.verbose = bool(verbose)

        # Backward-compatible attribute names used by older problem code.
        self.ODE = self.dynamics
        self.bc = self.boundary_conditions
        self.V = self.objective

        self.last_solution = None
        self.last_control_coefficients = None
        self.last_gradient_values = None
        self.last_gradient_coefficients = None
        self.last_gradient_norm = None
        self.last_iterations = 0
        self.last_converged = False
        self.last_message = ""
        self.last_result: BvpBbOpenLoopResult | None = None

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def initial_control_coefficients(self) -> np.ndarray:
        if self.initial_coefficients is None:
            return np.zeros(self.num_basis)
        if self.initial_coefficients.shape != (self.num_basis,):
            raise ValueError("initial_coefficients must have shape (num_basis,)")
        return self.initial_coefficients.copy()

    def solve_bvp_for_control(self, control: ControlFunction, guess=None):
        guess_values = self.guess if guess is None else guess
        return solve_bvp(
            lambda t, y: self.dynamics(t, y, control),
            self.boundary_conditions,
            self.grid,
            guess_values,
            tol=self.bvp_tol,
            max_nodes=self.max_nodes,
        )

    @staticmethod
    def barzilai_borwein_step(
        previous_coefficients: np.ndarray,
        current_coefficients: np.ndarray,
        previous_gradient_coefficients: np.ndarray,
        current_gradient_coefficients: np.ndarray,
    ) -> tuple[float, float, float]:
        step_delta = current_coefficients - previous_coefficients
        gradient_delta = current_gradient_coefficients - previous_gradient_coefficients

        curvature = float(np.dot(step_delta, gradient_delta))
        step_norm_squared = float(np.dot(step_delta, step_delta))
        gradient_norm_squared = float(np.dot(gradient_delta, gradient_delta))

        epsilon = 1e-20
        bb1 = step_norm_squared / (curvature + epsilon)
        bb2 = curvature / (gradient_norm_squared + epsilon)
        return float(bb1), float(bb2), curvature

    def evaluate_control(
        self,
        control_coefficients: np.ndarray,
        guess=None,
    ) -> _ControlEvaluation:
        control_coefficients = np.asarray(control_coefficients, dtype=float)
        control = utils.gen_legendre(control_coefficients, domain=self.domain)
        solution = self.solve_bvp_for_control(control, guess)
        if not solution.success:
            raise RuntimeError(str(solution.message))

        control_values = control(solution.x)
        gradient_values = self.gradient(control_values, solution.y[3, :])
        gradient_coefficients = utils.fit_legendre(
            solution.x,
            gradient_values,
            self.num_basis,
            domain=self.domain,
        )
        value = self.objective(
            solution.x,
            control_values,
            solution.y[0, :],
            solution.y[1, :],
        )
        gradient_norm = utils.L2(solution.x, gradient_values)

        return _ControlEvaluation(
            control=control,
            control_coefficients=control_coefficients.copy(),
            solution=solution,
            gradient_values=gradient_values,
            gradient_coefficients=gradient_coefficients,
            gradient_norm=float(gradient_norm),
            value=float(value),
        )

    def select_step_size(
        self,
        iteration: int,
        previous: _ControlEvaluation,
        current: _ControlEvaluation,
    ) -> float:
        bb1, bb2, curvature = self.barzilai_borwein_step(
            previous.control_coefficients,
            current.control_coefficients,
            previous.gradient_coefficients,
            current.gradient_coefficients,
        )
        trial_step = bb1 if iteration % 2 == 0 else bb2
        if (
            curvature <= 0
            or trial_step <= 0
            or not np.isfinite(trial_step)
        ):
            return self.alpha_default
        return float(np.clip(trial_step, self.alpha_min, self.alpha_max))

    def line_search(
        self,
        current: _ControlEvaluation,
        step_size: float,
    ) -> tuple[_ControlEvaluation | None, str]:
        descent_direction = -current.gradient_coefficients
        accepted_step = float(step_size)
        last_failure = ""

        for _line_search_iter in range(self.line_search_max_iter):
            trial_coefficients = (
                current.control_coefficients + accepted_step * descent_direction
            )
            try:
                trial = self.evaluate_control(
                    trial_coefficients,
                    current.solution.sol(self.grid),
                )
            except RuntimeError as exc:
                last_failure = str(exc)
            else:
                allowed_value = current.value * (1.0 + self.line_search_cost_tol)
                if np.isfinite(trial.value) and trial.value <= allowed_value:
                    return trial, ""
                last_failure = (
                    f"trial cost {trial.value:.6e} exceeded "
                    f"allowed cost {allowed_value:.6e}"
                )

            accepted_step *= self.line_search_beta
            if accepted_step < self.alpha_min:
                break

        if last_failure:
            return None, f"line search failed: {last_failure}"
        return None, "line search failed to find a decreasing-cost step"

    def _failure_result(
        self,
        message: str,
        iterations: int = 0,
        evaluation: _ControlEvaluation | None = None,
    ) -> BvpBbOpenLoopResult:
        gradient = None
        if evaluation is not None:
            gradient = np.array(
                [
                    evaluation.solution.y[2, 0],
                    evaluation.solution.y[3, 0],
                ],
                dtype=float,
            )
        result = BvpBbOpenLoopResult(
            gradient=gradient,
            value=None if evaluation is None else float(evaluation.value),
            converged=False,
            message=message,
            iterations=iterations,
            gradient_norm=None if evaluation is None else float(evaluation.gradient_norm),
            control_coefficients=(
                None if evaluation is None else evaluation.control_coefficients.copy()
            ),
            gradient_values=None if evaluation is None else evaluation.gradient_values.copy(),
            gradient_coefficients=(
                None if evaluation is None else evaluation.gradient_coefficients.copy()
            ),
            solution=None if evaluation is None else evaluation.solution,
        )
        self._store_result(result)
        return result

    def _success_result(
        self,
        evaluation: _ControlEvaluation,
        iterations: int,
    ) -> BvpBbOpenLoopResult:
        gradient = np.array(
            [
                evaluation.solution.y[2, 0],
                evaluation.solution.y[3, 0],
            ],
            dtype=float,
        )
        return BvpBbOpenLoopResult(
            gradient=gradient,
            value=float(evaluation.value),
            converged=True,
            message="converged",
            iterations=int(iterations),
            gradient_norm=float(evaluation.gradient_norm),
            control_coefficients=evaluation.control_coefficients.copy(),
            gradient_values=evaluation.gradient_values.copy(),
            gradient_coefficients=evaluation.gradient_coefficients.copy(),
            solution=evaluation.solution,
        )

    def _store_result(self, result: BvpBbOpenLoopResult) -> None:
        self.last_result = result
        self.last_solution = result.solution
        self.last_control_coefficients = (
            None
            if result.control_coefficients is None
            else result.control_coefficients.copy()
        )
        self.last_gradient_values = (
            None if result.gradient_values is None else result.gradient_values.copy()
        )
        self.last_gradient_coefficients = (
            None
            if result.gradient_coefficients is None
            else result.gradient_coefficients.copy()
        )
        self.last_gradient_norm = result.gradient_norm
        self.last_iterations = int(result.iterations)
        self.last_converged = bool(result.converged)
        self.last_message = result.message

    def optimize_result(self) -> BvpBbOpenLoopResult:
        """Run the optimization and return a structured result."""
        initial_coefficients = self.initial_control_coefficients()

        self._log("Solving initial BVP...")
        try:
            previous = self.evaluate_control(initial_coefficients)
        except RuntimeError as exc:
            message = f"initial BVP failed: {exc}"
            self._log(message)
            return self._failure_result(message)

        current_coefficients = (
            previous.control_coefficients
            - self.initial_step_scale * previous.gradient_coefficients
        )
        try:
            current = self.evaluate_control(
                current_coefficients,
                previous.solution.sol(self.grid),
            )
        except RuntimeError as exc:
            message = f"first BB BVP failed: {exc}"
            self._log(message)
            return self._failure_result(message)

        iteration = 1
        while current.gradient_norm >= self.tol:
            step_size = self.select_step_size(iteration, previous, current)
            trial, failure_message = self.line_search(current, step_size)
            if trial is None:
                self._log(f"Warning: {failure_message}")
                result = self._failure_result(failure_message, iteration, current)
                return result

            previous = current
            current = trial
            iteration += 1

            self._log(
                " k = "
                f"{iteration}, cost = {current.value}, "
                f"norm G = {current.gradient_norm}"
            )

            if not np.isfinite(current.gradient_norm) or not np.isfinite(current.value):
                message = "numerical instability detected"
                self._log(f"Warning: {message}")
                return self._failure_result(message, iteration, current)

            if current.gradient_norm < self.tol:
                break

            if iteration >= self.max_it:
                message = "maximum iterations reached"
                self._log(message)
                self._log(f" norm G = {current.gradient_norm}")
                return self._failure_result(message, iteration, current)

        result = self._success_result(current, iteration)
        self._store_result(result)
        self._log(f" cost = {result.value}")
        self._log(f" norm G = {result.gradient_norm}")
        return result

    def optimize(self):
        """Run optimization and return the historical ``(gradient, value)`` tuple."""
        return self.optimize_result().as_legacy_tuple()


__all__ = ["BvpBbOpenLoopOptimizer", "BvpBbOpenLoopResult"]
