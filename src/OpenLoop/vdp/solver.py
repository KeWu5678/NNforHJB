"""Clean VDP open-loop solver for smooth PDAP training data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.polynomial.legendre import legval, legvander
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, root

from src.OpenLoop.value_samples import ValueSamples
from src.OpenLoop.vdp.problem import VdpOptimalControlProblem
from src.paths import DATA_DIR


def _trapz_norm(time_grid: np.ndarray, values: np.ndarray) -> float:
    return float(np.sqrt(np.trapezoid(np.asarray(values, dtype=np.float64) ** 2, time_grid)))


def _trapz_dot(time_grid: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    return float(np.trapezoid(np.asarray(left) * np.asarray(right), time_grid))


@dataclass(frozen=True)
class VdpOpenLoopSolverConfig:
    """Numerical configuration for VDP value-sample generation."""

    profile: str = "paper"
    time_step: float | None = None
    num_time_points: int | None = None
    num_control_basis: int = 30
    convergence_tol: float = 1e-5
    max_iter: int = 500
    ivp_rtol: float = 1e-7
    ivp_atol: float = 1e-9
    cn_root_tol: float = 1e-10
    initial_step_scale: float = 0.1
    alpha_min: float = 1e-8
    alpha_max: float = 1e2
    alpha_default: float = 1e-1
    line_search_beta: float = 0.5
    line_search_max_iter: int = 50
    line_search_cost_tol: float = 1e-2
    store_trajectories: bool = False

    def __post_init__(self) -> None:
        if self.profile not in ("paper", "fast"):
            raise ValueError("profile must be 'paper' or 'fast'")
        if self.time_step is not None and self.time_step <= 0.0:
            raise ValueError("time_step must be positive")
        if self.num_time_points is not None and self.num_time_points < 2:
            raise ValueError("num_time_points must be at least 2")
        if self.num_control_basis <= 0:
            raise ValueError("num_control_basis must be positive")
        if self.convergence_tol <= 0.0:
            raise ValueError("convergence_tol must be positive")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.ivp_rtol <= 0.0 or self.ivp_atol <= 0.0:
            raise ValueError("ivp tolerances must be positive")
        if self.cn_root_tol <= 0.0:
            raise ValueError("cn_root_tol must be positive")


@dataclass(frozen=True)
class VdpSampleResult:
    initial_state: np.ndarray
    converged: bool
    message: str
    value: float | None
    gradient: np.ndarray | None
    reduced_gradient_norm: float | None
    coefficient_gradient_norm: float | None
    iterations: int
    control: np.ndarray | None = None
    state_trajectory: np.ndarray | None = None
    adjoint_trajectory: np.ndarray | None = None
    reduced_gradient: np.ndarray | None = None


@dataclass(frozen=True)
class VdpOpenLoopSolution:
    value_samples: ValueSamples
    sample_results: tuple[VdpSampleResult, ...]
    failed_initial_states: np.ndarray
    config: VdpOpenLoopSolverConfig

    def save_dataset(
        self,
        output_dir: str | Path = DATA_DIR,
        *,
        grid_shape: tuple[int, int] | None = None,
        date_tag: str | None = None,
    ) -> dict[str, Path]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        date = datetime.now().strftime("%Y%m%d") if date_tag is None else date_tag
        shape_tag = (
            f"grid_{grid_shape[0]}x{grid_shape[1]}"
            if grid_shape is not None
            else f"N{self.value_samples.size}"
        )
        stem = f"VDP_{self.config.profile}_{shape_tag}_{date}"
        data_path = self.value_samples.save_npz(output_path / f"{stem}.npz")
        meta_path = output_path / f"VDP_{self.config.profile}_meta_{shape_tag}_{date}.json"
        failed_path = output_path / f"VDP_{self.config.profile}_failed_{shape_tag}_{date}.json"

        meta = {
            "profile": self.config.profile,
            "retained_samples": self.value_samples.size,
            "requested_samples": len(self.sample_results),
            "failed_samples": int(self.failed_initial_states.shape[0]),
            "convergence_tol": self.config.convergence_tol,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        failed_payload = [
            {
                "initial_state": result.initial_state.tolist(),
                "message": result.message,
                "iterations": result.iterations,
                "reduced_gradient_norm": result.reduced_gradient_norm,
            }
            for result in self.sample_results
            if not result.converged
        ]
        failed_path.write_text(json.dumps(failed_payload, indent=2), encoding="utf-8")
        return {"data": data_path, "meta": meta_path, "failed": failed_path}


@dataclass(frozen=True)
class _Evaluation:
    value: float
    gradient: np.ndarray
    reduced_gradient_norm: float
    coefficient_gradient: np.ndarray | None
    coefficient_gradient_norm: float | None
    state: np.ndarray
    adjoint: np.ndarray
    control: np.ndarray


class VdpOpenLoopSolver:
    """Generate VDP `(x, V(x), dV(x))` samples for PDAP."""

    def __init__(
        self,
        problem: VdpOptimalControlProblem | None = None,
        config: VdpOpenLoopSolverConfig | None = None,
    ) -> None:
        self.problem = problem or VdpOptimalControlProblem()
        self.config = config or VdpOpenLoopSolverConfig()
        self.time_grid = self._build_time_grid()
        self.domain = (float(self.time_grid[0]), float(self.time_grid[-1]))
        self._basis_matrix = self._basis(self.time_grid)

    def solve(
        self,
        initial_states: np.ndarray,
        progress: Callable[[int, int], None] | None = None,
    ) -> VdpOpenLoopSolution:
        states = np.asarray(initial_states, dtype=np.float64)
        if states.ndim != 2 or states.shape[1] != 2:
            raise ValueError("initial_states must have shape (n, 2)")

        results: list[VdpSampleResult] = []
        rows_x: list[np.ndarray] = []
        rows_v: list[float] = []
        rows_dv: list[np.ndarray] = []
        failed: list[np.ndarray] = []
        for idx, initial_state in enumerate(states):
            if progress is not None:
                progress(idx, states.shape[0])
            result = self.solve_sample(initial_state)
            results.append(result)
            if result.converged and result.value is not None and result.gradient is not None:
                rows_x.append(initial_state.copy())
                rows_v.append(float(result.value))
                rows_dv.append(result.gradient.copy())
            else:
                failed.append(initial_state.copy())
        if progress is not None:
            progress(states.shape[0], states.shape[0])

        samples = ValueSamples(
            x=np.asarray(rows_x, dtype=np.float64),
            v=np.asarray(rows_v, dtype=np.float64),
            dv=np.asarray(rows_dv, dtype=np.float64),
        )
        failed_states = (
            np.asarray(failed, dtype=np.float64)
            if failed
            else np.empty((0, 2), dtype=np.float64)
        )
        return VdpOpenLoopSolution(
            value_samples=samples,
            sample_results=tuple(results),
            failed_initial_states=failed_states,
            config=self.config,
        )

    def solve_sample(self, initial_state: np.ndarray) -> VdpSampleResult:
        initial = np.asarray(initial_state, dtype=np.float64)
        if initial.shape != (2,):
            raise ValueError("initial_state must have shape (2,)")
        if self.config.profile == "paper":
            return self._solve_sample_paper(initial)
        return self._solve_sample_fast(initial)

    def _solve_sample_paper(self, initial_state: np.ndarray) -> VdpSampleResult:
        control = np.zeros(self.time_grid.shape[0], dtype=np.float64)
        try:
            current = self._evaluate_time_grid_control(initial_state, control)
        except RuntimeError as exc:
            return self._failure_result(initial_state, str(exc), 0)
        if current.reduced_gradient_norm <= self.config.convergence_tol:
            return self._success_result(initial_state, current, 0)

        trial_control = control - self.config.initial_step_scale * current.gradient
        try:
            trial = self._evaluate_time_grid_control(initial_state, trial_control)
        except RuntimeError as exc:
            return self._failure_result(initial_state, f"first BB step failed: {exc}", 1, current)

        previous_control = control
        previous_gradient = current.gradient
        current_control = trial_control
        current_eval = trial

        for iteration in range(1, self.config.max_iter + 1):
            if current_eval.reduced_gradient_norm <= self.config.convergence_tol:
                return self._success_result(initial_state, current_eval, iteration)

            step_size = self._bb_step(
                previous_control,
                current_control,
                previous_gradient,
                current_eval.gradient,
                iteration,
            )
            accepted_control, accepted_eval, failure = self._line_search_time_grid(
                initial_state,
                current_control,
                current_eval,
                step_size,
            )
            if accepted_eval is None:
                return self._failure_result(initial_state, failure, iteration, current_eval)

            previous_control = current_control
            previous_gradient = current_eval.gradient
            current_control = accepted_control
            current_eval = accepted_eval

        return self._failure_result(
            initial_state,
            "maximum iterations reached",
            self.config.max_iter,
            current_eval,
        )

    def _solve_sample_fast(self, initial_state: np.ndarray) -> VdpSampleResult:
        cache: dict[bytes, tuple[np.ndarray, float, np.ndarray, _Evaluation]] = {}

        def evaluate(coefficients: np.ndarray):
            coefficients = np.asarray(coefficients, dtype=np.float64)
            key = coefficients.tobytes()
            if key not in cache:
                try:
                    evaluation = self._evaluate_legendre_control(initial_state, coefficients)
                    gradient = evaluation.coefficient_gradient
                    if gradient is None:
                        raise RuntimeError("missing coefficient gradient")
                    value = evaluation.value
                except RuntimeError:
                    gradient = np.zeros_like(coefficients)
                    value = 1e100
                    evaluation = None
                cache.clear()
                cache[key] = (coefficients.copy(), float(value), gradient.copy(), evaluation)
            return cache[key]

        initial_coefficients = np.zeros(self.config.num_control_basis, dtype=np.float64)
        opt = minimize(
            lambda c: evaluate(c)[1],
            initial_coefficients,
            jac=lambda c: evaluate(c)[2],
            method="L-BFGS-B",
            options={
                "maxiter": self.config.max_iter,
                "gtol": self.config.convergence_tol,
                "ftol": 1e-12,
                "maxls": self.config.line_search_max_iter,
            },
        )
        _coeff, _value, _coef_grad, evaluation = evaluate(opt.x)
        if evaluation is None:
            return self._failure_result(initial_state, str(opt.message), int(opt.nit))
        if evaluation.reduced_gradient_norm <= self.config.convergence_tol:
            return self._success_result(initial_state, evaluation, int(opt.nit), str(opt.message))
        return self._failure_result(
            initial_state,
            (
                f"{opt.message}; reduced_gradient_norm="
                f"{evaluation.reduced_gradient_norm:.3e} > {self.config.convergence_tol:.3e}"
            ),
            int(opt.nit),
            evaluation,
        )

    def _evaluate_time_grid_control(
        self,
        initial_state: np.ndarray,
        control: np.ndarray,
    ) -> _Evaluation:
        control = np.asarray(control, dtype=np.float64)
        if control.shape != self.time_grid.shape:
            raise ValueError("control must have the same shape as time_grid")
        state = self._integrate_state_crank_nicolson(initial_state, control)
        adjoint = self._integrate_adjoint_crank_nicolson(state)
        return self._build_evaluation(state, adjoint, control, coefficient_gradient=None)

    def _evaluate_legendre_control(
        self,
        initial_state: np.ndarray,
        coefficients: np.ndarray,
    ) -> _Evaluation:
        coefficients = np.asarray(coefficients, dtype=np.float64)

        def control_at(t: float) -> float:
            return float(self._legendre_values(coefficients, np.array([t]))[0])

        def state_rhs(t: float, y: np.ndarray) -> np.ndarray:
            return self.problem.dynamics(t, y, control_at(t))

        state_sol = solve_ivp(
            state_rhs,
            self.domain,
            initial_state,
            t_eval=self.time_grid,
            dense_output=True,
            rtol=self.config.ivp_rtol,
            atol=self.config.ivp_atol,
        )
        if not state_sol.success:
            raise RuntimeError(f"state integration failed: {state_sol.message}")

        terminal_adjoint = self.problem.terminal_gradient(state_sol.y[:, -1])

        def adjoint_rhs(t: float, p: np.ndarray) -> np.ndarray:
            return self.problem.adjoint_rhs(t, state_sol.sol(t), p)

        adjoint_sol = solve_ivp(
            adjoint_rhs,
            (self.domain[1], self.domain[0]),
            terminal_adjoint,
            t_eval=self.time_grid[::-1],
            rtol=self.config.ivp_rtol,
            atol=self.config.ivp_atol,
        )
        if not adjoint_sol.success:
            raise RuntimeError(f"adjoint integration failed: {adjoint_sol.message}")

        state = state_sol.y
        adjoint = adjoint_sol.y[:, ::-1]
        control = self._legendre_values(coefficients, self.time_grid)
        reduced_gradient = self.problem.reduced_gradient(adjoint, control)
        coefficient_gradient = np.array(
            [
                np.trapezoid(reduced_gradient * self._basis_matrix[:, i], self.time_grid)
                for i in range(self.config.num_control_basis)
            ],
            dtype=np.float64,
        )
        return self._build_evaluation(state, adjoint, control, coefficient_gradient)

    def _integrate_state_crank_nicolson(
        self,
        initial_state: np.ndarray,
        control: np.ndarray,
    ) -> np.ndarray:
        state = np.empty((2, self.time_grid.shape[0]), dtype=np.float64)
        state[:, 0] = initial_state
        for idx in range(self.time_grid.shape[0] - 1):
            t0 = float(self.time_grid[idx])
            t1 = float(self.time_grid[idx + 1])
            dt = t1 - t0
            y0 = state[:, idx]
            u0 = float(control[idx])
            u1 = float(control[idx + 1])
            f0 = self.problem.dynamics(t0, y0, u0)

            def equation(y_next: np.ndarray) -> np.ndarray:
                return y_next - y0 - 0.5 * dt * (
                    f0 + self.problem.dynamics(t1, y_next, u1)
                )

            guess = y0 + dt * f0
            solved = root(equation, guess, tol=self.config.cn_root_tol)
            if not solved.success:
                raise RuntimeError(f"state CN step {idx} failed: {solved.message}")
            state[:, idx + 1] = solved.x
        return state

    def _integrate_adjoint_crank_nicolson(self, state: np.ndarray) -> np.ndarray:
        adjoint = np.empty_like(state)
        adjoint[:, -1] = self.problem.terminal_gradient(state[:, -1])
        for idx in range(self.time_grid.shape[0] - 2, -1, -1):
            t0 = float(self.time_grid[idx])
            t1 = float(self.time_grid[idx + 1])
            dt = t1 - t0
            p1 = adjoint[:, idx + 1]
            g1 = self.problem.adjoint_rhs(t1, state[:, idx + 1], p1)

            def equation(p0: np.ndarray) -> np.ndarray:
                g0 = self.problem.adjoint_rhs(t0, state[:, idx], p0)
                return p1 - p0 - 0.5 * dt * (g0 + g1)

            solved = root(equation, p1, tol=self.config.cn_root_tol)
            if not solved.success:
                raise RuntimeError(f"adjoint CN step {idx} failed: {solved.message}")
            adjoint[:, idx] = solved.x
        return adjoint

    def _build_evaluation(
        self,
        state: np.ndarray,
        adjoint: np.ndarray,
        control: np.ndarray,
        coefficient_gradient: np.ndarray | None,
    ) -> _Evaluation:
        running = self.problem.running_cost(state, control)
        value = float(np.trapezoid(running, self.time_grid))
        reduced_gradient = self.problem.reduced_gradient(adjoint, control)
        reduced_gradient_norm = _trapz_norm(self.time_grid, reduced_gradient)
        coefficient_gradient_norm = (
            None
            if coefficient_gradient is None
            else float(np.linalg.norm(coefficient_gradient))
        )
        return _Evaluation(
            value=value,
            gradient=reduced_gradient,
            reduced_gradient_norm=reduced_gradient_norm,
            coefficient_gradient=coefficient_gradient,
            coefficient_gradient_norm=coefficient_gradient_norm,
            state=state,
            adjoint=adjoint,
            control=control,
        )

    def _bb_step(
        self,
        previous_control: np.ndarray,
        current_control: np.ndarray,
        previous_gradient: np.ndarray,
        current_gradient: np.ndarray,
        iteration: int,
    ) -> float:
        step_delta = current_control - previous_control
        gradient_delta = current_gradient - previous_gradient
        sy = _trapz_dot(self.time_grid, step_delta, gradient_delta)
        ss = _trapz_dot(self.time_grid, step_delta, step_delta)
        yy = _trapz_dot(self.time_grid, gradient_delta, gradient_delta)
        if sy <= 0.0 or ss <= 0.0 or yy <= 0.0:
            return self.config.alpha_default
        step = ss / sy if iteration % 2 == 0 else sy / yy
        if not np.isfinite(step) or step <= 0.0:
            step = self.config.alpha_default
        return float(np.clip(step, self.config.alpha_min, self.config.alpha_max))

    def _line_search_time_grid(
        self,
        initial_state: np.ndarray,
        current_control: np.ndarray,
        current_eval: _Evaluation,
        step_size: float,
    ) -> tuple[np.ndarray, _Evaluation | None, str]:
        accepted_step = float(step_size)
        last_failure = ""
        for _ in range(self.config.line_search_max_iter):
            trial_control = current_control - accepted_step * current_eval.gradient
            try:
                trial_eval = self._evaluate_time_grid_control(initial_state, trial_control)
            except RuntimeError as exc:
                last_failure = str(exc)
            else:
                allowed_value = current_eval.value * (1.0 + self.config.line_search_cost_tol)
                if np.isfinite(trial_eval.value) and trial_eval.value <= allowed_value:
                    return trial_control, trial_eval, ""
                last_failure = (
                    f"trial value {trial_eval.value:.6e} exceeded "
                    f"allowed value {allowed_value:.6e}"
                )
            accepted_step *= self.config.line_search_beta
            if accepted_step < self.config.alpha_min:
                break
        return current_control, None, f"line search failed: {last_failure}"

    def _success_result(
        self,
        initial_state: np.ndarray,
        evaluation: _Evaluation,
        iterations: int,
        message: str = "converged",
    ) -> VdpSampleResult:
        return VdpSampleResult(
            initial_state=initial_state.copy(),
            converged=True,
            message=message,
            value=float(evaluation.value),
            gradient=evaluation.adjoint[:, 0].copy(),
            reduced_gradient_norm=float(evaluation.reduced_gradient_norm),
            coefficient_gradient_norm=evaluation.coefficient_gradient_norm,
            iterations=int(iterations),
            control=evaluation.control.copy() if self.config.store_trajectories else None,
            state_trajectory=evaluation.state.copy() if self.config.store_trajectories else None,
            adjoint_trajectory=evaluation.adjoint.copy() if self.config.store_trajectories else None,
            reduced_gradient=evaluation.gradient.copy() if self.config.store_trajectories else None,
        )

    def _failure_result(
        self,
        initial_state: np.ndarray,
        message: str,
        iterations: int,
        evaluation: _Evaluation | None = None,
    ) -> VdpSampleResult:
        return VdpSampleResult(
            initial_state=initial_state.copy(),
            converged=False,
            message=message,
            value=None if evaluation is None else float(evaluation.value),
            gradient=None if evaluation is None else evaluation.adjoint[:, 0].copy(),
            reduced_gradient_norm=None if evaluation is None else float(evaluation.reduced_gradient_norm),
            coefficient_gradient_norm=None if evaluation is None else evaluation.coefficient_gradient_norm,
            iterations=int(iterations),
            control=(
                evaluation.control.copy()
                if evaluation is not None and self.config.store_trajectories
                else None
            ),
            state_trajectory=(
                evaluation.state.copy()
                if evaluation is not None and self.config.store_trajectories
                else None
            ),
            adjoint_trajectory=(
                evaluation.adjoint.copy()
                if evaluation is not None and self.config.store_trajectories
                else None
            ),
            reduced_gradient=(
                evaluation.gradient.copy()
                if evaluation is not None and self.config.store_trajectories
                else None
            ),
        )

    def _build_time_grid(self) -> np.ndarray:
        if self.config.num_time_points is not None:
            return np.linspace(0.0, self.problem.T_final, self.config.num_time_points)
        if self.config.time_step is not None:
            count = int(round(self.problem.T_final / self.config.time_step))
            return np.linspace(0.0, self.problem.T_final, count + 1)
        if self.config.profile == "paper":
            count = int(round(self.problem.T_final / 1e-4))
            return np.linspace(0.0, self.problem.T_final, count + 1)
        return np.linspace(0.0, self.problem.T_final, 301)

    def _map_to_legendre_domain(self, t: np.ndarray) -> np.ndarray:
        a, b = self.domain
        return (2.0 * np.asarray(t, dtype=np.float64) - (a + b)) / (b - a)

    def _basis(self, t: np.ndarray) -> np.ndarray:
        return legvander(self._map_to_legendre_domain(t), self.config.num_control_basis - 1)

    def _legendre_values(self, coefficients: np.ndarray, t: np.ndarray) -> np.ndarray:
        return legval(self._map_to_legendre_domain(t), coefficients)


__all__ = [
    "VdpOpenLoopSolver",
    "VdpOpenLoopSolverConfig",
    "VdpOpenLoopSolution",
    "VdpSampleResult",
]
