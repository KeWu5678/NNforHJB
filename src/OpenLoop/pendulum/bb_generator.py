"""BB open-loop data generation for the pendulum swing-up problem."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.linalg import solve_continuous_are

from src.OpenLoop.bvp_bb_optimizer import BvpBbOpenLoopOptimizer


PENDULUM_BB_DTYPE = np.dtype(
    [("x", "2float64"), ("dv", "2float64"), ("v", "float64")]
)


@dataclass
class PendulumBBResult:
    initial_state: np.ndarray
    gradient: np.ndarray
    value: float
    converged: bool
    message: str
    seed_index: int
    iterations: int
    gradient_norm: float


class PendulumBBDataGenerator:
    """Generate pendulum value/gradient data with the repo's BVP/BB optimizer."""

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
        dt: float = 0.02,
        num_basis: int = 30,
        control_seed_amplitudes: tuple[float, ...] = (0.0, 2.0, -2.0),
        bvp_tol: float = 1e-6,
        max_nodes: int = 10000,
        line_search_cost_tol: float = 0.0,
        alpha_max: float = 1.0,
    ) -> None:
        self.mass = float(mass)
        self.length = float(length)
        self.damping = float(damping)
        self.gravity = float(gravity)
        self.state_weights = np.asarray(state_weights, dtype=float)
        self.control_weight = float(control_weight)
        self.terminal_weight = float(terminal_weight)
        self.T_final = float(T_final)
        self.dt = float(dt)
        self.num_basis = int(num_basis)
        self.control_seed_amplitudes = tuple(float(a) for a in control_seed_amplitudes)
        self.bvp_tol = float(bvp_tol)
        self.max_nodes = int(max_nodes)
        self.line_search_cost_tol = float(line_search_cost_tol)
        self.alpha_max = float(alpha_max)
        self.x0_values: np.ndarray | None = None

        self._validate()
        self.control_gain = 1.0 / (self.mass * self.length**2)
        self.damping_gain = self.damping / (self.mass * self.length**2)
        self.gravity_gain = self.gravity / self.length
        self.local_lqr_matrix = self._compute_local_lqr_matrix()

    def _validate(self) -> None:
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
        if self.terminal_weight < 0.0:
            raise ValueError("terminal_weight must be nonnegative")
        if self.T_final <= 0.0:
            raise ValueError("T_final must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.num_basis <= 0:
            raise ValueError("num_basis must be positive")

    @staticmethod
    def wrap_angle(theta: np.ndarray | float) -> np.ndarray:
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

    def time_grid(self) -> np.ndarray:
        n_steps = int(np.ceil(self.T_final / self.dt))
        return np.linspace(0.0, self.T_final, n_steps + 1)

    def initial_guess(self, initial_state: np.ndarray, grid: np.ndarray) -> np.ndarray:
        initial_state = np.asarray(initial_state, dtype=float)
        guess = np.zeros((4, grid.size), dtype=float)
        ramp = 1.0 - grid / grid[-1]
        guess[0, :] = initial_state[0] * ramp
        guess[1, :] = initial_state[1] * ramp
        return guess

    def terminal_cost(self, terminal_state: np.ndarray) -> float:
        if self.terminal_weight == 0.0:
            return 0.0
        diff = np.asarray(terminal_state, dtype=float).copy()
        diff[0] = float(self.wrap_angle(diff[0]))
        return float(self.terminal_weight * diff @ self.local_lqr_matrix @ diff)

    def terminal_gradient(self, terminal_state: np.ndarray) -> np.ndarray:
        if self.terminal_weight == 0.0:
            return np.zeros(2, dtype=float)
        diff = np.asarray(terminal_state, dtype=float).copy()
        diff[0] = float(self.wrap_angle(diff[0]))
        return 2.0 * self.terminal_weight * self.local_lqr_matrix @ diff

    def pendulum_tpbvp(self, _t: np.ndarray, y: np.ndarray, u) -> np.ndarray:
        theta = y[0, :]
        omega = y[1, :]
        p1 = y[2, :]
        p2 = y[3, :]
        q1, q2 = self.state_weights
        u_values = u(_t)
        return np.vstack(
            [
                omega,
                -self.damping_gain * omega
                + self.gravity_gain * np.sin(theta)
                + self.control_gain * u_values,
                -2.0 * q1 * np.sin(theta) - self.gravity_gain * np.cos(theta) * p2,
                -2.0 * q2 * omega - p1 + self.damping_gain * p2,
            ]
        )

    def gradient(self, u_values: np.ndarray, p2_values: np.ndarray) -> np.ndarray:
        if len(u_values) != len(p2_values):
            raise ValueError("u and p2 must have the same length")
        return self.control_gain * p2_values + 2.0 * self.control_weight * u_values

    def running_cost(
        self,
        theta_values: np.ndarray,
        omega_values: np.ndarray,
        u_values: np.ndarray,
    ) -> np.ndarray:
        q1, q2 = self.state_weights
        return (
            q1 * (2.0 - 2.0 * np.cos(theta_values))
            + q2 * omega_values * omega_values
            + self.control_weight * u_values * u_values
        )

    def V(
        self,
        grid: np.ndarray,
        u_values: np.ndarray,
        theta_values: np.ndarray,
        omega_values: np.ndarray,
    ) -> float:
        running = self.running_cost(theta_values, omega_values, u_values)
        terminal_state = np.array([theta_values[-1], omega_values[-1]], dtype=float)
        return float(np.trapezoid(running, grid) + self.terminal_cost(terminal_state))

    def gen_bc(self, initial_state: np.ndarray):
        initial_state = np.asarray(initial_state, dtype=float)

        def bc(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
            p_terminal = self.terminal_gradient(yb[:2])
            return np.array(
                [
                    ya[0] - initial_state[0],
                    ya[1] - initial_state[1],
                    yb[2] - p_terminal[0],
                    yb[3] - p_terminal[1],
                ]
            )

        return bc

    def feedback_from_gradient(self, gradient: np.ndarray) -> float:
        gradient = np.asarray(gradient, dtype=float)
        return float(-self.control_gain * gradient[1] / (2.0 * self.control_weight))

    def hjb_residual(
        self,
        state: np.ndarray,
        gradient: np.ndarray,
        control: float | None = None,
    ) -> float:
        state = np.asarray(state, dtype=float)
        gradient = np.asarray(gradient, dtype=float)
        if control is None:
            control = self.feedback_from_gradient(gradient)
        theta, omega = state
        dynamics = np.array(
            [
                omega,
                -self.damping_gain * omega
                + self.gravity_gain * np.sin(theta)
                + self.control_gain * control,
            ]
        )
        running = self.running_cost(
            np.array([theta]),
            np.array([omega]),
            np.array([control]),
        )[0]
        return float(running + gradient @ dynamics)

    def default_control_seeds(self) -> list[np.ndarray]:
        seeds = []
        seen: set[float] = set()
        for amplitude in self.control_seed_amplitudes:
            if amplitude in seen:
                continue
            seen.add(amplitude)
            coeff = np.zeros(self.num_basis, dtype=float)
            coeff[0] = amplitude
            seeds.append(coeff)
        return seeds

    def solve_initial_state(
        self,
        initial_state: np.ndarray,
        tol: float = 1e-5,
        max_it: int = 200,
        control_seeds: Iterable[np.ndarray] | None = None,
        verbose: bool = False,
    ) -> tuple[PendulumBBResult | None, list[PendulumBBResult]]:
        initial_state = np.asarray(initial_state, dtype=float)
        grid = self.time_grid()
        guess = self.initial_guess(initial_state, grid)
        seeds = list(self.default_control_seeds() if control_seeds is None else control_seeds)
        results: list[PendulumBBResult] = []

        for seed_index, seed_coefficients in enumerate(seeds):
            optimizer = BvpBbOpenLoopOptimizer(
                self.pendulum_tpbvp,
                self.gen_bc(initial_state),
                self.V,
                self.gradient,
                grid,
                guess,
                tol,
                max_it,
                num_basis=self.num_basis,
                domain=(0.0, self.T_final),
                bvp_tol=self.bvp_tol,
                max_nodes=self.max_nodes,
                initial_coefficients=np.asarray(seed_coefficients, dtype=float),
                alpha_max=self.alpha_max,
                line_search_cost_tol=self.line_search_cost_tol,
                verbose=verbose,
            )
            try:
                gradient, value = optimizer.optimize()
            except Exception as exc:
                results.append(
                    PendulumBBResult(
                        initial_state=initial_state,
                        gradient=np.full(2, np.nan),
                        value=np.nan,
                        converged=False,
                        message=str(exc),
                        seed_index=seed_index,
                        iterations=0,
                        gradient_norm=np.inf,
                    )
                )
                continue

            converged = gradient is not None and np.isfinite(value)
            results.append(
                PendulumBBResult(
                    initial_state=initial_state,
                    gradient=np.asarray(gradient if converged else np.full(2, np.nan)),
                    value=float(value) if converged else np.nan,
                    converged=bool(converged),
                    message=optimizer.last_message,
                    seed_index=seed_index,
                    iterations=int(optimizer.last_iterations),
                    gradient_norm=float(
                        optimizer.last_gradient_norm
                        if optimizer.last_gradient_norm is not None
                        else np.inf
                    ),
                )
            )

        successful = [result for result in results if result.converged]
        if not successful:
            return None, results
        return min(successful, key=lambda result: result.value), results

    def apply_initial_gridding(
        self,
        nx_theta: int,
        nx_omega: int,
        theta_range: tuple[float, float] = (-np.pi, np.pi),
        omega_range: tuple[float, float] = (-8.0, 8.0),
    ) -> None:
        theta_values = np.linspace(theta_range[0], theta_range[1], nx_theta)
        omega_values = np.linspace(omega_range[0], omega_range[1], nx_omega)
        theta_grid, omega_grid = np.meshgrid(theta_values, omega_values)
        self.x0_values = np.column_stack((theta_grid.ravel(), omega_grid.ravel()))
        print(f"Created {self.x0_values.shape[0]} pendulum initial states")

    def apply_random_initial_sampling(
        self,
        n_samples: int,
        theta_range: tuple[float, float] = (-np.pi, np.pi),
        omega_range: tuple[float, float] = (-8.0, 8.0),
        seed: int | None = None,
    ) -> None:
        rng = np.random.default_rng(seed)
        theta_values = rng.uniform(theta_range[0], theta_range[1], n_samples)
        omega_values = rng.uniform(omega_range[0], omega_range[1], n_samples)
        self.x0_values = np.column_stack((theta_values, omega_values))
        print(f"Sampled {self.x0_values.shape[0]} pendulum initial states")

    def data_generation(
        self,
        tol: float = 1e-5,
        max_it: int = 200,
        verbose: bool = False,
    ) -> tuple[np.ndarray, list[dict[str, object]]]:
        if self.x0_values is None:
            raise RuntimeError("generate initial values before data_generation().")

        dataset = np.zeros(self.x0_values.shape[0], dtype=PENDULUM_BB_DTYPE)
        dataset["x"] = self.x0_values
        dataset["dv"] = np.nan
        dataset["v"] = np.nan
        failed: list[dict[str, object]] = []

        for i, initial_state in enumerate(self.x0_values):
            print(f"i = {i}, ini = {initial_state}")
            best, attempts = self.solve_initial_state(
                initial_state,
                tol=tol,
                max_it=max_it,
                verbose=verbose,
            )
            if best is None:
                failed.append(
                    {
                        "index": i,
                        "x": initial_state.tolist(),
                        "attempts": [self.result_to_dict(attempt) for attempt in attempts],
                    }
                )
                continue
            dataset[i] = (initial_state, best.gradient, best.value)
        return dataset[np.isfinite(dataset["v"])], failed

    @staticmethod
    def result_to_dict(result: PendulumBBResult) -> dict[str, object]:
        return {
            "x": result.initial_state.tolist(),
            "dv": result.gradient.tolist(),
            "v": result.value,
            "converged": result.converged,
            "message": result.message,
            "seed_index": result.seed_index,
            "iterations": result.iterations,
            "gradient_norm": result.gradient_norm,
        }

    def data_save(
        self,
        dataset: np.ndarray,
        failed: list[dict[str, object]],
        output_dir: str | Path,
        tag: str | None = None,
    ) -> tuple[Path, Path | None]:
        base_dir = Path(output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        date_tag = datetime.now().strftime("%Y%m%d") if tag is None else tag
        data_path = base_dir / f"PENDULUM_bb_openloop_{date_tag}.npy"
        failed_path = base_dir / f"PENDULUM_bb_openloop_failed_{date_tag}.json"

        np.save(data_path, dataset)
        if failed:
            failed_path.write_text(json.dumps(failed, indent=2), encoding="utf-8")
            return data_path, failed_path
        return data_path, None
