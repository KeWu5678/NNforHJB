"""BB open-loop data generation for the pendulum swing-up problem."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from src.OpenLoop.bvp_bb_optimizer import BvpBbOpenLoopOptimizer
from src.OpenLoop.pendulum.swingup_dynamics import PendulumSwingUpDynamics
from src.OpenLoop.sample_sets import (
    grid_initial_states,
    random_initial_states,
    save_dataset_bundle,
)


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
        self.dynamics = PendulumSwingUpDynamics(
            mass=mass,
            length=length,
            damping=damping,
            gravity=gravity,
            state_weights=state_weights,
            control_weight=control_weight,
        )
        self.mass = self.dynamics.mass
        self.length = self.dynamics.length
        self.damping = self.dynamics.damping
        self.gravity = self.dynamics.gravity
        self.state_weights = self.dynamics.state_weights
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
        self.control_gain = self.dynamics.control_gain
        self.damping_gain = self.dynamics.damping_gain
        self.gravity_gain = self.dynamics.gravity_gain
        self.local_lqr_matrix = self.dynamics.local_lqr_matrix

    def _validate(self) -> None:
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
        return PendulumSwingUpDynamics.wrap_angle(theta)

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
        return self.dynamics.local_lqr_value(
            terminal_state,
            weight=self.terminal_weight,
        )

    def terminal_gradient(self, terminal_state: np.ndarray) -> np.ndarray:
        if self.terminal_weight == 0.0:
            return np.zeros(2, dtype=float)
        return self.dynamics.local_lqr_gradient(
            terminal_state,
            weight=self.terminal_weight,
        )

    def pendulum_tpbvp(self, _t: np.ndarray, y: np.ndarray, u) -> np.ndarray:
        state = y[:2, :].T
        costate = y[2:4, :].T
        u_values = u(_t)
        state_rhs = self.dynamics.forward_dynamics(state, u_values).T
        costate_rhs = self.dynamics.costate_rhs(state, costate).T
        return np.vstack(
            [
                state_rhs,
                costate_rhs,
            ]
        )

    def gradient(self, u_values: np.ndarray, p2_values: np.ndarray) -> np.ndarray:
        if len(u_values) != len(p2_values):
            raise ValueError("u and p2 must have the same length")
        costate = np.column_stack((np.zeros_like(p2_values), p2_values))
        return self.dynamics.stationarity_residual(costate, u_values)

    def running_cost(
        self,
        theta_values: np.ndarray,
        omega_values: np.ndarray,
        u_values: np.ndarray,
    ) -> np.ndarray:
        state = np.column_stack((theta_values, omega_values))
        return self.dynamics.running_cost(state, u_values)

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
        return float(self.dynamics.minimizing_control(gradient))

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
        state_rhs = self.dynamics.forward_dynamics(state, control)
        running = self.running_cost(
            np.array([state[0]]),
            np.array([state[1]]),
            np.array([control]),
        )[0]
        return float(running + gradient @ state_rhs)

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
                result = optimizer.optimize_result()
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

            converged = (
                result.gradient is not None
                and result.value is not None
                and result.converged
                and np.isfinite(result.value)
            )
            results.append(
                PendulumBBResult(
                    initial_state=initial_state,
                    gradient=np.asarray(
                        result.gradient if converged else np.full(2, np.nan)
                    ),
                    value=float(result.value) if converged else np.nan,
                    converged=bool(converged),
                    message=result.message,
                    seed_index=seed_index,
                    iterations=int(result.iterations),
                    gradient_norm=float(
                        result.gradient_norm if result.gradient_norm is not None else np.inf
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
        self.x0_values = grid_initial_states(
            nx_theta,
            nx_omega,
            theta_range,
            omega_range,
        )
        print(f"Created {self.x0_values.shape[0]} pendulum initial states")

    def apply_random_initial_sampling(
        self,
        n_samples: int,
        theta_range: tuple[float, float] = (-np.pi, np.pi),
        omega_range: tuple[float, float] = (-8.0, 8.0),
        seed: int | None = None,
    ) -> None:
        self.x0_values = random_initial_states(
            n_samples,
            theta_range,
            omega_range,
            seed=seed,
        )
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

        saved = save_dataset_bundle(
            base_dir,
            arrays={data_path.name: dataset},
            json_files={failed_path.name: failed} if failed else None,
        )
        return saved[data_path.name], saved.get(failed_path.name)
