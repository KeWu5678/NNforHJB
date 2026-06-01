"""Accurate pendulum open-loop data from forward-backward reduced gradients."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable

import numpy as np

from src.OpenLoop.forward_backward_optimizer import (
    ForwardBackwardOpenLoopOptimizer,
    ForwardBackwardOpenLoopResult,
)
from src.OpenLoop.pendulum.finite_horizon_problem import PendulumSwingUpProblem
from src.OpenLoop.sample_sets import (
    grid_initial_states,
    random_initial_states,
    save_dataset_bundle,
)


PENDULUM_FINITE_HORIZON_DTYPE = np.dtype(
    [("x", "2float64"), ("dv", "2float64"), ("v", "float64")]
)
PENDULUM_TRANSIENT_DTYPE = PENDULUM_FINITE_HORIZON_DTYPE


@dataclass
class PendulumFiniteHorizonAttempt:
    initial_state: np.ndarray
    seed_amplitude: float
    value: float
    gradient: np.ndarray
    accepted: bool
    optimizer_converged: bool
    coefficient_gradient_norm: float
    residual_l2_squared: float
    iterations: int
    message: str


class PendulumFiniteHorizonDataGenerator:
    """Generate training samples with forward state and backward adjoint solves."""

    def __init__(
        self,
        T_final: float = 3.0,
        terminal_weight: float = 1.0,
        num_basis: int = 20,
        time_steps: int = 301,
        stationarity_tol: float = 1e-5,
        coefficient_gradient_tol: float = 1e-5,
        optimizer_method: str = "BFGS",
        max_iter: int = 500,
        ivp_rtol: float = 1e-8,
        ivp_atol: float = 1e-10,
        control_seed_amplitudes: tuple[float, ...] = (0.0, 2.0, -2.0, 5.0, -5.0, 10.0, -10.0),
    ) -> None:
        self.problem = PendulumSwingUpProblem(
            T_final=T_final,
            terminal_weight=terminal_weight,
        )
        self.num_basis = int(num_basis)
        self.time_steps = int(time_steps)
        self.stationarity_tol = float(stationarity_tol)
        self.coefficient_gradient_tol = float(coefficient_gradient_tol)
        self.optimizer_method = optimizer_method
        self.max_iter = int(max_iter)
        self.ivp_rtol = float(ivp_rtol)
        self.ivp_atol = float(ivp_atol)
        self.control_seed_amplitudes = tuple(float(a) for a in control_seed_amplitudes)
        self.x0_values: np.ndarray | None = None

        if self.num_basis <= 0:
            raise ValueError("num_basis must be positive")
        if self.time_steps < 2:
            raise ValueError("time_steps must be at least 2")
        if self.stationarity_tol <= 0.0:
            raise ValueError("stationarity_tol must be positive")
        if self.coefficient_gradient_tol <= 0.0:
            raise ValueError("coefficient_gradient_tol must be positive")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")

    def time_grid(self) -> np.ndarray:
        return np.linspace(0.0, self.problem.T_final, self.time_steps)

    def initial_coefficients(self, amplitude: float) -> np.ndarray:
        coefficients = np.zeros(self.num_basis, dtype=float)
        coefficients[0] = float(amplitude)
        return coefficients

    def solve_seed(
        self,
        initial_state: np.ndarray,
        seed_amplitude: float,
    ) -> PendulumFiniteHorizonAttempt:
        optimizer = ForwardBackwardOpenLoopOptimizer(
            problem=self.problem,
            initial_state=np.asarray(initial_state, dtype=float),
            fixed_target_sign=1,
            num_basis=self.num_basis,
            time_grid=self.time_grid(),
            initial_coefficients=self.initial_coefficients(seed_amplitude),
            optimizer_method=self.optimizer_method,
            max_iter=self.max_iter,
            coefficient_gradient_tol=self.coefficient_gradient_tol,
            optimizer_ftol=0.0,
            max_line_search=100,
            ivp_rtol=self.ivp_rtol,
            ivp_atol=self.ivp_atol,
        )
        try:
            result = optimizer.optimize()
            residual_l2_squared = self.residual_l2_squared(optimizer, result)
            accepted = bool(
                np.isfinite(result.value)
                and np.isfinite(residual_l2_squared)
                and residual_l2_squared <= self.stationarity_tol
            )
            return PendulumFiniteHorizonAttempt(
                initial_state=np.asarray(initial_state, dtype=float),
                seed_amplitude=float(seed_amplitude),
                value=float(result.value),
                gradient=result.gradient.copy(),
                accepted=accepted,
                optimizer_converged=bool(result.converged),
                coefficient_gradient_norm=float(result.coefficient_gradient_norm),
                residual_l2_squared=float(residual_l2_squared),
                iterations=int(result.iterations),
                message=result.message,
            )
        except Exception as exc:
            return PendulumFiniteHorizonAttempt(
                initial_state=np.asarray(initial_state, dtype=float),
                seed_amplitude=float(seed_amplitude),
                value=np.nan,
                gradient=np.full(2, np.nan),
                accepted=False,
                optimizer_converged=False,
                coefficient_gradient_norm=np.inf,
                residual_l2_squared=np.inf,
                iterations=0,
                message=str(exc),
            )

    @staticmethod
    def residual_l2_squared(
        optimizer: ForwardBackwardOpenLoopOptimizer,
        result: ForwardBackwardOpenLoopResult,
    ) -> float:
        _value, _gradient, info = optimizer.evaluate_fixed_target(
            result.control_coefficients,
            result.fixed_target_sign,
        )
        return float(np.trapezoid(info["residual"] ** 2, optimizer.time_grid))

    def solve_initial_state(
        self,
        initial_state: np.ndarray,
        seed_amplitudes: Iterable[float] | None = None,
    ) -> tuple[PendulumFiniteHorizonAttempt | None, list[PendulumFiniteHorizonAttempt]]:
        seeds = self.control_seed_amplitudes if seed_amplitudes is None else tuple(seed_amplitudes)
        attempts = [self.solve_seed(initial_state, seed) for seed in seeds]
        accepted = [attempt for attempt in attempts if attempt.accepted]
        if not accepted:
            return None, attempts
        return min(accepted, key=lambda attempt: attempt.value), attempts

    def apply_initial_gridding(
        self,
        nx_theta: int,
        nx_omega: int,
        theta_range: tuple[float, float] = (-np.pi, np.pi),
        omega_range: tuple[float, float] = (-4.0, 4.0),
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
        omega_range: tuple[float, float] = (-4.0, 4.0),
        seed: int | None = None,
    ) -> None:
        self.x0_values = random_initial_states(
            n_samples,
            theta_range,
            omega_range,
            seed=seed,
        )
        print(f"Sampled {self.x0_values.shape[0]} pendulum initial states")

    def data_generation(self) -> tuple[np.ndarray, list[dict[str, object]], list[dict[str, object]]]:
        if self.x0_values is None:
            raise RuntimeError("generate initial values before data_generation().")

        rows: list[tuple[np.ndarray, np.ndarray, float]] = []
        failed: list[dict[str, object]] = []
        diagnostics: list[dict[str, object]] = []

        for i, initial_state in enumerate(self.x0_values):
            print(f"i = {i}, ini = {initial_state}")
            best, attempts = self.solve_initial_state(initial_state)
            diagnostics.extend(self.attempt_to_dict(i, attempt) for attempt in attempts)
            if best is None:
                failed.append(
                    {
                        "index": i,
                        "x": initial_state.tolist(),
                        "best_residual_l2_squared": float(
                            min(attempt.residual_l2_squared for attempt in attempts)
                        ),
                        "attempts": [self.attempt_to_dict(i, attempt) for attempt in attempts],
                    }
                )
                continue
            rows.append((initial_state.copy(), best.gradient.copy(), float(best.value)))

        dataset = np.zeros(len(rows), dtype=PENDULUM_FINITE_HORIZON_DTYPE)
        for i, (x_value, gradient, value) in enumerate(rows):
            dataset[i] = (x_value, gradient, value)
        return dataset, failed, diagnostics

    def data_generation_parallel(
        self,
        workers: int,
        checkpoint_dir: Path | None = None,
        checkpoint_tag: str | None = None,
        checkpoint_every: int = 1,
    ) -> tuple[np.ndarray, list[dict[str, object]], list[dict[str, object]]]:
        if self.x0_values is None:
            raise RuntimeError("generate initial values before data_generation_parallel().")
        if workers <= 1:
            return self.data_generation()
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if checkpoint_tag is None:
                checkpoint_tag = "pendulum_transient_checkpoint"

        config = self.worker_config()
        rows: list[tuple[int, np.ndarray, np.ndarray, float]] = []
        failed: list[dict[str, object]] = []
        diagnostics: list[dict[str, object]] = []
        total = int(self.x0_values.shape[0])

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    solve_pendulum_finite_horizon_point,
                    config,
                    i,
                    self.x0_values[i].copy(),
                )
                for i in range(total)
            ]
            for completed, future in enumerate(as_completed(futures), start=1):
                index, row, failure, point_diagnostics = future.result()
                diagnostics.extend(point_diagnostics)
                if failure is not None:
                    failed.append(failure)
                    status = "failed"
                else:
                    rows.append((index, row[0], row[1], row[2]))
                    status = "accepted"
                print(f"completed {completed}/{total}: i = {index}, {status}", flush=True)
                if (
                    checkpoint_dir is not None
                    and checkpoint_tag is not None
                    and completed % checkpoint_every == 0
                ):
                    self.write_checkpoint(
                        checkpoint_dir,
                        checkpoint_tag,
                        rows,
                        failed,
                        diagnostics,
                        completed,
                        total,
                    )

        rows.sort(key=lambda item: item[0])
        failed.sort(key=lambda item: item["index"])
        diagnostics.sort(key=lambda item: (item["index"], item["seed_amplitude"]))

        dataset = self.rows_to_dataset(rows)
        if checkpoint_dir is not None and checkpoint_tag is not None:
            self.write_checkpoint(
                checkpoint_dir,
                checkpoint_tag,
                rows,
                failed,
                diagnostics,
                total,
                total,
            )
        return dataset, failed, diagnostics

    @staticmethod
    def rows_to_dataset(
        rows: list[tuple[int, np.ndarray, np.ndarray, float]],
    ) -> np.ndarray:
        dataset = np.zeros(len(rows), dtype=PENDULUM_FINITE_HORIZON_DTYPE)
        for i, (_index, x_value, gradient, value) in enumerate(rows):
            dataset[i] = (x_value, gradient, value)
        return dataset

    def write_checkpoint(
        self,
        checkpoint_dir: Path,
        checkpoint_tag: str,
        rows: list[tuple[int, np.ndarray, np.ndarray, float]],
        failed: list[dict[str, object]],
        diagnostics: list[dict[str, object]],
        completed: int,
        total: int,
    ) -> None:
        dataset = self.rows_to_dataset(rows)
        data_path = checkpoint_dir / f"PENDULUM_transient_openloop_{checkpoint_tag}_checkpoint.npy"
        failed_path = checkpoint_dir / f"PENDULUM_transient_openloop_failed_{checkpoint_tag}_checkpoint.json"
        diagnostics_path = checkpoint_dir / f"PENDULUM_transient_openloop_diagnostics_{checkpoint_tag}_checkpoint.json"
        meta_path = checkpoint_dir / f"PENDULUM_transient_openloop_{checkpoint_tag}_checkpoint_meta.json"

        np.save(data_path, dataset)
        failed_path.write_text(json.dumps(sorted(failed, key=lambda item: item["index"]), indent=2), encoding="utf-8")
        diagnostics_path.write_text(
            json.dumps(sorted(diagnostics, key=lambda item: (item["index"], item["seed_amplitude"])), indent=2),
            encoding="utf-8",
        )
        meta = {
            "completed": int(completed),
            "total": int(total),
            "accepted": int(dataset.shape[0]),
            "failed": int(len(failed)),
            "stationarity_tol": self.stationarity_tol,
            "coefficient_gradient_tol": self.coefficient_gradient_tol,
            "T_final": self.problem.T_final,
            "num_basis": self.num_basis,
            "time_steps": self.time_steps,
            "optimizer_method": self.optimizer_method,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def worker_config(self) -> dict[str, object]:
        return {
            "T_final": self.problem.T_final,
            "terminal_weight": self.problem.terminal_weight,
            "num_basis": self.num_basis,
            "time_steps": self.time_steps,
            "stationarity_tol": self.stationarity_tol,
            "coefficient_gradient_tol": self.coefficient_gradient_tol,
            "optimizer_method": self.optimizer_method,
            "max_iter": self.max_iter,
            "ivp_rtol": self.ivp_rtol,
            "ivp_atol": self.ivp_atol,
            "control_seed_amplitudes": self.control_seed_amplitudes,
        }

    @staticmethod
    def attempt_to_dict(index: int, attempt: PendulumFiniteHorizonAttempt) -> dict[str, object]:
        return {
            "index": int(index),
            "x": attempt.initial_state.tolist(),
            "seed_amplitude": attempt.seed_amplitude,
            "v": attempt.value,
            "dv": attempt.gradient.tolist(),
            "accepted": attempt.accepted,
            "optimizer_converged": attempt.optimizer_converged,
            "coefficient_gradient_norm": attempt.coefficient_gradient_norm,
            "residual_l2_squared": attempt.residual_l2_squared,
            "iterations": attempt.iterations,
            "message": attempt.message,
        }

    def data_save(
        self,
        dataset: np.ndarray,
        failed: list[dict[str, object]],
        diagnostics: list[dict[str, object]],
        output_dir: str | Path,
        tag: str | None = None,
    ) -> tuple[Path, Path, Path]:
        save_dir = Path(output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        date_tag = datetime.now().strftime("%Y%m%d_%H%M%S") if tag is None else tag
        data_path = save_dir / f"PENDULUM_transient_openloop_{date_tag}.npy"
        failed_path = save_dir / f"PENDULUM_transient_openloop_failed_{date_tag}.json"
        diagnostics_path = save_dir / f"PENDULUM_transient_openloop_diagnostics_{date_tag}.json"

        saved = save_dataset_bundle(
            save_dir,
            arrays={data_path.name: dataset},
            json_files={
                failed_path.name: failed,
                diagnostics_path.name: diagnostics,
            },
        )
        return saved[data_path.name], saved[failed_path.name], saved[diagnostics_path.name]


def solve_pendulum_finite_horizon_point(
    config: dict[str, object],
    index: int,
    initial_state: np.ndarray,
) -> tuple[
    int,
    tuple[np.ndarray, np.ndarray, float] | None,
    dict[str, object] | None,
    list[dict[str, object]],
]:
    generator = PendulumFiniteHorizonDataGenerator(**config)
    best, attempts = generator.solve_initial_state(initial_state)
    diagnostics = [generator.attempt_to_dict(index, attempt) for attempt in attempts]
    if best is None:
        failure = {
            "index": int(index),
            "x": np.asarray(initial_state, dtype=float).tolist(),
            "best_residual_l2_squared": float(
                min(attempt.residual_l2_squared for attempt in attempts)
            ),
            "attempts": diagnostics,
        }
        return int(index), None, failure, diagnostics
    row = (
        np.asarray(initial_state, dtype=float).copy(),
        best.gradient.copy(),
        float(best.value),
    )
    return int(index), row, None, diagnostics
