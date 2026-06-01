"""Compare BVP/BB and forward-backward open-loop optimizers.

The comparison is focused on the data needed for gradient-augmented training:

- value sample: V(0, x)
- gradient sample: p(0), which equals nabla V(0, x) under the PMP-HJB link

The BVP/BB path mirrors the reusable BVP/BB optimizer but is kept
inside this script so the existing optimizer file remains unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np
from scipy.integrate import solve_bvp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import utils
from src.OpenLoop.forward_backward_optimizer import (  # noqa: E402
    ForwardBackwardOpenLoopOptimizer,
)
from src.OpenLoop.vdp.phase_capture import VdpPhaseCaptureProblem  # noqa: E402


@dataclass
class BvpBbResult:
    value: float
    gradient: np.ndarray
    control_coefficients: np.ndarray
    residual_l2_squared: float
    coefficient_gradient_norm: float
    iterations: int
    converged: bool
    message: str
    seconds: float


def _legendre_control(coefficients: np.ndarray, domain: tuple[float, float]):
    return utils.gen_legendre(coefficients, domain=domain)


def _bvp_rhs(problem: VdpPhaseCaptureProblem, control):
    def rhs(t, z):
        u_values = control(t)
        y1 = z[0]
        y2 = z[1]
        p1 = z[2]
        p2 = z[3]
        return np.vstack(
            [
                y2,
                problem.mu * (1.0 - y1**2) * y2 - y1 + u_values,
                -2.0 * problem.state_weight * y1
                + p2 * (1.0 + 2.0 * problem.mu * y1 * y2),
                -2.0 * problem.state_weight * y2
                - p1
                - problem.mu * (1.0 - y1**2) * p2,
            ]
        )

    return rhs


def _bvp_bc(problem: VdpPhaseCaptureProblem, initial_state: np.ndarray, fixed_target_sign: int):
    def bc(ya, yb):
        p_terminal = problem.terminal_gradient(yb[:2], fixed_target_sign)
        return np.array(
            [
                ya[0] - initial_state[0],
                ya[1] - initial_state[1],
                yb[2] - p_terminal[0],
                yb[3] - p_terminal[1],
            ]
        )

    return bc


def _objective(
    problem: VdpPhaseCaptureProblem,
    grid: np.ndarray,
    control_values: np.ndarray,
    state_values: np.ndarray,
    fixed_target_sign: int,
) -> float:
    running = problem.running_cost(state_values, control_values)
    value = float(np.trapezoid(running, grid))
    value += problem.terminal_cost(state_values[:, -1], fixed_target_sign)
    return value


def _residual_and_coefficients(
    problem: VdpPhaseCaptureProblem,
    grid: np.ndarray,
    control_values: np.ndarray,
    adjoint_values: np.ndarray,
    num_basis: int,
    domain: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, float]:
    residual = problem.stationarity_residual(adjoint_values, control_values)
    residual_coeff = utils.fit_legendre(grid, residual, num_basis, domain=domain)
    residual_l2_squared = utils.L2(grid, residual)
    return residual, residual_coeff, residual_l2_squared


def solve_with_bvp_bb(
    problem: VdpPhaseCaptureProblem,
    initial_state: np.ndarray,
    fixed_target_sign: int,
    grid: np.ndarray,
    num_basis: int,
    tol: float,
    max_iter: int,
    bvp_tol: float,
    max_nodes: int,
) -> BvpBbResult:
    started = time.perf_counter()
    domain = (float(grid[0]), float(grid[-1]))
    guess = np.zeros((4, grid.size), dtype=float)
    guess[0, :] = initial_state[0]
    guess[1, :] = initial_state[1]

    def solve_for_coefficients(coefficients: np.ndarray, guess_values: np.ndarray):
        control = _legendre_control(coefficients, domain)
        return solve_bvp(
            _bvp_rhs(problem, control),
            _bvp_bc(problem, initial_state, fixed_target_sign),
            grid,
            guess_values,
            tol=bvp_tol,
            max_nodes=max_nodes,
        )

    def eval_coefficients(coefficients: np.ndarray, guess_values: np.ndarray):
        sol = solve_for_coefficients(coefficients, guess_values)
        if not sol.success:
            raise RuntimeError(sol.message)
        control = _legendre_control(coefficients, domain)
        u_values = control(sol.x)
        residual, residual_coeff, residual_l2_squared = _residual_and_coefficients(
            problem,
            sol.x,
            u_values,
            sol.y[2:4, :],
            num_basis,
            domain,
        )
        value = _objective(problem, sol.x, u_values, sol.y[0:2, :], fixed_target_sign)
        return sol, residual, residual_coeff, residual_l2_squared, value

    try:
        coeff0 = np.zeros(num_basis, dtype=float)
        sol0, _G0, G0_coeff, _G0_l2, _V0 = eval_coefficients(coeff0, guess)

        coeff1 = -0.1 * G0_coeff
        sol1, G1, G1_coeff, G1_l2, value1 = eval_coefficients(coeff1, sol0.sol(grid))

        alpha_min = 1e-5
        alpha_max = 10.0
        alpha_default = 0.1
        line_search_beta = 0.5
        line_search_cost_tol = 1e-2
        line_search_max_iter = 100
        k = 1

        while G1_l2 >= tol and k < max_iter:
            s = coeff1 - coeff0
            y = G1_coeff - G0_coeff
            sy = float(np.dot(s, y))
            ss = float(np.dot(s, s))
            yy = float(np.dot(y, y))
            eps = 1e-20
            alpha_bb1 = ss / (sy + eps)
            alpha_bb2 = sy / (yy + eps)
            alpha_trial = alpha_bb1 if k % 2 == 0 else alpha_bb2

            if sy <= 0 or alpha_trial <= 0 or not np.isfinite(alpha_trial):
                alpha = alpha_default
            else:
                alpha = float(np.clip(alpha_trial, alpha_min, alpha_max))

            direction = -G1_coeff
            accepted = False
            alpha_ls = alpha
            best_trial = None

            for _ in range(line_search_max_iter):
                trial_coeff = coeff1 + alpha_ls * direction
                try:
                    trial = eval_coefficients(trial_coeff, sol1.sol(grid))
                except RuntimeError:
                    trial = None

                if trial is not None:
                    trial_sol, trial_residual, trial_G_coeff, trial_l2, trial_value = trial
                    if np.isfinite(trial_value) and trial_value <= value1 * (
                        1.0 + line_search_cost_tol
                    ):
                        best_trial = (
                            trial_coeff,
                            trial_sol,
                            trial_residual,
                            trial_G_coeff,
                            trial_l2,
                            trial_value,
                        )
                        accepted = True
                        break

                alpha_ls *= line_search_beta
                if alpha_ls < alpha_min:
                    break

            if not accepted:
                return BvpBbResult(
                    value=float(value1),
                    gradient=sol1.y[2:4, 0].copy(),
                    control_coefficients=coeff1.copy(),
                    residual_l2_squared=float(G1_l2),
                    coefficient_gradient_norm=float(np.linalg.norm(G1_coeff)),
                    iterations=k,
                    converged=False,
                    message="line search failed",
                    seconds=time.perf_counter() - started,
                )

            coeff0, G0_coeff = coeff1, G1_coeff
            coeff1, sol1, G1, G1_coeff, G1_l2, value1 = best_trial
            k += 1

        return BvpBbResult(
            value=float(value1),
            gradient=sol1.y[2:4, 0].copy(),
            control_coefficients=coeff1.copy(),
            residual_l2_squared=float(G1_l2),
            coefficient_gradient_norm=float(np.linalg.norm(G1_coeff)),
            iterations=k,
            converged=bool(G1_l2 < tol),
            message="converged" if G1_l2 < tol else "maximum iterations reached",
            seconds=time.perf_counter() - started,
        )
    except RuntimeError as exc:
        return BvpBbResult(
            value=np.nan,
            gradient=np.full(2, np.nan),
            control_coefficients=np.full(num_basis, np.nan),
            residual_l2_squared=np.inf,
            coefficient_gradient_norm=np.inf,
            iterations=0,
            converged=False,
            message=f"BVP failed: {exc}",
            seconds=time.perf_counter() - started,
        )


def parse_point(raw: str) -> np.ndarray:
    values = [float(part) for part in raw.split(",")]
    if len(values) != 2:
        raise argparse.ArgumentTypeError("points must have form x1,x2")
    return np.array(values, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", nargs="+", type=parse_point, default=[np.array([0.5, 0.0])])
    parser.add_argument("--fixed-target-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--T-final", type=float, default=3.0)
    parser.add_argument("--grid-size", type=int, default=151)
    parser.add_argument("--num-basis", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--bvp-max-iter", type=int, default=50)
    parser.add_argument("--fb-max-iter", type=int, default=100)
    parser.add_argument("--bvp-tol", type=float, default=1e-6)
    parser.add_argument("--ivp-rtol", type=float, default=1e-7)
    parser.add_argument("--ivp-atol", type=float, default=1e-9)
    parser.add_argument("--max-nodes", type=int, default=10000)
    args = parser.parse_args()

    problem = VdpPhaseCaptureProblem(T_final=args.T_final)
    grid = np.linspace(0.0, args.T_final, args.grid_size)

    print("Comparing p(0)=nabla V samples")
    print(f"fixed_target_sign = {args.fixed_target_sign}")
    print(f"T = {args.T_final}, grid_size = {args.grid_size}, num_basis = {args.num_basis}")
    print()

    for point in args.points:
        print(f"x = {point}")
        bvp = solve_with_bvp_bb(
            problem=problem,
            initial_state=point,
            fixed_target_sign=args.fixed_target_sign,
            grid=grid,
            num_basis=args.num_basis,
            tol=args.tol,
            max_iter=args.bvp_max_iter,
            bvp_tol=args.bvp_tol,
            max_nodes=args.max_nodes,
        )

        started = time.perf_counter()
        fb_optimizer = ForwardBackwardOpenLoopOptimizer(
            problem=problem,
            initial_state=point,
            fixed_target_sign=args.fixed_target_sign,
            num_basis=args.num_basis,
            time_grid=grid,
            max_iter=args.fb_max_iter,
            coefficient_gradient_tol=args.tol,
            ivp_rtol=args.ivp_rtol,
            ivp_atol=args.ivp_atol,
        )
        fb = fb_optimizer.optimize()
        fb_seconds = time.perf_counter() - started

        value_delta = abs(bvp.value - fb.value)
        gradient_delta = np.linalg.norm(bvp.gradient - fb.gradient)

        print(
            "  BVP/BB:           "
            f"V={bvp.value:.8e}, p0={bvp.gradient}, "
            f"resL2sq={bvp.residual_l2_squared:.3e}, "
            f"coefGrad={bvp.coefficient_gradient_norm:.3e}, "
            f"it={bvp.iterations}, ok={bvp.converged}, sec={bvp.seconds:.2f}"
        )
        print(
            "  forward-backward: "
            f"V={fb.value:.8e}, p0={fb.gradient}, "
            f"coefGrad={fb.coefficient_gradient_norm:.3e}, "
            f"it={fb.iterations}, ok={fb.converged}, sec={fb_seconds:.2f}"
        )
        print(f"  |Delta V|={value_delta:.3e}, |Delta p0|={gradient_delta:.3e}")
        print(f"  BVP message: {bvp.message}")
        print(f"  FB  message: {fb.message}")
        print()


if __name__ == "__main__":
    main()
