import numpy as np

from src.OpenLoop.value_samples import ValueSamples
from src.OpenLoop.vdp import (
    VdpOpenLoopSolver,
    VdpOpenLoopSolverConfig,
    VdpOptimalControlProblem,
    grid_initial_states,
)


def test_vdp_problem_matches_reduced_gradient_equations() -> None:
    problem = VdpOptimalControlProblem(beta=0.1, mu=1.0)
    y = np.array([0.5, -0.25])
    p = np.array([0.2, -0.4])

    dynamics = problem.dynamics(0.0, y, u=0.3)
    adjoint = problem.adjoint_rhs(0.0, y, p)
    residual = problem.reduced_gradient(p.reshape(2, 1), np.array([0.3]))

    assert np.allclose(dynamics, [-0.25, -0.3875])
    assert np.allclose(adjoint, [-1.3, 0.6])
    assert np.allclose(residual, [-0.34])


def test_paper_profile_solves_zero_initial_state() -> None:
    problem = VdpOptimalControlProblem(T_final=0.1)
    config = VdpOpenLoopSolverConfig(
        profile="paper",
        time_step=0.05,
        convergence_tol=1e-8,
        max_iter=3,
        store_trajectories=True,
    )
    solver = VdpOpenLoopSolver(problem=problem, config=config)

    result = solver.solve_sample(np.array([0.0, 0.0]))

    assert result.converged
    assert result.value == 0.0
    assert np.allclose(result.gradient, [0.0, 0.0])
    assert result.reduced_gradient_norm == 0.0
    assert result.state_trajectory.shape == (2, 3)
    assert result.control.shape == (3,)


def test_paper_profile_returns_pdap_value_samples_and_save_artifacts(tmp_path) -> None:
    problem = VdpOptimalControlProblem(T_final=0.1)
    config = VdpOpenLoopSolverConfig(
        profile="paper",
        time_step=0.05,
        convergence_tol=1e-8,
        max_iter=3,
    )
    solver = VdpOpenLoopSolver(problem=problem, config=config)

    solution = solver.solve(np.array([[0.0, 0.0], [0.1, 0.0]]))
    paths = solution.save_dataset(tmp_path, grid_shape=(1, 2), date_tag="20260605")

    assert solution.value_samples.size >= 1
    assert solution.failed_initial_states.shape[1] == 2
    assert paths["data"].name == "VDP_paper_grid_1x2_20260605.npz"
    loaded = ValueSamples.load_npz(paths["data"])
    assert loaded.x.shape[1] == 2
    assert loaded.v.ndim == 1
    assert loaded.dv.shape == loaded.x.shape
    assert paths["meta"].exists()
    assert paths["failed"].exists()


def test_fast_profile_solves_zero_initial_state() -> None:
    problem = VdpOptimalControlProblem(T_final=0.1)
    config = VdpOpenLoopSolverConfig(
        profile="fast",
        num_time_points=11,
        convergence_tol=1e-8,
        max_iter=3,
    )
    solver = VdpOpenLoopSolver(problem=problem, config=config)

    result = solver.solve_sample(np.array([0.0, 0.0]))

    assert result.converged
    assert result.value == 0.0
    assert np.allclose(result.gradient, [0.0, 0.0])
    assert result.reduced_gradient_norm == 0.0
    assert result.coefficient_gradient_norm == 0.0


def test_vdp_grid_sampling_feeds_solver() -> None:
    initial_states = grid_initial_states(1, 2, (0.0, 0.0), (0.0, 0.1))
    problem = VdpOptimalControlProblem(T_final=0.1)
    config = VdpOpenLoopSolverConfig(profile="fast", num_time_points=11, max_iter=1)
    solver = VdpOpenLoopSolver(problem=problem, config=config)

    solution = solver.solve(initial_states)

    assert len(solution.sample_results) == 2
    assert solution.value_samples.x.shape[1] == 2
