import numpy as np

from src.OpenLoop.pendulum.nonsmooth import (
    NonsmoothCurve,
    restrict_trajectory_to_curve,
)
from src.OpenLoop.pendulum.problem import PendulumSwingUpProblem
from src.OpenLoop.pendulum.samples import ValueSamples
from src.OpenLoop.pendulum.solver import PendulumPmpSolver, PendulumPmpSolverConfig
from src.OpenLoop.pendulum.trajectories import (
    PmpTrajectory,
    backward_pmp_rhs,
    integrate_pmp_trajectory,
)


def test_problem_evaluates_hamiltonian_equations() -> None:
    problem = PendulumSwingUpProblem(control_weight=0.7)
    state = np.array([0.4, -0.7])
    costate = np.array([1.2, -0.5])
    control = 0.6
    q1, q2 = problem.state_weights

    expected_state_rhs = np.array(
        [
            state[1],
            -problem.damping_gain * state[1]
            + problem.gravity_gain * np.sin(state[0])
            + problem.control_gain * control,
        ]
    )
    expected_costate_rhs = np.array(
        [
            -2.0 * q1 * np.sin(state[0])
            - problem.gravity_gain * np.cos(state[0]) * costate[1],
            -2.0 * q2 * state[1] - costate[0] + problem.damping_gain * costate[1],
        ]
    )
    expected_running_cost = (
        q1 * (2.0 - 2.0 * np.cos(state[0]))
        + q2 * state[1] ** 2
        + problem.control_weight * control**2
    )
    expected_stationarity = (
        problem.control_gain * costate[1] + 2.0 * problem.control_weight * control
    )

    assert np.allclose(problem.dynamics(state, control), expected_state_rhs)
    assert np.allclose(problem.costate_rhs(state, costate), expected_costate_rhs)
    assert np.isclose(problem.running_cost(state, control), expected_running_cost)
    assert np.isclose(
        problem.stationarity_residual(costate, control),
        expected_stationarity,
    )


def test_boundary_state_has_requested_lqr_value() -> None:
    problem = PendulumSwingUpProblem()

    state = problem.boundary_state(0.37, epsilon=2e-4)
    value = state @ problem.local_lqr_matrix @ state

    assert np.isclose(value, 2e-4)


def test_backward_integration_reaches_value_cap() -> None:
    problem = PendulumSwingUpProblem()
    trajectory = integrate_pmp_trajectory(
        problem,
        angle=0.37,
        epsilon=2e-4,
        value_max=0.5,
        t_final=2.0,
        max_step=0.02,
        rtol=1e-10,
        atol=1e-12,
    )

    assert trajectory.success, trajectory.message
    assert trajectory.hit_value_event
    assert np.isclose(trajectory.value[0], 2e-4)
    assert np.isclose(trajectory.value[-1], 0.5)
    assert np.all(np.diff(trajectory.value) >= -1e-10)


def test_backward_rhs_matches_paper_equations() -> None:
    problem = PendulumSwingUpProblem()
    z = np.array([0.4, -0.7, 1.2, -0.5, 0.0])
    theta, omega, p1, p2, _value = z
    u = -problem.control_gain * p2 / (2.0 * problem.control_weight)
    q1, q2 = problem.state_weights

    expected = np.array(
        [
            -omega,
            -(
                -omega * problem.damping_gain
                + problem.gravity_gain * np.sin(theta)
                + problem.control_gain * u
            ),
            -(-2.0 * q1 * np.sin(theta) - problem.gravity_gain * np.cos(theta) * p2),
            -(-2.0 * q2 * omega - p1 + problem.damping_gain * p2),
            q1 * (2.0 - 2.0 * np.cos(theta))
            + q2 * omega**2
            + problem.control_weight * u**2,
        ]
    )

    assert np.allclose(backward_pmp_rhs(problem, 0.0, z), expected)


def test_optimal_control_satisfies_stationarity() -> None:
    problem = PendulumSwingUpProblem()
    costates = np.array(
        [
            [0.0, -2.0],
            [1.0, 0.5],
            [-1.5, 3.0],
        ]
    )

    controls = problem.minimizing_control(costates)

    assert np.allclose(
        controls,
        -problem.control_gain * costates[:, 1] / (2.0 * problem.control_weight),
    )
    assert np.allclose(problem.stationarity_residual(costates, controls), 0.0)


def test_hjb_residual_is_near_zero_along_pmp_trajectory() -> None:
    problem = PendulumSwingUpProblem()
    trajectory = integrate_pmp_trajectory(
        problem,
        angle=0.37,
        epsilon=2e-4,
        value_max=1.0,
        t_final=4.0,
        max_step=0.01,
        rtol=1e-10,
        atol=1e-12,
    )

    residual = problem.hjb_residual(trajectory.state, trajectory.costate)

    assert trajectory.success, trajectory.message
    assert np.max(np.abs(residual)) < 1e-7
    assert np.max(np.abs(residual - residual[0])) < 1e-10


def test_backward_pmp_value_matches_forward_ode_constrained_cost() -> None:
    problem = PendulumSwingUpProblem()
    trajectory = integrate_pmp_trajectory(
        problem,
        angle=0.37,
        epsilon=2e-4,
        value_max=1.0,
        t_final=4.0,
        max_step=0.0005,
        rtol=1e-10,
        atol=1e-12,
    )

    forward_time = trajectory.tau[-1] - trajectory.tau[::-1]
    forward_state = trajectory.state[::-1]
    forward_control = trajectory.control[::-1]
    terminal_value = problem.local_lqr_value(forward_state[-1])
    running = problem.running_cost(forward_state, forward_control)
    cost = float(np.trapezoid(running, forward_time) + terminal_value)

    assert np.isclose(cost, trajectory.value[-1], rtol=2e-6, atol=2e-8)


def test_value_samples_are_pdap_compatible_and_npz_roundtrip(tmp_path) -> None:
    samples = ValueSamples(
        x=np.array([[0.0, 1.0], [2.0, 3.0]]),
        v=np.array([4.0, 5.0]),
        dv=np.array([[6.0, 7.0], [8.0, 9.0]]),
    )

    pdap_data = samples.to_pdap_dict()
    path = samples.save_npz(tmp_path / "pendulum_samples.npz")
    loaded = ValueSamples.load_npz(path)

    assert set(pdap_data) == {"x", "v", "dv"}
    assert pdap_data["x"].dtype == np.float64
    assert np.allclose(loaded.x, samples.x)
    assert np.allclose(loaded.v, samples.v)
    assert np.allclose(loaded.dv, samples.dv)


def test_nonsmooth_curve_exposes_basin_polygon() -> None:
    curve = NonsmoothCurve(
        points=np.array(
            [
                [-np.pi, 0.0],
                [-np.pi, 1.0],
                [np.pi, 0.0],
                [np.pi, 1.0],
                [3.0 * np.pi, 0.0],
                [3.0 * np.pi, 1.0],
                [5.0 * np.pi, 0.0],
                [5.0 * np.pi, 1.0],
            ]
        ),
        value_levels=np.array([26.0, 26.25, 26.0, 26.25, 26.0, 26.25, 26.0, 26.25]),
        basin=np.array(
            [
                [-1.0, -1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, -1.0],
            ]
        ),
    )

    assert curve.basin_polygon() is not None
    assert curve.as_linestring().geom_type == "MultiLineString"


def test_nonsmooth_curve_npz_roundtrip_preserves_basin(tmp_path) -> None:
    curve = NonsmoothCurve(
        points=np.array([[0.0, 1.0], [2.0, 3.0]]),
        value_levels=np.array([26.0, 26.25]),
        basin=np.array([[-1.0, -1.0], [1.0, -1.0], [0.0, 1.0], [-1.0, -1.0]]),
    )

    loaded = NonsmoothCurve.load_npz(curve.save_npz(tmp_path / "curve.npz"))

    assert np.allclose(loaded.points, curve.points)
    assert np.allclose(loaded.value_levels, curve.value_levels)
    assert np.allclose(loaded.basin, curve.basin)


def test_branch_restriction_discards_points_after_first_basin_exit() -> None:
    trajectory = _trajectory(
        [[0.0, 0.0], [0.0, 0.5], [0.0, 1.5], [0.0, 2.0]],
        [0.0, 1.0, 2.0, 3.0],
        0,
    )
    curve = NonsmoothCurve(
        points=np.array([[-1.0, 0.0], [1.0, 0.0]]),
        value_levels=np.array([1.0, 1.0]),
        basin=np.array(
            [
                [-1.0, -1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, -1.0],
            ]
        ),
    )

    restricted, discarded = restrict_trajectory_to_curve(trajectory, curve)

    assert discarded == 2
    assert np.allclose(restricted.state, trajectory.state[:2])


def test_solver_emits_value_samples_and_diagnostics() -> None:
    solver = PendulumPmpSolver(
        config=PendulumPmpSolverConfig(
            value_max=0.5,
            t_final=2.0,
            max_step=0.02,
            num_trajectories=4,
            adaptive_sampling=False,
            contour_delta=0.25,
            periodic_copies=1,
        )
    )

    solution = solver.solve()

    assert solution.value_samples.size > 0
    assert solution.value_samples.x.shape[1] == 2
    assert solution.value_samples.dv.shape == solution.value_samples.x.shape
    assert solution.diagnostics.integrated_trajectories == 4
    assert solution.diagnostics.retained_points == solution.value_samples.size


def test_pendulum_solution_save_dataset_uses_run_folder(tmp_path) -> None:
    solver = PendulumPmpSolver(
        config=PendulumPmpSolverConfig(
            value_max=0.5,
            t_final=2.0,
            max_step=0.02,
            num_trajectories=4,
            adaptive_sampling=False,
            contour_delta=0.25,
        )
    )

    solution = solver.solve()
    paths = solution.save_dataset(tmp_path, date_tag="20260605")

    assert paths["run_dir"].name.startswith("Pendulum_20260605_")
    assert paths["data"].name == "Pendulum_pmp_value_samples_4_20260605.npz"
    assert paths["data"].parent == paths["run_dir"]
    assert paths["meta"].parent == paths["run_dir"]
    assert paths["failed"].parent == paths["run_dir"]
    loaded = ValueSamples.load_npz(paths["data"])
    assert loaded.x.shape[1] == 2
    assert paths["meta"].exists()
    assert paths["failed"].exists()


def _trajectory(
    states: list[list[float]],
    values: list[float],
    trajectory_id: int,
) -> PmpTrajectory:
    state = np.asarray(states, dtype=np.float64)
    value = np.asarray(values, dtype=np.float64)
    costate = np.zeros_like(state)
    return PmpTrajectory(
        boundary_angle=0.0,
        tau=np.arange(state.shape[0], dtype=np.float64),
        state=state,
        costate=costate,
        value=value,
        control=np.zeros(state.shape[0], dtype=np.float64),
        hamiltonian=np.zeros(state.shape[0], dtype=np.float64),
        trajectory_id=trajectory_id,
    )
