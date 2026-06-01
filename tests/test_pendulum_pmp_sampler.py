import numpy as np

from src.OpenLoop.pendulum.swingup_dynamics import PendulumSwingUpDynamics
from src.OpenLoop.pendulum.pmp_sampler import PendulumPmpParameters, PendulumPmpSampler


def test_swingup_dynamics_evaluates_shared_hamiltonian_equations() -> None:
    dynamics = PendulumSwingUpDynamics(control_weight=0.7)
    state = np.array([0.4, -0.7])
    costate = np.array([1.2, -0.5])
    control = 0.6
    q1, q2 = dynamics.state_weights

    expected_state_rhs = np.array(
        [
            state[1],
            -dynamics.damping_gain * state[1]
            + dynamics.gravity_gain * np.sin(state[0])
            + dynamics.control_gain * control,
        ]
    )
    expected_costate_rhs = np.array(
        [
            -2.0 * q1 * np.sin(state[0])
            - dynamics.gravity_gain * np.cos(state[0]) * costate[1],
            -2.0 * q2 * state[1] - costate[0] + dynamics.damping_gain * costate[1],
        ]
    )
    expected_running_cost = (
        q1 * (2.0 - 2.0 * np.cos(state[0]))
        + q2 * state[1] ** 2
        + dynamics.control_weight * control**2
    )
    expected_stationarity = (
        dynamics.control_gain * costate[1] + 2.0 * dynamics.control_weight * control
    )

    assert np.allclose(dynamics.forward_dynamics(state, control), expected_state_rhs)
    assert np.allclose(dynamics.costate_rhs(state, costate), expected_costate_rhs)
    assert np.isclose(dynamics.running_cost(state, control), expected_running_cost)
    assert np.isclose(
        dynamics.stationarity_residual(costate, control),
        expected_stationarity,
    )


def test_boundary_state_has_requested_lqr_value() -> None:
    sampler = PendulumPmpSampler()

    state = sampler.boundary_state(0.37)
    value = state @ sampler.local_lqr_matrix @ state

    assert np.isclose(value, sampler.parameters.epsilon)


def test_backward_integration_reaches_value_cap() -> None:
    parameters = PendulumPmpParameters(value_max=0.5, t_final=2.0, max_step=0.02)
    sampler = PendulumPmpSampler(parameters)

    trajectory = sampler.integrate_angle(0.37)

    assert trajectory.success, trajectory.message
    assert trajectory.hit_value_event
    assert np.isclose(trajectory.value[0], parameters.epsilon)
    assert np.isclose(trajectory.value[-1], parameters.value_max)
    assert np.all(np.diff(trajectory.value) >= -1e-10)


def test_backward_rhs_matches_released_matlab_equations() -> None:
    sampler = PendulumPmpSampler()
    z = np.array([0.4, -0.7, 1.2, -0.5, 0.0])
    theta, omega, p1, p2, _value = z
    u = -sampler.control_gain * p2 / (2.0 * sampler.parameters.control_weight)
    q1, q2 = sampler.state_weights

    expected = np.array(
        [
            -omega,
            -(
                -omega * sampler.damping_gain
                + sampler.gravity_gain * np.sin(theta)
                + sampler.control_gain * u
            ),
            -(-2.0 * q1 * np.sin(theta) - sampler.gravity_gain * np.cos(theta) * p2),
            -(-2.0 * q2 * omega - p1 + sampler.damping_gain * p2),
            q1 * (2.0 - 2.0 * np.cos(theta))
            + q2 * omega**2
            + sampler.parameters.control_weight * u**2,
        ]
    )

    assert np.allclose(sampler.backward_pmp_rhs(0.0, z), expected)


def test_optimal_control_satisfies_hjb_stationarity() -> None:
    sampler = PendulumPmpSampler()
    costates = np.array(
        [
            [0.0, -2.0],
            [1.0, 0.5],
            [-1.5, 3.0],
        ]
    )

    controls = sampler.optimal_control(costates)

    assert np.allclose(
        controls,
        -sampler.control_gain * costates[:, 1]
        / (2.0 * sampler.parameters.control_weight),
    )
    assert np.allclose(sampler.stationarity_residual(costates, controls), 0.0)


def test_hjb_residual_is_near_zero_along_pmp_trajectories() -> None:
    parameters = PendulumPmpParameters(value_max=1.0, t_final=4.0, max_step=0.01)
    sampler = PendulumPmpSampler(parameters)

    for angle in (0.0, 0.37, 1.0, 3.0):
        trajectory = sampler.integrate_angle(angle)
        residual = sampler.hjb_residual(trajectory.state, trajectory.costate)

        assert trajectory.success, trajectory.message
        assert np.max(np.abs(residual)) < 1e-7
        assert np.max(np.abs(residual - residual[0])) < 1e-10


def test_backward_pmp_value_matches_forward_ode_constrained_cost() -> None:
    parameters = PendulumPmpParameters(value_max=1.0, t_final=4.0, max_step=0.0005)
    sampler = PendulumPmpSampler(parameters)
    trajectory = sampler.integrate_angle(0.37)

    forward_time = trajectory.tau[-1] - trajectory.tau[::-1]
    forward_state = trajectory.state[::-1]
    forward_control = trajectory.control[::-1]
    terminal_value = sampler.local_lqr_value(forward_state[-1])
    cost = sampler.trajectory_cost(
        forward_time,
        forward_state,
        forward_control,
        terminal_value=terminal_value,
    )

    assert np.isclose(cost, trajectory.value[-1], rtol=2e-6, atol=2e-8)


def test_dataset_uses_training_fields_and_periodic_copies() -> None:
    parameters = PendulumPmpParameters(value_max=0.5, t_final=2.0, max_step=0.02)
    sampler = PendulumPmpSampler(parameters)
    trajectory = sampler.integrate_angle(0.37, trajectory_id=7)

    data = sampler.trajectories_to_dataset([trajectory], periodic_copies=1)

    assert {"x", "dv", "v"}.issubset(data.dtype.names)
    assert set(np.unique(data["periodic_copy"])) == {-1, 0, 1}
    assert np.all(data["trajectory_id"] == 7)

    n = trajectory.tau.size
    assert data.shape[0] == 3 * n
    assert np.allclose(data[:n]["x"][:, 0], trajectory.state[:, 0] - 2.0 * np.pi)
    assert np.allclose(data[n : 2 * n]["x"][:, 0], trajectory.state[:, 0])
    assert np.allclose(data[2 * n :]["x"][:, 0], trajectory.state[:, 0] + 2.0 * np.pi)
    assert np.allclose(data["dv"][:n], trajectory.costate)
