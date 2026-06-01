import numpy as np

from src.OpenLoop.bvp_bb_optimizer import BvpBbOpenLoopOptimizer, BvpBbOpenLoopResult
from src.OpenLoop.pendulum.bb_generator import PendulumBBDataGenerator


def test_pendulum_bb_tpbvp_matches_hamiltonian_equations() -> None:
    generator = PendulumBBDataGenerator(terminal_weight=0.0)
    t = np.array([0.0, 0.5])
    y = np.array(
        [
            [0.4, -0.2],
            [-0.7, 0.3],
            [1.2, -0.4],
            [-0.5, 0.8],
        ]
    )

    def control(time: np.ndarray) -> np.ndarray:
        return np.array([0.6, -0.1]) + 0.0 * time

    theta, omega, p1, p2 = y
    u = control(t)
    q1, q2 = generator.state_weights
    expected = np.vstack(
        [
            omega,
            -generator.damping_gain * omega
            + generator.gravity_gain * np.sin(theta)
            + generator.control_gain * u,
            -2.0 * q1 * np.sin(theta)
            - generator.gravity_gain * np.cos(theta) * p2,
            -2.0 * q2 * omega - p1 + generator.damping_gain * p2,
        ]
    )

    assert np.allclose(generator.pendulum_tpbvp(t, y, control), expected)


def test_pendulum_bb_terminal_gradient_matches_lqr_tail_finite_difference() -> None:
    generator = PendulumBBDataGenerator(terminal_weight=1.3)
    state = np.array([0.2, -0.4])
    direction = np.array([0.7, -0.3])
    eps = 1e-6

    finite_difference = (
        generator.terminal_cost(state + eps * direction)
        - generator.terminal_cost(state - eps * direction)
    ) / (2.0 * eps)

    assert np.isclose(
        finite_difference,
        generator.terminal_gradient(state) @ direction,
        rtol=1e-6,
        atol=1e-8,
    )


def test_pendulum_bb_stationarity_residual_vanishes_for_feedback() -> None:
    generator = PendulumBBDataGenerator(control_weight=0.7)
    gradient = np.array([1.2, -0.5])
    control = generator.feedback_from_gradient(gradient)

    residual = generator.control_gain * gradient[1] + 2.0 * generator.control_weight * control

    assert np.isclose(residual, 0.0)


def test_pendulum_bb_solves_one_near_equilibrium_sample() -> None:
    generator = PendulumBBDataGenerator(
        T_final=0.2,
        dt=0.1,
        num_basis=3,
        control_seed_amplitudes=(0.0,),
        bvp_tol=1e-5,
    )

    best, attempts = generator.solve_initial_state(
        np.array([0.01, 0.0]),
        tol=1e3,
        max_it=2,
        verbose=False,
    )

    assert attempts
    assert best is not None
    assert best.converged
    assert best.gradient.shape == (2,)
    assert np.isfinite(best.value)
    assert best.value >= 0.0


def test_bvp_bb_optimizer_exposes_structured_result() -> None:
    generator = PendulumBBDataGenerator(
        T_final=0.2,
        dt=0.1,
        num_basis=3,
        bvp_tol=1e-5,
    )
    initial_state = np.array([0.01, 0.0])
    grid = generator.time_grid()
    guess = generator.initial_guess(initial_state, grid)

    optimizer = BvpBbOpenLoopOptimizer(
        generator.pendulum_tpbvp,
        generator.gen_bc(initial_state),
        generator.V,
        generator.gradient,
        grid,
        guess,
        tol=1e3,
        max_it=2,
        num_basis=generator.num_basis,
        domain=(0.0, generator.T_final),
        initial_coefficients=np.zeros(generator.num_basis),
        verbose=False,
    )

    result = optimizer.optimize_result()

    assert isinstance(result, BvpBbOpenLoopResult)
    assert result.converged
    assert result.gradient.shape == (2,)
    assert np.isfinite(result.value)
    assert result.as_legacy_tuple()[0].shape == (2,)
