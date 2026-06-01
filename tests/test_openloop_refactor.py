import numpy as np
import subprocess
import sys


def test_forward_backward_public_imports_are_available() -> None:
    from src.OpenLoop.forward_backward_optimizer import (
        ForwardBackwardOpenLoopOptimizer,
        ForwardBackwardOpenLoopResult,
        ForwardBackwardOpenLoopProblem,
    )
    from src.OpenLoop.pendulum.finite_horizon_problem import PendulumSwingUpProblem

    problem = PendulumSwingUpProblem(T_final=0.1)

    assert problem.T_final == 0.1
    assert ForwardBackwardOpenLoopOptimizer.__name__ == "ForwardBackwardOpenLoopOptimizer"
    assert ForwardBackwardOpenLoopResult.__name__ == "ForwardBackwardOpenLoopResult"
    assert ForwardBackwardOpenLoopProblem.__name__ == "ForwardBackwardOpenLoopProblem"


def test_sample_set_helpers_create_grid_random_and_saved_files(tmp_path) -> None:
    from src.OpenLoop.sample_sets import (
        grid_initial_states,
        random_initial_states,
        save_dataset_bundle,
    )

    grid = grid_initial_states(2, 3, (-1.0, 1.0), (-2.0, 2.0))
    random_a = random_initial_states(4, (-1.0, 1.0), (-2.0, 2.0), seed=7)
    random_b = random_initial_states(4, (-1.0, 1.0), (-2.0, 2.0), seed=7)

    assert grid.shape == (6, 2)
    assert np.allclose(grid[0], [-1.0, -2.0])
    assert np.allclose(grid[-1], [1.0, 2.0])
    assert random_a.shape == (4, 2)
    assert np.allclose(random_a, random_b)

    saved = save_dataset_bundle(
        output_dir=tmp_path,
        arrays={"sample.npy": grid},
        json_files={"meta.json": {"n": int(grid.shape[0])}},
    )

    assert saved["sample.npy"] == tmp_path / "sample.npy"
    assert saved["meta.json"] == tmp_path / "meta.json"
    assert np.load(saved["sample.npy"]).shape == (6, 2)


def test_finite_horizon_generator_is_available() -> None:
    from src.OpenLoop.pendulum.finite_horizon_generator import (
        PendulumFiniteHorizonDataGenerator,
    )

    generator = PendulumFiniteHorizonDataGenerator(T_final=0.1, time_steps=3)

    assert generator.time_grid().shape == (3,)


def test_vdp_data_save_uses_explicit_output_dir(tmp_path) -> None:
    from src.OpenLoop.vdp.generator import DataGenerator

    dataset = np.zeros(
        1,
        dtype=[("x", "2float64"), ("dv", "2float64"), ("v", "float64")],
    )

    generator = DataGenerator()
    data_path, failed_path = generator.data_save(
        dataset,
        failed_ini=[],
        output_dir=tmp_path,
        tag="unit",
    )

    assert data_path == tmp_path / "VDP_dataset_unit.npy"
    assert failed_path is None
    assert data_path.exists()


def test_pendulum_bb_data_save_uses_explicit_output_dir(tmp_path) -> None:
    from src.OpenLoop.pendulum.bb_generator import PENDULUM_BB_DTYPE, PendulumBBDataGenerator

    dataset = np.zeros(1, dtype=PENDULUM_BB_DTYPE)

    generator = PendulumBBDataGenerator()
    data_path, failed_path = generator.data_save(
        dataset,
        failed=[],
        output_dir=tmp_path,
        tag="unit",
    )

    assert data_path == tmp_path / "PENDULUM_bb_openloop_unit.npy"
    assert failed_path is None
    assert data_path.exists()


def test_pendulum_finite_horizon_data_save_uses_explicit_output_dir(tmp_path) -> None:
    from src.OpenLoop.pendulum.finite_horizon_generator import (
        PENDULUM_FINITE_HORIZON_DTYPE,
        PendulumFiniteHorizonDataGenerator,
    )

    dataset = np.zeros(1, dtype=PENDULUM_FINITE_HORIZON_DTYPE)

    generator = PendulumFiniteHorizonDataGenerator()
    data_path, failed_path, diagnostics_path = generator.data_save(
        dataset,
        failed=[],
        diagnostics=[],
        output_dir=tmp_path,
        tag="unit",
    )

    assert data_path == tmp_path / "PENDULUM_transient_openloop_unit.npy"
    assert failed_path == tmp_path / "PENDULUM_transient_openloop_failed_unit.json"
    assert diagnostics_path == tmp_path / "PENDULUM_transient_openloop_diagnostics_unit.json"
    assert data_path.exists()
    assert failed_path.exists()
    assert diagnostics_path.exists()


def test_finite_horizon_and_transient_dataset_scripts_have_help() -> None:
    scripts = [
        "scripts/run_pendulum_finite_horizon_openloop_dataset.py",
        "scripts/run_pendulum_transient_openloop_dataset.py",
    ]

    for script in scripts:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert "--save" in result.stdout


def test_vdp_phase_capture_public_imports_are_available() -> None:
    from src.OpenLoop.vdp.phase_capture import (
        VdpPhaseCaptureProblem,
        solve_phase_capture_sample,
    )

    problem = VdpPhaseCaptureProblem(T_final=0.1)

    assert problem.T_final == 0.1
    assert callable(solve_phase_capture_sample)
