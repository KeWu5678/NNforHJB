import numpy as np
import subprocess
import sys


def test_openloop_value_samples_public_import_is_available() -> None:
    from src.OpenLoop import ValueSamples

    samples = ValueSamples(
        x=np.zeros((1, 2)),
        v=np.zeros(1),
        dv=np.zeros((1, 2)),
    )

    assert samples.size == 1
    assert samples.to_pdap_dict()["x"].dtype == np.float64


def test_vdp_sampling_helpers_create_grid_and_random_points() -> None:
    from src.OpenLoop.vdp import grid_initial_states, random_initial_states

    grid = grid_initial_states(2, 3, (-1.0, 1.0), (-2.0, 2.0))
    random_a = random_initial_states(4, (-1.0, 1.0), (-2.0, 2.0), seed=7)
    random_b = random_initial_states(4, (-1.0, 1.0), (-2.0, 2.0), seed=7)

    assert grid.shape == (6, 2)
    assert np.allclose(grid[0], [-1.0, -2.0])
    assert np.allclose(grid[-1], [1.0, 2.0])
    assert random_a.shape == (4, 2)
    assert np.allclose(random_a, random_b)


def test_pendulum_solver_public_imports_are_available() -> None:
    from src.OpenLoop.pendulum import (
        PendulumPmpSolver,
        PendulumPmpSolverConfig,
        PendulumSwingUpProblem,
        ValueSamples,
    )

    problem = PendulumSwingUpProblem(control_limit=2.0)
    config = PendulumPmpSolverConfig(value_max=0.5, t_final=1.0)
    solver = PendulumPmpSolver(problem=problem, config=config)

    assert problem.control_limit == 2.0
    assert solver.config.value_max == 0.5
    assert ValueSamples.__name__ == "ValueSamples"


def test_value_samples_save_npz_uses_explicit_output_path(tmp_path) -> None:
    from src.OpenLoop import ValueSamples

    samples = ValueSamples(
        x=np.zeros((1, 2)),
        v=np.zeros(1),
        dv=np.zeros((1, 2)),
    )

    path = samples.save_npz(tmp_path / "value_samples.npz")

    assert path == tmp_path / "value_samples.npz"
    assert path.exists()
    with np.load(path) as data:
        assert set(data.files) == {"x", "v", "dv"}


def test_pendulum_pmp_dataset_script_has_help() -> None:
    scripts = ["scripts/run_pendulum_pmp_openloop_example.py"]

    for script in scripts:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert "--output-dir" in result.stdout


def test_vdp_solver_public_imports_are_available() -> None:
    from src.OpenLoop.vdp import (
        VdpOpenLoopSolver,
        VdpOpenLoopSolverConfig,
        VdpOptimalControlProblem,
    )

    problem = VdpOptimalControlProblem(T_final=0.1)
    config = VdpOpenLoopSolverConfig(profile="fast", num_time_points=11, max_iter=1)
    solver = VdpOpenLoopSolver(problem=problem, config=config)

    assert solver.problem.T_final == 0.1
    assert solver.config.profile == "fast"
