import json
import logging
import importlib.util
from pathlib import Path

from src.experiment_logging import ExperimentRun, RunRecordWriter
from src.logging_config import configure_logging


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_experiment_run_writes_completed_run_record(tmp_path):
    run = ExperimentRun(
        tmp_path,
        name="activation_search",
        run_id="relu_seed42",
        config={"activation": "relu", "seed": 42},
    )

    run.log_metrics({"h1": 0.12, "neurons": 78}, step=0)
    path = run.finish(status="completed")

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["name"] == "activation_search"
    assert record["run_id"] == "relu_seed42"
    assert record["status"] == "completed"
    assert record["config"] == {"activation": "relu", "seed": 42}
    assert record["metrics"] == [{"step": 0, "values": {"h1": 0.12, "neurons": 78}}]
    assert path == tmp_path / "relu_seed42.json"


def test_experiment_run_writes_failed_run_record_with_error(tmp_path):
    run = ExperimentRun(
        tmp_path,
        name="activation_search",
        run_id="bad_activation_seed42",
        config={"activation": "bad_activation", "seed": 42},
    )

    path = run.fail(RuntimeError("unknown activation"))

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["error"]["type"] == "RuntimeError"
    assert record["error"]["message"] == "unknown activation"


def test_experiment_run_records_artifacts(tmp_path):
    plot_path = tmp_path / "plots" / "pareto.png"
    run = ExperimentRun(tmp_path, name="activation_search", run_id="relu_seed42")

    run.add_artifact("pareto_plot", plot_path)
    path = run.finish()

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["artifacts"] == [{"name": "pareto_plot", "path": str(plot_path)}]


def test_configure_logging_writes_readable_diagnostics(tmp_path, capsys):
    log_path = tmp_path / "run.log"
    logger = configure_logging(verbose=True, log_file=log_path, level=logging.INFO)

    logger.info("run started: name=activation_search seed=42")

    captured = capsys.readouterr()
    assert "INFO run started: name=activation_search seed=42" in captured.err
    assert "INFO run started: name=activation_search seed=42" in log_path.read_text(encoding="utf-8")


def test_experiment_run_preserves_runner_summary_fields(tmp_path):
    run = ExperimentRun(
        tmp_path,
        name="activation_search",
        run_id="relu_seed42",
        config={"activation": "relu", "seed": 42},
    )

    path = run.finish(summary={"activation": "relu", "seed": 42, "best_score": 18.3})

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["activation"] == "relu"
    assert record["seed"] == 42
    assert record["best_score"] == 18.3
    assert record["status"] == "completed"
    assert record["name"] == "activation_search"


def test_run_record_writer_derives_identity_config_and_metrics(tmp_path):
    writer = RunRecordWriter(
        tmp_path,
        name="activation_search",
        id_fields=("activation", "seed"),
        config_fields=("activation", "seed", "num_iterations"),
        metric_field="per_gamma",
        metric_step_field="gamma",
    )
    summary = {
        "activation": "relu",
        "seed": 42,
        "num_iterations": 10,
        "best_score": 18.3,
        "per_gamma": [{"gamma": 0.1, "h1": 0.12, "n": 78}],
    }

    path = writer.write(summary)

    record = json.loads(path.read_text(encoding="utf-8"))
    assert path == tmp_path / "relu_seed42.json"
    assert record["run_id"] == "relu_seed42"
    assert record["best_score"] == 18.3
    assert record["config"] == {"activation": "relu", "seed": 42, "num_iterations": 10}
    assert record["metrics"] == [{"step": 0.1, "values": {"gamma": 0.1, "h1": 0.12, "n": 78}}]
    assert record["name"] == "activation_search"


def test_run_record_writer_does_not_assume_gamma_metric_steps(tmp_path):
    writer = RunRecordWriter(
        tmp_path,
        name="training_curve",
        id_fields=("model", "seed"),
        config_fields=("model", "seed"),
        metric_field="per_epoch",
        metric_step_field="epoch",
    )
    summary = {
        "model": "pdpa",
        "seed": 42,
        "per_epoch": [{"epoch": 3, "train_loss": 0.2, "val_loss": 0.25}],
    }

    path = writer.write(summary)

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["run_id"] == "pdpa_seed42"
    assert record["metrics"] == [{"step": 3, "values": {"epoch": 3, "train_loss": 0.2, "val_loss": 0.25}}]


def test_run_record_writer_preserves_runner_elapsed_time(tmp_path):
    writer = RunRecordWriter(
        tmp_path,
        name="activation_search",
        id_fields=("activation", "seed"),
        config_fields=("activation", "seed"),
    )

    path = writer.write({"activation": "relu", "seed": 42, "elapsed_s": 12.34})

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["elapsed_s"] == 12.34


def test_activation_runners_use_shared_run_record_writer():
    activation_script = load_script_module("run_activation_experiment", "scripts/run_activation_experiment.py")
    analytical_script = load_script_module(
        "run_discontinuous_activation_experiment", "scripts/run_discontinuous_activation_experiment.py",
    )
    assert activation_script.RUN_RECORD.name == "activation_search"
    assert activation_script.RUN_RECORD.run_id({"activation": "relu", "seed": 42}) == "relu_seed42"
    assert analytical_script.RUN_RECORD.name == "activation_search_analytical"
    assert analytical_script.RUN_RECORD.run_id({"activation": "relu", "seed": 42}) == "relu_seed42"


def test_pendulum_model_comparison_writes_standard_run_record(tmp_path):
    script = load_script_module("run_pendulum_model_comparison", "scripts/run_pendulum_model_comparison.py")
    summary = {
        "model": "v2",
        "activation": "relu",
        "seed": 42,
        "dataset": "pmp",
        "status": "ok",
        "best_score": 1.2,
    }

    script.write_outputs(tmp_path, summary)

    path = tmp_path / "v2_relu_seed42.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["model"] == "v2"
    assert record["dataset"] == "pmp"
    assert record["status"] == "ok"
    assert record["name"] == "pendulum_model_comparison"
    assert record["run_id"] == "v2_relu_seed42"
