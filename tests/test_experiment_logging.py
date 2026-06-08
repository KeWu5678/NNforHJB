import json
import logging

from src.experiment_logging import ExperimentRun
from src.logging_config import configure_logging


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
