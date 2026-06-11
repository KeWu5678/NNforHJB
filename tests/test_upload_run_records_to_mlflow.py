import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def load_upload_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "upload_run_records_to_mlflow.py"
    spec = importlib.util.spec_from_file_location("upload_run_records_to_mlflow", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fake_mlflow(monkeypatch):
    calls = {
        "tracking_uri": None,
        "experiment": None,
        "run_names": [],
        "params": {},
        "metrics": [],
        "tags": {},
        "ended": [],
    }

    def set_tracking_uri(uri):
        calls["tracking_uri"] = uri

    def set_experiment(name):
        calls["experiment"] = name

    def start_run(*, run_name):
        calls["run_names"].append(run_name)

    def log_param(key, value):
        calls["params"][key] = value

    def log_metric(key, value, step=None):
        calls["metrics"].append((key, value, step))

    def set_tag(key, value):
        calls["tags"][key] = value

    def end_run(*, status):
        calls["ended"].append(status)

    monkeypatch.setitem(
        sys.modules,
        "mlflow",
        SimpleNamespace(
            set_tracking_uri=set_tracking_uri,
            set_experiment=set_experiment,
            start_run=start_run,
            log_param=log_param,
            log_metric=log_metric,
            set_tag=set_tag,
            end_run=end_run,
        ),
    )
    return calls


def write_run_record(run_dir: Path) -> Path:
    record = {
        "name": "activationsearch",
        "run_id": "signed_profile_relu_power1p0_gamma0p0_h1_seed42",
        "status": "completed",
        "config": {
            "model": {"activation": "relu", "gamma": 0.0},
            "data": {"path": "VDP_beta_0.1_grid_30x30.npy"},
        },
        "metrics": [{"values": {"rel_h1_val": 0.12, "best_neurons": 58}}],
        "artifacts": [{"name": "fit_history", "path": str(run_dir / "result.pkl")}],
        "started_at": "2026-06-10T12:00:00+00:00",
        "ended_at": "2026-06-10T12:01:00+00:00",
        "elapsed_s": 60.0,
    }
    path = run_dir / "signed_profile_relu_power1p0_gamma0p0_h1_seed42.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_discover_records_skips_hydra_directory(tmp_path):
    upload = load_upload_module()
    run_dir = tmp_path / "rawdata" / "logs" / "multirun" / "activationsearch" / "0"
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir(parents=True)
    record_path = write_run_record(run_dir)
    (hydra_dir / "hydra.json").write_text("{}", encoding="utf-8")

    assert upload.discover_records([tmp_path]) == [record_path]


def _set_mtime(path: Path, when: float) -> None:
    os.utime(path, (when, when))


def test_latest_run_keeps_records_after_the_multirun_marker(tmp_path):
    upload = load_upload_module()
    sweep_root = tmp_path / "rawdata" / "logs" / "multirun" / "activationsearch"
    stale_dir = sweep_root / "0"
    fresh_dir = sweep_root / "1"
    stale_dir.mkdir(parents=True)
    fresh_dir.mkdir(parents=True)
    stale = write_run_record(stale_dir)
    fresh = write_run_record(fresh_dir)
    marker = sweep_root / "multirun.yaml"
    marker.write_text("{}", encoding="utf-8")

    launch = 1_800_000_000
    _set_mtime(stale, launch - 100)   # leftover from an earlier sweep
    _set_mtime(marker, launch)        # latest sweep launched here
    _set_mtime(fresh, launch + 100)   # produced by the latest sweep

    assert upload.discover_records([sweep_root], latest_run=True) == [fresh]


def test_latest_run_picks_the_newest_multirun_marker(tmp_path):
    upload = load_upload_module()
    root = tmp_path / "rawdata" / "logs" / "multirun"
    older_sweep = root / "activationsearch"
    newer_sweep = root / "penaltypowers"
    (older_sweep / "0").mkdir(parents=True)
    (newer_sweep / "0").mkdir(parents=True)
    older_record = write_run_record(older_sweep / "0")
    newer_record = write_run_record(newer_sweep / "0")
    older_marker = older_sweep / "multirun.yaml"
    newer_marker = newer_sweep / "multirun.yaml"
    older_marker.write_text("{}", encoding="utf-8")
    newer_marker.write_text("{}", encoding="utf-8")

    _set_mtime(older_marker, 1_700_000_000)
    _set_mtime(older_record, 1_700_000_100)
    _set_mtime(newer_marker, 1_800_000_000)
    _set_mtime(newer_record, 1_800_000_100)

    assert upload.discover_records([root], latest_run=True) == [newer_record]


def test_latest_run_falls_back_when_no_marker(tmp_path):
    upload = load_upload_module()
    run_dir = tmp_path / "rawdata" / "logs" / "multirun" / "activationsearch" / "0"
    run_dir.mkdir(parents=True)
    record = write_run_record(run_dir)

    assert upload.discover_records([run_dir.parent.parent], latest_run=True) == [record]


def test_load_record_adds_hydra_overrides_from_adjacent_directory(tmp_path):
    upload = load_upload_module()
    run_dir = tmp_path / "rawdata" / "logs" / "multirun" / "activationsearch" / "0"
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir(parents=True)
    record_path = write_run_record(run_dir)
    (hydra_dir / "overrides.yaml").write_text(
        "- +experiment=activationsearch\n- data=vdp\n- model.activation=relu\n",
        encoding="utf-8",
    )

    record = upload.load_record(record_path)

    assert record["hydra"]["output_dir"] == str(run_dir)
    assert record["hydra"]["runtime"]["choices"] == {
        "experiment": "activationsearch",
        "data": "vdp",
    }
    assert record["hydra"]["overrides"]["task"] == [
        "+experiment=activationsearch",
        "data=vdp",
        "model.activation=relu",
    ]


def test_existing_record_can_be_uploaded_to_mlflow(tmp_path, monkeypatch):
    upload = load_upload_module()
    calls = fake_mlflow(monkeypatch)
    run_dir = tmp_path / "rawdata" / "logs" / "multirun" / "activationsearch" / "0"
    (run_dir / ".hydra").mkdir(parents=True)
    record_path = write_run_record(run_dir)

    record = upload.load_record(record_path)
    upload.publish_record_to_mlflow(record, record_path, tracking_uri="http://localhost:5000")

    assert calls["tracking_uri"] == "http://localhost:5000"
    assert calls["experiment"] == "activationsearch"
    assert calls["run_names"] == ["signed_profile_relu_power1p0_gamma0p0_h1_seed42"]
    assert calls["params"]["model.activation"] == "relu"
    assert ("rel_h1_val", 0.12, None) in calls["metrics"]
    assert calls["tags"]["run_record.path"] == str(record_path)
    assert calls["ended"] == ["FINISHED"]
