"""Experiment run records and progress helpers."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExperimentRun:
    """Runtime accumulator for one experiment run.

    The local JSON Run Record is the source of truth. When MLFLOW_TRACKING_URI is
    set, the completed record is also projected to MLflow for dashboard views.
    """

    output_dir: str | Path
    name: str
    run_id: str
    config: dict[str, Any] = field(default_factory=dict)
    hydra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.started_at = _utc_now()
        self._t0 = time.monotonic()
        self._metrics: list[dict[str, Any]] = []
        self._artifacts: list[dict[str, str]] = []

    def log_metrics(self, values: dict[str, Any], *, step: int | float | str | None = None) -> None:
        event: dict[str, Any] = {"values": dict(values)}
        if step is not None:
            event["step"] = step
        self._metrics.append(event)

    def add_artifact(self, name: str, path: str | Path) -> None:
        self._artifacts.append({"name": name, "path": str(path)})

    def finish(
        self,
        *,
        status: str = "completed",
        summary: dict[str, Any] | None = None,
    ) -> Path:
        return self._write_record(status=status, summary=summary)

    def fail(self, error: BaseException) -> Path:
        return self._write_record(
            status="failed",
            error={"type": type(error).__name__, "message": str(error)},
        )

    def _write_record(
        self,
        *,
        status: str,
        summary: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{self.run_id}.json"
        record = dict(summary or {})
        record.update({
            "name": self.name,
            "run_id": self.run_id,
            "status": status,
            "config": self.config,
            "metrics": self._metrics,
            "artifacts": self._artifacts,
            "started_at": self.started_at,
            "ended_at": _utc_now(),
            "elapsed_s": round(time.monotonic() - self._t0, 3),
        })
        if error is not None:
            record["error"] = error
        if self.hydra:
            record["hydra"] = self.hydra
        if summary is not None and "elapsed_s" in summary:
            record["elapsed_s"] = summary["elapsed_s"]
        path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
        publish_record_to_mlflow(record, path)
        return path


def publish_record_to_mlflow(
    record: dict[str, Any],
    record_path: str | Path,
    *,
    tracking_uri: str | None = None,
) -> bool:
    """Project a completed Run Record to MLflow when a tracking URI is available."""

    record_path = Path(record_path)
    tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return False

    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(str(record["name"]))
    mlflow.start_run(run_name=str(record["run_id"]))
    try:
        for key, value in _flatten(record.get("config", {})):
            mlflow.log_param(key, _format_param(value))
        for key, value in _hydra_params(record.get("hydra", {})):
            mlflow.log_param(key, _format_param(value))

        for key, value, step in _metric_events(record):
            if step is None:
                mlflow.log_metric(key, float(value))
            else:
                mlflow.log_metric(key, float(value), step=step)
        for key, value in _summary_metrics(record):
            mlflow.log_metric(key, float(value))

        mlflow.set_tag("run_id", str(record["run_id"]))
        mlflow.set_tag("status", str(record["status"]))
        mlflow.set_tag("run_record.path", str(record_path))
        for key, value in _hydra_tags(record.get("hydra", {})):
            mlflow.set_tag(key, str(value))
        for artifact in record.get("artifacts", []):
            name = artifact.get("name", "artifact")
            path = artifact.get("path")
            if path:
                mlflow.set_tag(f"artifact.{name}.path", str(path))
        if "error" in record:
            error = record["error"]
            mlflow.set_tag("error.type", str(error.get("type", "")))
            mlflow.set_tag("error.message", str(error.get("message", "")))
    finally:
        status = "FAILED" if record.get("status") == "failed" else "FINISHED"
        mlflow.end_run(status=status)
    return True


def _flatten(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        items: list[tuple[str, Any]] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.extend(_flatten(child, child_prefix))
        return items
    return [(prefix, value)]


def _format_param(value: Any) -> str | int | float | bool:
    if isinstance(value, str | int | float | bool):
        return value
    return json.dumps(value, default=str)


def _is_metric_value(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _metric_events(record: dict[str, Any]) -> list[tuple[str, Real, int | None]]:
    metrics: list[tuple[str, Real, int | None]] = []
    for event in record.get("metrics", []):
        raw_step = event.get("step")
        step = raw_step if isinstance(raw_step, int) and not isinstance(raw_step, bool) else None
        for key, value in event.get("values", {}).items():
            if _is_metric_value(value):
                metrics.append((str(key), value, step))
    return metrics


def _summary_metrics(record: dict[str, Any]) -> list[tuple[str, Real]]:
    reserved = {
        "name",
        "run_id",
        "status",
        "config",
        "metrics",
        "artifacts",
        "started_at",
        "ended_at",
        "elapsed_s",
        "error",
        "hydra",
    }
    return [
        (key, value)
        for key, value in record.items()
        if key not in reserved and _is_metric_value(value)
    ]


def _hydra_params(hydra: dict[str, Any]) -> list[tuple[str, Any]]:
    choices = hydra.get("runtime", {}).get("choices", {})
    return [(f"hydra.choice.{key}", value) for key, value in choices.items()]


def _hydra_tags(hydra: dict[str, Any]) -> list[tuple[str, Any]]:
    tags: list[tuple[str, Any]] = []
    if output_dir := hydra.get("output_dir"):
        tags.append(("hydra.output_dir", output_dir))
    job = hydra.get("job", {})
    for key in ("name", "id", "num"):
        if job.get(key) is not None:
            tags.append((f"hydra.job.{key}", job[key]))
    task_overrides = hydra.get("overrides", {}).get("task")
    if task_overrides:
        tags.append(("hydra.overrides.task", json.dumps(task_overrides, default=str)))
    return tags
