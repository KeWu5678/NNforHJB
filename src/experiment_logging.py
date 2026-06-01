"""Experiment run records and progress helpers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExperimentRun:
    """Durable JSON record for one experiment run."""

    output_dir: str | Path
    name: str
    run_id: str
    config: dict[str, Any] = field(default_factory=dict)

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
        path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
        return path
