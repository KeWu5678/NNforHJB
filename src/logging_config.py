"""Diagnostic logging configuration for scripts and notebooks."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


DIAGNOSTIC_FORMAT = "%(levelname)s %(message)s"


def configure_logging(
    *,
    verbose: bool = True,
    log_file: str | Path | None = None,
    level: int | str = logging.INFO,
) -> logging.Logger:
    """Configure concise diagnostic logging and return the project logger."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    formatter = logging.Formatter(DIAGNOSTIC_FORMAT)
    if verbose:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(formatter)
        console.setLevel(level)
        root.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root.addHandler(file_handler)

    return logging.getLogger("nnforhjb")


def setup_logging(
    verbose: bool = True,
    log_file: str | Path | None = None,
    log_level: int | str = logging.INFO,
    rotation: str | None = None,
) -> logging.Logger:
    """Backward-compatible alias for older callers.

    ``rotation`` is accepted for old Loguru-style calls but is not used by the
    standard-library implementation.
    """
    del rotation
    return configure_logging(verbose=verbose, log_file=log_file, level=log_level)


def get_logger(name: str = "nnforhjb") -> logging.Logger:
    return logging.getLogger(name)
