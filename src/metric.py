"""Table / summary helpers for experiment results.

All figure-producing helpers live in ``src/plots.py``; this
module keeps only the tabular summaries plus the shared result-loading
helper (``_load_results``) that the plotting module imports.
"""

from __future__ import annotations
from typing import Any, Callable, Mapping, Sequence
from pathlib import Path
import os
import pickle
import numpy as np


def _stringify_cell(value: Any, formatter: str | Callable[[Any], str] | None = None) -> str:
    if formatter is None:
        if isinstance(value, float):
            return f"{value:.3g}"
        return str(value)
    if isinstance(formatter, str):
        return formatter.format(value)
    return formatter(value)


def format_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    *,
    headers: Mapping[str, str] | None = None,
    formats: Mapping[str, str | Callable[[Any], str]] | None = None,
    title: str | None = None,
) -> str:
    """Return a compact Markdown table.

    This intentionally avoids pandas.  It is for small experiment summaries that
    should be easy to read in terminals, Markdown files, and notebooks.
    """
    headers = headers or {}
    formats = formats or {}
    labels = [headers.get(col, col) for col in columns]
    body = [
        [_stringify_cell(row.get(col, ""), formats.get(col)) for col in columns]
        for row in rows
    ]
    widths = [
        max(len(labels[i]), *(len(row[i]) for row in body)) if body else len(labels[i])
        for i in range(len(columns))
    ]

    def line(values: Sequence[str]) -> str:
        return "| " + " | ".join(values[i].ljust(widths[i]) for i in range(len(values))) + " |"

    sep = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
    parts = []
    if title:
        parts.extend([title, ""])
    parts.append(line(labels))
    parts.append(sep)
    parts.extend(line(row) for row in body)
    return "\n".join(parts)


def print_table(*args, **kwargs) -> None:
    """Print :func:`format_table` output."""
    print(format_table(*args, **kwargs))


def _load_results(results: Sequence[dict] | str | os.PathLike[str]) -> list[dict]:
    """Accept a list of result dicts or a pkl_path, return a list of dicts."""
    if isinstance(results, (str, os.PathLike)):
        p = Path(results)
        if not p.is_file():
            raise FileNotFoundError(f"File not found: {p}")
        with open(p, "rb") as f:
            results = pickle.load(f)
    if not isinstance(results, (list, tuple)) or not results:
        raise TypeError("results must be a non-empty list of result dicts")
    return list(results)


def print_experiment_hyperparameters(results: Sequence[dict] | str | os.PathLike[str]) -> None:
    """Print hyperparameters from experiment results.

    Args:
        results: list of result dicts, or path to a pickle containing one.
    """
    result_list = _load_results(results)

    HPARAM_KEYS = ("alpha", "power", "activation", "loss_weights",
                   "optimizer", "num_iterations", "num_insertion")

    # All results share the same hyperparameters (except gamma), so use the first
    r0 = result_list[0]
    gammas = [r["gamma"] for r in result_list]

    print(f"gammas: {gammas}")
    for k in HPARAM_KEYS:
        if k in r0:
            print(f"  {k}: {r0[k]!r}")


def summarize_final_neuron_count_and_loss(
    results: Sequence[dict] | str | os.PathLike[str],
    *,
    loss: str = "valid",
) -> dict[str, Any]:
    """Summarize neuron count and best loss per gamma from experiment results.

    Args:
        results: list of result dicts, or path to a pickle containing one.
        loss: ``"valid"`` or ``"train"`` — which loss history to use.
    """
    if loss not in {"valid", "train"}:
        raise ValueError(f"loss must be 'valid' or 'train', got {loss!r}")
    loss_key = "val_loss" if loss == "valid" else "train_loss"

    result_list = _load_results(results)
    gammas = np.array([r["gamma"] for r in result_list], dtype=float)

    best_neurons: list[float] = []
    best_losses: list[float] = []
    best_err_l2: list[float] = []
    best_err_h1: list[float] = []

    suffix = "val" if loss == "valid" else "train"

    for r in result_list:
        loss_hist = np.asarray(r[loss_key], dtype=float).reshape(-1)
        safe = np.where(np.isfinite(loss_hist), loss_hist, np.inf)

        if safe.size == 0 or not np.any(np.isfinite(safe)):
            best_losses.append(np.nan)
            best_neurons.append(np.nan)
            best_err_l2.append(np.nan)
            best_err_h1.append(np.nan)
            continue

        best_it = int(np.argmin(safe))
        best_losses.append(float(safe[best_it]))

        try:
            w = r["inner_weights"][best_it]["weight"]
            best_neurons.append(float(w.shape[0]))
        except Exception:
            best_neurons.append(np.nan)

        l2_hist = r.get(f"err_l2_{suffix}")
        h1_hist = r.get(f"err_h1_{suffix}")
        if l2_hist is not None and best_it < len(l2_hist):
            best_err_l2.append(float(l2_hist[best_it]))
        else:
            best_err_l2.append(np.nan)
        if h1_hist is not None and best_it < len(h1_hist):
            best_err_h1.append(float(h1_hist[best_it]))
        else:
            best_err_h1.append(np.nan)

    best_neurons_arr = np.asarray(best_neurons, dtype=float)
    best_losses_arr = np.asarray(best_losses, dtype=float)
    best_err_l2_arr = np.asarray(best_err_l2, dtype=float)
    best_err_h1_arr = np.asarray(best_err_h1, dtype=float)

    loss_col = "best_val_loss" if loss == "valid" else "best_train_loss"
    result: dict[str, Any] = {
        "gammas": gammas,
        "best_neurons": best_neurons_arr,
        loss_col: best_losses_arr,
        "best_err_l2": best_err_l2_arr,
        "best_err_h1": best_err_h1_arr,
    }

    rows = []
    for idx, gamma in enumerate(gammas):
        row: dict[str, Any] = {
            "gamma": float(gamma),
            "best_neurons": best_neurons_arr[idx],
            loss_col: best_losses_arr[idx],
        }
        if np.isfinite(best_err_l2_arr[idx]):
            row["err_l2"] = best_err_l2_arr[idx]
        if np.isfinite(best_err_h1_arr[idx]):
            row["err_h1"] = best_err_h1_arr[idx]
        rows.append(row)
    columns = ["gamma", "best_neurons", loss_col]
    if any("err_l2" in row for row in rows):
        columns.append("err_l2")
    if any("err_h1" in row for row in rows):
        columns.append("err_h1")
    result["table"] = format_table(
        rows,
        columns,
        formats={
            "gamma": "{:g}",
            "best_neurons": lambda v: "nan" if not np.isfinite(v) else f"{v:.0f}",
            loss_col: "{:.2e}",
            "err_l2": "{:.2e}",
            "err_h1": "{:.2e}",
        },
    )

    return result
