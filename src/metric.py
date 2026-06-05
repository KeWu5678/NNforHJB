"""Table / summary helpers for experiment results.

All figure-producing helpers live in ``src/plot_value_function.py``; this
module keeps only the tabular summaries plus the shared result-loading
helper (``_load_results``) that the plotting module imports.
"""

from __future__ import annotations
from typing import Any, Sequence
from pathlib import Path
import os
import pickle
import numpy as np


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

    HPARAM_KEYS = ("alpha", "power", "activation", "loss_weights", "use_sphere",
                   "optimizer", "num_iterations", "num_insertion", "threshold")

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

    try:
        import pandas as pd  # type: ignore

        data_dict: dict[str, Any] = {
            "best_neurons": best_neurons_arr,
            loss_col: best_losses_arr,
        }
        if np.any(np.isfinite(best_err_l2_arr)):
            data_dict["err_l2"] = best_err_l2_arr
        if np.any(np.isfinite(best_err_h1_arr)):
            data_dict["err_h1"] = best_err_h1_arr

        df = pd.DataFrame(data_dict, index=[f"gamma={g:g}" for g in gammas])
        df_fmt = df.copy()
        df_fmt["best_neurons"] = df_fmt["best_neurons"].map(lambda v: f"{v:.0f}" if np.isfinite(v) else "nan")
        df_fmt[loss_col] = df_fmt[loss_col].map(lambda v: f"{v:.2e}" if np.isfinite(v) else "nan")
        for col in ("err_l2", "err_h1"):
            if col in df_fmt.columns:
                df_fmt[col] = df_fmt[col].map(lambda v: f"{v:.2e}" if np.isfinite(v) else "nan")
        result["table_df"] = df_fmt
    except Exception:
        pass

    return result
