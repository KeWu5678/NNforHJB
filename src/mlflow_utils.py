"""MLflow integration utilities for NNforHJB experiments."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import mlflow
import numpy as np


def setup_mlflow(
    experiment_name: str,
    tracking_uri: str = "file:./mlruns",
) -> str:
    """Initialize MLflow tracking and return the experiment ID.

    Args:
        experiment_name: Name for the MLflow experiment.
        tracking_uri: MLflow tracking URI (default: local file store).

    Returns:
        The MLflow experiment ID.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    return experiment.experiment_id


# ---------------------------------------------------------------------------
# Retroactive import from pickle
# ---------------------------------------------------------------------------


def log_experiment_from_pickle(
    pkl_path: str,
    experiment_name: Optional[str] = None,
    tracking_uri: str = "file:./mlruns",
) -> str:
    """Import a completed experiment pickle into MLflow.

    Creates one **parent run** (per pickle file) with **nested child
    runs** for each gamma value.

    Args:
        pkl_path: Path to a ``.pkl`` file containing a list of result dicts.
        experiment_name: MLflow experiment name.  Defaults to the parent
            directory name (e.g. ``experiment_9``).
        tracking_uri: MLflow tracking URI.

    Returns:
        The parent run ID.
    """
    pkl_path = Path(pkl_path)
    with open(pkl_path, "rb") as f:
        result_list: List[Dict[str, Any]] = pickle.load(f)

    if experiment_name is None:
        experiment_name = pkl_path.parent.name

    setup_mlflow(experiment_name, tracking_uri)

    # Extract seed from filename (last part before .pkl, e.g. "…_h1_-2.pkl" -> "-2")
    seed = pkl_path.stem.split("_")[-1]

    # Shared hyperparameters from the first result
    r0 = result_list[0]

    with mlflow.start_run(run_name=f"seed_{seed}") as parent_run:
        # -- shared hyperparameters ------------------------------------------
        for k in ("alpha", "power", "activation", "loss_weights", "optimizer",
                   "use_sphere", "num_iterations", "num_insertion", "threshold"):
            if k in r0:
                mlflow.log_param(k, str(r0[k]))
        mlflow.log_param("seed", seed)
        mlflow.set_tag("source_file", str(pkl_path))

        # -- per-gamma child runs --------------------------------------------
        for r in result_list:
            gamma = r["gamma"]

            with mlflow.start_run(
                run_name=f"gamma_{gamma:g}",
                nested=True,
            ):
                mlflow.log_param("gamma", float(gamma))

                train_loss = list(r["train_loss"])
                val_loss = list(r["val_loss"])
                err_l2_train = list(r.get("err_l2_train", []))
                err_l2_val = list(r.get("err_l2_val", []))
                err_h1_train = list(r.get("err_h1_train", []))
                err_h1_val = list(r.get("err_h1_val", []))

                # Step-based metrics
                for step in range(len(train_loss)):
                    metrics: Dict[str, float] = {
                        "train_loss": train_loss[step],
                        "val_loss": val_loss[step],
                    }
                    if step < len(err_l2_train):
                        metrics["err_l2_train"] = err_l2_train[step]
                    if step < len(err_l2_val):
                        metrics["err_l2_val"] = err_l2_val[step]
                    if step < len(err_h1_train):
                        metrics["err_h1_train"] = err_h1_train[step]
                    if step < len(err_h1_val):
                        metrics["err_h1_val"] = err_h1_val[step]
                    if step < len(r.get("inner_weights", [])):
                        n_neurons = int(
                            r["inner_weights"][step]["weight"].shape[0]
                        )
                        metrics["num_neurons"] = n_neurons
                    mlflow.log_metrics(metrics, step=step)

                # Summary metrics
                best_iter = r.get("best_iteration")
                if best_iter is None:
                    safe = np.where(
                        np.isfinite(train_loss), train_loss, np.inf
                    )
                    best_iter = int(np.argmin(safe))

                mlflow.log_metric("best_iteration", best_iter)
                if best_iter < len(train_loss):
                    mlflow.log_metric(
                        "best_train_loss", train_loss[best_iter]
                    )
                if best_iter < len(val_loss):
                    mlflow.log_metric("best_val_loss", val_loss[best_iter])
                if best_iter < len(r.get("inner_weights", [])):
                    mlflow.log_metric(
                        "best_num_neurons",
                        int(
                            r["inner_weights"][best_iter]["weight"].shape[0]
                        ),
                    )
                for key in ("best_err_l2_train", "best_err_h1_train"):
                    if key in r:
                        mlflow.log_metric(key, float(r[key]))

    return parent_run.info.run_id


# ---------------------------------------------------------------------------
# Live training logging
# ---------------------------------------------------------------------------


def _activation_name(activation) -> str:
    """Extract a clean name from an activation function."""
    if callable(activation):
        return getattr(activation, "__name__", str(activation))
    return str(activation)


def _loss_type(loss_weights) -> str:
    """Map loss_weights tuple to a short label."""
    if isinstance(loss_weights, str):
        return loss_weights.lower()
    w = tuple(loss_weights)
    if w == (1.0, 0.0):
        return "l2"
    if w == (1.0, 1.0):
        return "h1"
    return f"w{w[0]:g}_{w[1]:g}"


def log_training_run(
    result: Dict[str, Any],
    seed: int,
    artifacts: Optional[Dict[str, str]] = None,
    tracking_uri: str = "file:./mlruns",
) -> str:
    """Log a training run to MLflow from a result dict.

    Call this *after* training completes for a single gamma value.

    Hierarchy:
        Experiment = ``"{activation}_p{power}_{loss_type}_seed{seed}"``
        Run        = ``"gamma_{gamma}"``

    Args:
        result: A result dict returned by ``PDPA_v2.retrain()``.
        seed: Random seed used for the experiment.
        artifacts: Optional ``{name: file_path}`` dict of artifacts to log.
        tracking_uri: MLflow tracking URI.

    Returns:
        The MLflow run ID.
    """
    activation = _activation_name(result.get("activation", "unknown"))
    power = result.get("power", 1)
    loss = _loss_type(result.get("loss_weights", (1.0, 1.0)))
    gamma = result["gamma"]

    experiment_name = f"{activation}_p{power:g}_{loss}_seed{seed}"
    run_name = f"gamma_{gamma:g}"
    setup_mlflow(experiment_name, tracking_uri)

    with mlflow.start_run(run_name=run_name) as run:
        # Log hyperparameters
        for k in ("alpha", "gamma", "power", "optimizer", "use_sphere",
                   "num_iterations", "num_insertion", "threshold"):
            if k in result:
                mlflow.log_param(k, str(result[k]))
        mlflow.log_param("activation", activation)
        mlflow.log_param("loss_type", loss)
        mlflow.log_param("seed", seed)

        train_loss = list(result["train_loss"])
        val_loss = list(result["val_loss"])
        err_l2_train = list(result.get("err_l2_train", []))
        err_l2_val = list(result.get("err_l2_val", []))
        err_h1_train = list(result.get("err_h1_train", []))
        err_h1_val = list(result.get("err_h1_val", []))

        # Step-based metrics
        for step in range(len(train_loss)):
            metrics: Dict[str, float] = {
                "train_loss": train_loss[step],
                "val_loss": val_loss[step],
            }
            if step < len(err_l2_train):
                metrics["err_l2_train"] = err_l2_train[step]
            if step < len(err_l2_val):
                metrics["err_l2_val"] = err_l2_val[step]
            if step < len(err_h1_train):
                metrics["err_h1_train"] = err_h1_train[step]
            if step < len(err_h1_val):
                metrics["err_h1_val"] = err_h1_val[step]
            if step < len(result.get("inner_weights", [])):
                n_neurons = int(
                    result["inner_weights"][step]["weight"].shape[0]
                )
                metrics["num_neurons"] = n_neurons
            mlflow.log_metrics(metrics, step=step)

        # Summary metrics
        best_iter = result.get("best_iteration", 0)
        mlflow.log_metric("best_iteration", best_iter)
        if best_iter < len(train_loss):
            mlflow.log_metric("best_train_loss", train_loss[best_iter])
        if best_iter < len(val_loss):
            mlflow.log_metric("best_val_loss", val_loss[best_iter])
        if best_iter < len(result.get("inner_weights", [])):
            mlflow.log_metric(
                "best_num_neurons",
                int(result["inner_weights"][best_iter]["weight"].shape[0]),
            )
        for key in ("best_err_l2_train", "best_err_h1_train"):
            if key in result:
                mlflow.log_metric(key, float(result[key]))

        # Artifacts
        if artifacts:
            for path in artifacts.values():
                mlflow.log_artifact(path)

    return run.info.run_id


# ---------------------------------------------------------------------------
# Batch retroactive import
# ---------------------------------------------------------------------------


def import_all_experiments(
    models_dir: str = "models",
    tracking_uri: str = "file:./mlruns",
) -> Dict[str, List[str]]:
    """Import all existing pickle experiments into MLflow.

    Args:
        models_dir: Root directory containing experiment folders.
        tracking_uri: MLflow tracking URI.

    Returns:
        A dict mapping experiment folder names to lists of parent run IDs.
    """
    models_path = Path(models_dir)
    results: Dict[str, List[str]] = {}

    for experiment_dir in sorted(models_path.iterdir()):
        if not experiment_dir.is_dir():
            continue
        pkl_files = sorted(experiment_dir.glob("*.pkl"))
        if not pkl_files:
            continue

        run_ids: List[str] = []
        for pkl_file in pkl_files:
            print(f"Importing {pkl_file} ...")
            try:
                run_id = log_experiment_from_pickle(
                    str(pkl_file),
                    experiment_name=experiment_dir.name,
                    tracking_uri=tracking_uri,
                )
                run_ids.append(run_id)
                print(f"  -> parent run {run_id}")
            except Exception as e:
                print(f"  ERROR: {e}")
        results[experiment_dir.name] = run_ids

    return results
