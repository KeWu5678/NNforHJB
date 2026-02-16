"""MLflow integration utilities for NNforHJB experiments."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

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


def _pick_pdpa_key(d: Mapping[str, Any]) -> str:
    """Auto-detect the PDPA list key in a pickle dict."""
    for cand in ("pdpa_list_h1", "pdpa_list_l2", "pdpa_list"):
        if cand in d:
            return cand
    raise KeyError(
        "Could not infer PDPA list key. Expected one of "
        "('pdpa_list_h1', 'pdpa_list_l2', 'pdpa_list') in the pickle dict. "
        f"Available keys: {list(d.keys())}"
    )


def _pick_best_iteration_key(d: Mapping[str, Any]) -> Optional[str]:
    """Auto-detect the best_iteration key in a pickle dict."""
    for cand in ("best_iteration_h1", "best_iteration_l2", "best_iteration"):
        if cand in d:
            return cand
    return None


def _get_pdpa_for_index(
    gammas_arr: np.ndarray,
    pdpa_list_or_map: Any,
    i: int,
) -> Any:
    """Retrieve the PDPA object for gamma index *i*."""
    if isinstance(pdpa_list_or_map, Mapping):
        g = gammas_arr[i].item() if hasattr(gammas_arr[i], "item") else gammas_arr[i]
        if g in pdpa_list_or_map:
            return pdpa_list_or_map[g]
        if str(g) in pdpa_list_or_map:
            return pdpa_list_or_map[str(g)]
        raise KeyError(f"pdpa map has no entry for gamma={g!r}")
    if isinstance(pdpa_list_or_map, (list, tuple)):
        return pdpa_list_or_map[i]
    raise TypeError(
        f"PDPA container must be a list, tuple, or mapping; got {type(pdpa_list_or_map)}"
    )


def _loss_norm_tag(pdpa_key: str) -> str:
    """Derive a short tag like 'h1' or 'l2' from the pdpa key."""
    if "h1" in pdpa_key:
        return "h1"
    if "l2" in pdpa_key:
        return "l2"
    return "unknown"


# ---------------------------------------------------------------------------
# Retroactive import from pickle
# ---------------------------------------------------------------------------


def log_experiment_from_pickle(
    pkl_path: str,
    experiment_name: Optional[str] = None,
    tracking_uri: str = "file:./mlruns",
) -> str:
    """Import a completed experiment pickle into MLflow.

    Creates one **parent run** (per pickle file / seed) with **nested child
    runs** for each gamma value.

    Args:
        pkl_path: Path to the ``.pkl`` file.
        experiment_name: MLflow experiment name.  Defaults to the parent
            directory name (e.g. ``experiment_9_v1``).
        tracking_uri: MLflow tracking URI.

    Returns:
        The parent run ID.
    """
    pkl_path = Path(pkl_path)
    with open(pkl_path, "rb") as f:
        data: Dict[str, Any] = pickle.load(f)

    if experiment_name is None:
        experiment_name = pkl_path.parent.name

    setup_mlflow(experiment_name, tracking_uri)

    # Extract seed from filename (last part before .pkl, e.g. "…_h1_-2.pkl" → "-2")
    seed = pkl_path.stem.split("_")[-1]

    pdpa_key = _pick_pdpa_key(data)
    norm_tag = _loss_norm_tag(pdpa_key)
    gammas = np.asarray(data["gammas"]).reshape(-1)
    pdpa_list = data[pdpa_key]

    best_iter_key = _pick_best_iteration_key(data)
    best_iterations: Optional[List[int]] = (
        data[best_iter_key] if best_iter_key is not None else None
    )

    with mlflow.start_run(run_name=f"seed_{seed}") as parent_run:
        # -- shared hyperparameters ------------------------------------------
        mlflow.log_param("alpha", data.get("alpha"))
        mlflow.log_param("power", data.get("power"))
        mlflow.log_param("num_iteration", data.get("num_iteration"))
        mlflow.log_param("num_insertion", data.get("num_insertion"))
        mlflow.log_param("pruning_threshold", data.get("pruning_threshold"))
        mlflow.log_param("loss_weights", str(data.get("loss_weights")))
        mlflow.log_param("optimizer", str(data.get("optimizer")))
        mlflow.log_param("lr", str(data.get("lr")))
        mlflow.log_param("seed", seed)
        mlflow.log_param("norm", norm_tag)
        mlflow.set_tag("source_file", str(pkl_path))

        # -- per-gamma child runs --------------------------------------------
        for i, gamma in enumerate(gammas):
            pdpa = _get_pdpa_for_index(gammas, pdpa_list, i)

            with mlflow.start_run(
                run_name=f"gamma_{gamma:g}",
                nested=True,
            ):
                mlflow.log_param("gamma", float(gamma))

                train_loss = list(pdpa.train_loss)
                val_loss = list(pdpa.val_loss)

                # Step-based metrics
                for step in range(len(train_loss)):
                    metrics: Dict[str, float] = {
                        "train_loss": train_loss[step],
                        "val_loss": val_loss[step],
                    }
                    if hasattr(pdpa, "inner_weights") and step < len(
                        pdpa.inner_weights
                    ):
                        n_neurons = int(
                            pdpa.inner_weights[step]["weight"].shape[0]
                        )
                        metrics["num_neurons"] = n_neurons
                    mlflow.log_metrics(metrics, step=step)

                # Summary metrics
                if best_iterations is not None and i < len(best_iterations):
                    best_iter = int(best_iterations[i])
                else:
                    # Fallback: argmin of train_loss
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
                if (
                    hasattr(pdpa, "inner_weights")
                    and best_iter < len(pdpa.inner_weights)
                ):
                    mlflow.log_metric(
                        "best_num_neurons",
                        int(
                            pdpa.inner_weights[best_iter]["weight"].shape[0]
                        ),
                    )

    return parent_run.info.run_id


# ---------------------------------------------------------------------------
# Live training logging
# ---------------------------------------------------------------------------


def log_training_run(
    experiment_name: str,
    params: Dict[str, Any],
    gamma: float,
    pdpa_object: Any,
    best_iteration: int,
    artifacts: Optional[Dict[str, str]] = None,
    tracking_uri: str = "file:./mlruns",
) -> str:
    """Log a live training run to MLflow.

    Call this *after* training completes for a single gamma value.

    Args:
        experiment_name: MLflow experiment name.
        params: Shared hyperparameters dict (alpha, power, …).
        gamma: The gamma value for this run.
        pdpa_object: A trained ``PDPA_v1`` (or ``PDPA``) instance.
        best_iteration: Index of the best iteration.
        artifacts: Optional ``{name: file_path}`` dict of artifacts to log.
        tracking_uri: MLflow tracking URI.

    Returns:
        The MLflow run ID.
    """
    setup_mlflow(experiment_name, tracking_uri)

    with mlflow.start_run(run_name=f"gamma_{gamma:g}") as run:
        # Log hyperparameters
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("gamma", float(gamma))

        train_loss = list(pdpa_object.train_loss)
        val_loss = list(pdpa_object.val_loss)

        # Step-based metrics
        for step in range(len(train_loss)):
            metrics: Dict[str, float] = {
                "train_loss": train_loss[step],
                "val_loss": val_loss[step],
            }
            if hasattr(pdpa_object, "inner_weights") and step < len(
                pdpa_object.inner_weights
            ):
                n_neurons = int(
                    pdpa_object.inner_weights[step]["weight"].shape[0]
                )
                metrics["num_neurons"] = n_neurons
            mlflow.log_metrics(metrics, step=step)

        # Summary metrics
        best_iter = int(best_iteration)
        mlflow.log_metric("best_iteration", best_iter)
        if best_iter < len(train_loss):
            mlflow.log_metric("best_train_loss", train_loss[best_iter])
        if best_iter < len(val_loss):
            mlflow.log_metric("best_val_loss", val_loss[best_iter])
        if (
            hasattr(pdpa_object, "inner_weights")
            and best_iter < len(pdpa_object.inner_weights)
        ):
            mlflow.log_metric(
                "best_num_neurons",
                int(pdpa_object.inner_weights[best_iter]["weight"].shape[0]),
            )

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
