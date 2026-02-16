# MLflow Integration Plan for NNforHJB

## 1. Current State Analysis

### How Experiments Are Currently Managed

Experiments are saved as **pickle files** in `models/experiment_{N}_v{V}/` folders. Each `.pkl` file is a dictionary containing:

- **Hyperparameters**: `gammas`, `alpha`, `power`, `num_iteration`, `num_insertion`, `pruning_threshold`, `loss_weights`, `optimizer`, `lr`
- **PDPA objects** (one per gamma): each stores `train_loss`, `val_loss`, `inner_weights`, `outer_weights` per iteration
- **Metadata**: `best_iteration_h1` (best iteration index per gamma)

Analysis is done manually in `notebook/experiment_analysis.ipynb` using functions from `src/metric.py` that load pickles and compute aggregated statistics.

### Pain Points (What MLflow Solves)

| Problem | MLflow Solution |
|---------|----------------|
| No centralized experiment registry | MLflow Tracking Server / UI |
| Manual comparison via notebook code | MLflow UI comparison view |
| No searchable hyperparameter history | MLflow `search_runs()` API |
| Pickle files are opaque blobs | Structured params + metrics + artifacts |
| No run-level metadata (timestamps, status, tags) | Built-in run lifecycle tracking |
| Hard to reproduce past experiments | Full parameter + artifact logging |

---

## 2. MLflow Concepts Mapping

### Mapping: Current Structure → MLflow

```
Experiment Folder (e.g., experiment_9_v1/)  →  MLflow Experiment
  └── Pickle File (e.g., model_l2_-2.pkl)  →  MLflow Parent Run (one per seed/file)
        └── Per-Gamma PDPA result            →  MLflow Child Run (nested, one per gamma)
```

### What to Log

#### Parent Run (per pickle / per seed)
| MLflow Concept | What to Log |
|----------------|-------------|
| **Parameters** | `alpha`, `power`, `num_iteration`, `num_insertion`, `pruning_threshold`, `loss_weights`, `optimizer`, `lr`, `dataset`, `seed` |
| **Tags** | `dataset_name`, `experiment_version`, `activation_fn` |

#### Child Run (per gamma, nested under parent)
| MLflow Concept | What to Log |
|----------------|-------------|
| **Parameters** | `gamma` (the specific gamma for this run) |
| **Metrics** (step=iteration) | `train_loss`, `val_loss`, `num_neurons` |
| **Metrics** (final) | `best_train_loss`, `best_val_loss`, `best_num_neurons`, `best_iteration` |
| **Artifacts** | Convergence plot (PNG), weight snapshots (optional) |

### Why Nested Runs?

Each experiment trains multiple gamma values with the same hyperparameters and data. Nesting gamma runs under a parent keeps them grouped while allowing per-gamma metric tracking with proper step-based logging (which pickles cannot do cleanly).

---

## 3. Technical Implementation

### 3.1 Installation & Setup

```bash
pip install mlflow
```

Add to `requirements.txt` (already present as optional). Use local file-based tracking (no server needed initially):

```python
mlflow.set_tracking_uri("file:./mlruns")  # Store in project directory
```

Add `mlruns/` to `.gitignore`.

### 3.2 New Module: `src/mlflow_utils.py`

Create a utility module to wrap MLflow logging for this project's specific patterns.

```python
"""MLflow integration utilities for NNforHJB experiments."""
import mlflow
import pickle
import numpy as np
from pathlib import Path


def setup_mlflow(experiment_name: str, tracking_uri: str = "file:./mlruns"):
    """Initialize MLflow tracking."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_experiment_from_pickle(pkl_path: str, experiment_name: str = None):
    """
    Import a completed experiment from a pickle file into MLflow.
    Creates a parent run with child runs per gamma.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if experiment_name is None:
        experiment_name = Path(pkl_path).parent.name  # e.g., "experiment_9_v1"

    setup_mlflow(experiment_name)

    seed = Path(pkl_path).stem.split("_")[-1]  # Extract seed from filename

    with mlflow.start_run(run_name=f"seed_{seed}") as parent_run:
        # Log shared hyperparameters
        mlflow.log_param("alpha", data["alpha"])
        mlflow.log_param("power", data["power"])
        mlflow.log_param("num_iteration", data["num_iteration"])
        mlflow.log_param("num_insertion", data["num_insertion"])
        mlflow.log_param("pruning_threshold", data.get("pruning_threshold"))
        mlflow.log_param("loss_weights", str(data.get("loss_weights")))
        mlflow.log_param("optimizer", str(data.get("optimizer")))
        mlflow.log_param("lr", str(data.get("lr")))
        mlflow.log_param("seed", seed)
        mlflow.set_tag("source_file", pkl_path)

        # Determine the PDPA list key
        pdpa_key = None
        for key in ["pdpa_list_h1", "pdpa_list_l2"]:
            if key in data:
                pdpa_key = key
                break

        gammas = data["gammas"]
        pdpa_list = data[pdpa_key]
        best_iterations = data.get(f"best_iteration_{pdpa_key.split('_')[-1]}", [])

        for i, gamma in enumerate(gammas):
            pdpa = pdpa_list[i]
            with mlflow.start_run(
                run_name=f"gamma_{gamma}", nested=True
            ) as child_run:
                mlflow.log_param("gamma", gamma)

                # Log per-iteration metrics
                for step, (tl, vl) in enumerate(
                    zip(pdpa.train_loss, pdpa.val_loss)
                ):
                    mlflow.log_metrics(
                        {"train_loss": tl, "val_loss": vl},
                        step=step,
                    )
                    # Log neuron count at each iteration
                    if hasattr(pdpa, "inner_weights") and step < len(pdpa.inner_weights):
                        n_neurons = pdpa.inner_weights[step]["weight"].shape[0]
                        mlflow.log_metric("num_neurons", n_neurons, step=step)

                # Log best metrics
                if i < len(best_iterations):
                    best_iter = best_iterations[i]
                    mlflow.log_metric("best_iteration", best_iter)
                    mlflow.log_metric("best_train_loss", pdpa.train_loss[best_iter])
                    mlflow.log_metric("best_val_loss", pdpa.val_loss[best_iter])
                    if hasattr(pdpa, "inner_weights"):
                        best_neurons = pdpa.inner_weights[best_iter]["weight"].shape[0]
                        mlflow.log_metric("best_num_neurons", best_neurons)

    return parent_run.info.run_id


def log_training_run(
    experiment_name: str,
    params: dict,
    gamma: float,
    pdpa_object,
    best_iteration: int,
    artifacts: dict = None,
):
    """
    Log a live training run to MLflow (called from notebooks during training).

    Args:
        experiment_name: MLflow experiment name
        params: dict of hyperparameters
        gamma: the gamma value for this run
        pdpa_object: trained PDPA/PDPA_v1 object
        best_iteration: index of best iteration
        artifacts: optional dict of {name: file_path} to log
    """
    setup_mlflow(experiment_name)

    with mlflow.start_run(run_name=f"gamma_{gamma}"):
        # Log all hyperparameters
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_param("gamma", gamma)

        # Log per-iteration metrics
        for step, (tl, vl) in enumerate(
            zip(pdpa_object.train_loss, pdpa_object.val_loss)
        ):
            mlflow.log_metrics({"train_loss": tl, "val_loss": vl}, step=step)
            if hasattr(pdpa_object, "inner_weights") and step < len(pdpa_object.inner_weights):
                n_neurons = pdpa_object.inner_weights[step]["weight"].shape[0]
                mlflow.log_metric("num_neurons", n_neurons, step=step)

        # Log summary metrics
        mlflow.log_metric("best_iteration", best_iteration)
        mlflow.log_metric("best_train_loss", pdpa_object.train_loss[best_iteration])
        mlflow.log_metric("best_val_loss", pdpa_object.val_loss[best_iteration])

        # Log artifacts
        if artifacts:
            for name, path in artifacts.items():
                mlflow.log_artifact(path)


def import_all_experiments(models_dir: str = "models"):
    """Batch import all existing pickle files into MLflow."""
    models_path = Path(models_dir)
    for experiment_dir in sorted(models_path.iterdir()):
        if not experiment_dir.is_dir():
            continue
        for pkl_file in sorted(experiment_dir.glob("*.pkl")):
            print(f"Importing {pkl_file}...")
            try:
                log_experiment_from_pickle(str(pkl_file), experiment_dir.name)
            except Exception as e:
                print(f"  Error: {e}")
```

### 3.3 Notebook Integration

Modify notebooks to log to MLflow during training. Minimal changes needed:

```python
# At the top of the notebook
import mlflow
from src.mlflow_utils import setup_mlflow, log_training_run

setup_mlflow("experiment_14_v1")

# After training each gamma (existing loop already iterates over gammas)
for i, gamma in enumerate(gammas):
    pdpa = pdpa_list[i]
    # ... existing training code ...

    # NEW: Log to MLflow
    log_training_run(
        experiment_name="experiment_14_v1",
        params={
            "alpha": alpha,
            "power": power,
            "num_iteration": num_iteration,
            "num_insertion": num_insertion,
        },
        gamma=gamma,
        pdpa_object=pdpa,
        best_iteration=best_iterations[i],
    )
```

### 3.4 Retroactive Import of Existing Experiments

Run once to migrate all existing pickle data:

```python
from src.mlflow_utils import import_all_experiments
import_all_experiments("models")
```

This creates MLflow runs for all 10+ existing experiments (~50 pickle files).

### 3.5 Viewing Results

```bash
# Launch MLflow UI (local)
mlflow ui --port 5000
# Open http://localhost:5000
```

The UI provides:
- **Experiment list**: one entry per experiment folder
- **Run comparison**: side-by-side parameter/metric tables
- **Metric charts**: train_loss / val_loss / num_neurons over iterations
- **Search**: filter runs by `params.gamma = 0.1 AND metrics.best_val_loss < 0.01`

---

## 4. File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/mlflow_utils.py` | **New** | MLflow utility functions |
| `notebook/pdpa_vdp.ipynb` | **Edit** | Add MLflow logging after training |
| `notebook/experiment_analysis.ipynb` | **Edit** | Optional: add MLflow query-based analysis |
| `.gitignore` | **Edit** | Add `mlruns/` |
| `requirements.txt` | **Edit** | Ensure `mlflow>=2.0.0` is not commented out |

### What NOT to Change

- `src/PDPA_v1.py`, `src/model.py`: No changes to training code. MLflow logging is done externally after training completes — this keeps the training code clean and decoupled.
- `src/metric.py`: Keep existing analysis functions. They can coexist with MLflow, or be gradually replaced.

---

## 5. Implementation Steps

1. **Add `mlruns/` to `.gitignore`**
2. **Create `src/mlflow_utils.py`** with the utility functions above
3. **Run retroactive import** of all existing pickle experiments
4. **Verify** via `mlflow ui` that data appears correctly
5. **Update training notebooks** to log new experiments to MLflow
6. **Optionally** add MLflow-based queries to `experiment_analysis.ipynb`

---

## 6. Alternative Approaches Considered

| Approach | Pros | Cons |
|----------|------|------|
| **MLflow (chosen)** | Industry standard, great UI, search API, artifact storage | Extra dependency, learning curve |
| **Weights & Biases** | Superior visualization, cloud sync | Requires account, paid for teams |
| **TensorBoard** | Lightweight, PyTorch native | No parameter search, limited metadata |
| **Custom CSV/JSON** | No dependencies | No UI, manual comparison, not scalable |

MLflow is the best fit because:
- Works fully offline (file-based backend)
- Supports the nested run structure needed for gamma sweeps
- Step-based metric logging matches the iterative training pattern
- Python API integrates naturally with existing notebook workflow
- Can query runs programmatically for analysis in notebooks
