from __future__ import annotations
from typing import Any, Mapping, Optional, Sequence, Tuple
from pathlib import Path
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter


def _get_field(dataset: Any, name: str) -> np.ndarray:
    """Extract a field from a structured array or dict-like dataset."""
    if isinstance(dataset, np.ndarray) and dataset.dtype.fields is not None:
        if name not in dataset.dtype.fields:
            raise KeyError(f"Dataset is missing field '{name}'. Available: {list(dataset.dtype.fields.keys())}")
        return dataset[name]
    if isinstance(dataset, Mapping):
        if name not in dataset:
            raise KeyError(f"Dataset is missing key '{name}'. Available: {list(dataset.keys())}")
        return np.asarray(dataset[name])
    raise TypeError(
        "dataset must be a NumPy structured array (with fields) or a dict-like object containing keys "
        "'x', 'v', 'dv'."
    )


def plot_vdp_value_scatter3d(
    dataset: Any,
    *,
    title: str = "VDP dataset: value function V(x₀)",
    s: float = 30.0,
    alpha: float = 0.8,
    cmap: str = "viridis",
    elev: float = 30.0,
    azim: float = 45.0,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, Any]:
    """
    3D scatter plot of (x0[0], x0[1], v), colored by v.

    Args:
        dataset: structured array or dict-like with 'x' and 'v'
        title: plot title
        s: marker size
        alpha: marker transparency
        cmap: matplotlib colormap name
        elev/azim: 3D view angles
        save_path: optional path to save figure
        show: call plt.show()

    Returns:
        (fig, ax) where ax is a 3D axis.
    """
    x = _get_field(dataset, "x")
    v = _get_field(dataset, "v")

    x = np.asarray(x)
    v = np.asarray(v).reshape(-1)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"Expected dataset['x'] shape (N, 2), got {x.shape}")
    if v.shape[0] != x.shape[0]:
        raise ValueError(f"Mismatched lengths: x has {x.shape[0]} rows, v has {v.shape[0]} entries")

    x0_0 = x[:, 0]
    x0_1 = x[:, 1]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x0_0, x0_1, v, c=v, cmap=cmap, s=s, alpha=alpha)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("Value function V")

    ax.set_xlabel("x₀[0]")
    ax.set_ylabel("x₀[1]")
    ax.set_zlabel("V(x₀)")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_vdp_value_with_gradient_arrows2d(
    dataset: Any,
    *,
    title: str = "VDP dataset: V(x₀) with ∇V arrows",
    grid_size: int = 15,
    point_s: float = 20.0,
    point_alpha: float = 0.6,
    cmap: str = "viridis",
    arrow_color: str = "red",
    arrow_alpha: float = 0.7,
    arrow_scale: float = 0.15,
    head_width: float = 0.05,
    head_length: float = 0.08,
    normalize: bool = True,
    color_arrows_by_magnitude: bool = True,
    magnitude_cmap: str = "magma",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    2D scatter of x0 colored by v, with gradient arrows using dv.

    The arrows are drawn on an evenly spaced grid in (x0[0], x0[1]); for each grid
    location we find the nearest dataset point and use its dv as the arrow direction.

    Args:
        dataset: structured array or dict-like with 'x', 'v', 'dv'
        grid_size: number of grid points per axis for arrow placement
        normalize: whether to rescale arrows for legibility (similar to rawdata script)
        arrow_scale: global scale factor (used when normalize=True)
        save_path: optional path to save figure
        show: call plt.show()

    Returns:
        (fig, ax)
    """
    x = _get_field(dataset, "x")
    v = _get_field(dataset, "v")
    dv = _get_field(dataset, "dv")

    x = np.asarray(x)
    v = np.asarray(v).reshape(-1)
    dv = np.asarray(dv)
    grid_size = int(grid_size)

    x0_0 = x[:, 0]
    x0_1 = x[:, 1]

    fig, ax = plt.subplots(figsize=(12, 10))
    sc = ax.scatter(x0_0, x0_1, c=v, cmap=cmap, s=point_s, alpha=point_alpha)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Value function V")

    x_min, x_max = float(np.min(x0_0)), float(np.max(x0_0))
    y_min, y_max = float(np.min(x0_1)), float(np.max(x0_1))
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    Xg, Yg = np.meshgrid(x_grid, y_grid)

    # Collect arrows so we can optionally color by ||dv||
    X_list: list[float] = []
    Y_list: list[float] = []
    U_list: list[float] = []
    V_list: list[float] = []
    M_list: list[float] = []

    for i in range(grid_size):
        for j in range(grid_size):
            x_point = float(Xg[i, j])
            y_point = float(Yg[i, j])

            # nearest neighbor in dataset (squared Euclidean distance in x-space)
            d2 = (x0_0 - x_point) ** 2 + (x0_1 - y_point) ** 2
            idx = int(np.argmin(d2))

            dx = float(dv[idx, 0])
            dy = float(dv[idx, 1])
            mag = float(np.sqrt(dx * dx + dy * dy))
            if mag <= 0.0:
                continue

            if normalize:
                # Keep arrows legible; magnitude is shown by color if enabled.
                scale = arrow_scale * (1.0 / (1.0 + 0.5 * mag))
                dx_plot = dx * scale
                dy_plot = dy * scale
            else:
                # Arrow length corresponds to magnitude (may look cluttered).
                dx_plot = dx
                dy_plot = dy

            X_list.append(x_point)
            Y_list.append(y_point)
            U_list.append(dx_plot)
            V_list.append(dy_plot)
            M_list.append(mag)

    if color_arrows_by_magnitude and len(X_list) > 0:
        q = ax.quiver(
            np.array(X_list),
            np.array(Y_list),
            np.array(U_list),
            np.array(V_list),
            np.array(M_list),
            cmap=magnitude_cmap,
            alpha=arrow_alpha,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.003,
        )
        cbar2 = fig.colorbar(q, ax=ax, pad=0.02, fraction=0.046)
        cbar2.set_label(r"$\|\nabla V(x)\|$")
    else:
        for x_point, y_point, dx_plot, dy_plot in zip(X_list, Y_list, U_list, V_list):
            ax.arrow(
                x_point,
                y_point,
                dx_plot,
                dy_plot,
                head_width=head_width,
                head_length=head_length,
                fc=arrow_color,
                ec=arrow_color,
                alpha=arrow_alpha,
                length_includes_head=True,
            )

    ax.set_xlabel("x₀[0]")
    ax.set_ylabel("x₀[1]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def print_experiment_hyperparameters(folder: str | os.PathLike[str]) -> None:
    """
    1) given a folder like 'models/experiment_1'
    2) read all the pickle file (duplicate experiments of certain hyper parameter and saved model)
    3) print out the hyper parameter

    This expects the pickle structure used in the notebooks: a dict with keys like
    'gammas', 'alpha', 'power', 'num_iteration', 'num_insertion', 'pruning_threshold', ...,
    plus a PDPA list under 'pdpa_list_h1' or 'pdpa_list_l2'.
    """
    def _extract_hparams(d: Mapping[str, Any]) -> dict[str, Any]:
        # Keep only the things you would call "hyperparameters" (not trained PDPA objects / outcomes).
        wanted = ("gammas", "alpha", "power", "num_iteration", "num_insertion", "pruning_threshold")
        out: dict[str, Any] = {}
        for k in wanted:
            if k not in d:
                continue
            if k == "gammas":
                out[k] = np.asarray(d[k]).reshape(-1).tolist()
            else:
                out[k] = d[k]
        return out

    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Folder not found or not a directory: {folder_path}")

    pkl_files = sorted([p for p in folder_path.iterdir() if p.is_file() and p.suffix == ".pkl"])
    if len(pkl_files) == 0:
        print(f"[print_experiment_hyperparameters] No .pkl files found in: {folder_path}")
        return

    loaded: list[tuple[Path, dict[str, Any]]] = []
    for p in pkl_files:
        with open(p, "rb") as f:
            model_dict = pickle.load(f)
        if not isinstance(model_dict, Mapping):
            raise TypeError(f"Pickle '{p.name}' is not a dict-like object; got {type(model_dict)}")
        loaded.append((p, _extract_hparams(model_dict)))

    # Print only once (they should be the same across duplicate experiments).
    base_path, base_hparams = loaded[0]
    print(f"=== {folder_path} ({len(loaded)} files) ===")
    for k in sorted(base_hparams.keys(), key=str):
        print(f"  {k}: {base_hparams[k]!r}")

    # Warn if any file differs.
    diffs: list[str] = []
    for p, hp in loaded[1:]:
        if hp != base_hparams:
            diffs.append(p.name)
    if len(diffs) > 0:
        print(f"\n[warn] Hyperparameters differ in {len(diffs)} file(s): {diffs}")


def build_best_val_loss_table_by_gamma(
    folder: str | os.PathLike[str],
    *,
    pdpa_key: Optional[str] = None,
    gammas_key: str = "gammas",
    kind: str = "loss",
    loss: str = "valid",
) -> dict[str, Any]:
    """
    1) given a folder like 'models/experiment_1' OR a single '.pkl' file path
    2) return a table with rows = different iteration, columns = different gamma
    3) behavior depends on `kind`:
       - kind="loss": average (across duplicate runs) of best-so-far loss:
            best_loss[t] = min(loss_hist[:t+1])
       - kind="neuron": average (across duplicate runs) of neuron count at the best-so-far iteration:
            best_idx[t] = argmin(loss_hist[:t+1])
            best_neurons[t] = pdpa.inner_weights[best_idx[t]]["weight"].shape[0]

    If a single '.pkl' file path is provided, the output corresponds to that one run
    (i.e. the "average across runs" is just the run itself).

    Returns:
      {
        'gammas': np.ndarray shape (G,),
        'iterations': np.ndarray shape (T,),
        'best_val_loss_mean': np.ndarray shape (T, G) (NaN where missing),  # valid-only, kept name for compatibility
        'best_train_loss_mean': np.ndarray shape (T, G) (NaN where missing),  # only when loss="train"
        'counts': np.ndarray shape (T, G) number of runs contributing,
        'table_df': pandas DataFrame (optional, formatted strings),
        'files': list[str]
      }
    """
    if kind not in {"loss", "neuron"}:
        raise ValueError(f"kind must be 'loss' or 'neuron', got {kind!r}")
    if loss not in {"valid", "train"}:
        raise ValueError(f"loss must be 'valid' or 'train', got {loss!r}")

    loss_attr = "val_loss" if loss == "valid" else "train_loss"

    def _pick_pdpa_key(d: Mapping[str, Any]) -> str:
        if pdpa_key is not None:
            if pdpa_key not in d:
                raise KeyError(f"Requested pdpa_key '{pdpa_key}' not found. Available keys: {list(d.keys())}")
            return pdpa_key
        for cand in ("pdpa_list_h1", "pdpa_list_l2", "pdpa_list"):
            if cand in d:
                return cand
        raise KeyError(
            "Could not infer PDPA list key. Expected one of "
            "('pdpa_list_h1', 'pdpa_list_l2', 'pdpa_list') in the pickle dict."
        )

    def _get_pdpa_for_index(gammas_arr: np.ndarray, pdpa_list_or_map: Any, i: int) -> Any:
        if isinstance(pdpa_list_or_map, Mapping):
            g = gammas_arr[i].item() if hasattr(gammas_arr[i], "item") else gammas_arr[i]
            if g in pdpa_list_or_map:
                return pdpa_list_or_map[g]
            gs = str(g)
            if gs in pdpa_list_or_map:
                return pdpa_list_or_map[gs]
            raise KeyError(f"pdpa map has no entry for gamma={g!r}. Keys: {list(pdpa_list_or_map.keys())[:10]}")
        if not isinstance(pdpa_list_or_map, Sequence):
            raise TypeError(f"PDPA container must be a sequence or mapping; got {type(pdpa_list_or_map)}")
        if len(pdpa_list_or_map) != len(gammas_arr):
            raise ValueError(f"Length mismatch: len(gammas)={len(gammas_arr)} vs len(pdpa_list)={len(pdpa_list_or_map)}")
        return pdpa_list_or_map[i]

    input_path = Path(folder)
    if not input_path.exists():
        raise FileNotFoundError(f"Path not found: {input_path}")
    if input_path.is_dir():
        pkl_files = sorted([p for p in input_path.iterdir() if p.is_file() and p.suffix == ".pkl"])
        if len(pkl_files) == 0:
            raise FileNotFoundError(f"No .pkl files found in directory: {input_path}")
    elif input_path.is_file():
        if input_path.suffix != ".pkl":
            raise ValueError(f"Expected a directory or a '.pkl' file, got: {input_path}")
        pkl_files = [input_path]
    else:
        raise FileNotFoundError(f"Path is neither a file nor a directory: {input_path}")

    loaded: list[tuple[Path, Mapping[str, Any], str]] = []
    base_gammas: Optional[np.ndarray] = None
    max_iters = 0

    for p in pkl_files:
        with open(p, "rb") as f:
            model_dict = pickle.load(f)
        if not isinstance(model_dict, Mapping):
            raise TypeError(f"Pickle '{p.name}' is not a dict-like object; got {type(model_dict)}")

        k_pdpa = _pick_pdpa_key(model_dict)
        if gammas_key not in model_dict:
            raise KeyError(f"Pickle '{p.name}' is missing key '{gammas_key}'")

        gammas_arr = np.asarray(model_dict[gammas_key]).reshape(-1).astype(float)
        if base_gammas is None:
            base_gammas = gammas_arr.copy()
        else:
            # Duplicate experiments should share identical gamma grid (and ordering).
            if base_gammas.shape != gammas_arr.shape or not np.allclose(base_gammas, gammas_arr, rtol=0.0, atol=0.0):
                raise ValueError(
                    f"Gamma list mismatch in '{p.name}'.\n"
                    f"Expected: {base_gammas.tolist()}\n"
                    f"Got:      {gammas_arr.tolist()}"
                )

        pdpa_list_or_map = model_dict[k_pdpa]
        for i in range(len(gammas_arr)):
            pdpa = _get_pdpa_for_index(gammas_arr, pdpa_list_or_map, i)
            if not hasattr(pdpa, loss_attr):
                raise AttributeError(
                    f"PDPA object for gamma={gammas_arr[i]} in '{p.name}' has no attribute '{loss_attr}'"
                )
            max_iters = max(max_iters, int(np.asarray(getattr(pdpa, loss_attr)).reshape(-1).shape[0]))

        loaded.append((p, model_dict, k_pdpa))

    assert base_gammas is not None
    gammas = base_gammas
    G = int(gammas.shape[0])

    sums = np.zeros((max_iters, G), dtype=float)
    counts = np.zeros((max_iters, G), dtype=int)

    for p, model_dict, k_pdpa in loaded:
        gammas_arr = np.asarray(model_dict[gammas_key]).reshape(-1).astype(float)
        pdpa_list_or_map = model_dict[k_pdpa]
        for i in range(len(gammas_arr)):
            col = int(i)  # column is the gamma index (order preserved from the pickle)
            pdpa = _get_pdpa_for_index(gammas_arr, pdpa_list_or_map, i)
            loss_hist = np.asarray(getattr(pdpa, loss_attr), dtype=float).reshape(-1)
            if loss_hist.size == 0:
                continue

            safe = np.where(np.isfinite(loss_hist), loss_hist, np.inf)
            best_so_far_loss = np.minimum.accumulate(safe)

            if kind == "loss":
                best_so_far = np.where(np.isfinite(best_so_far_loss), best_so_far_loss, np.nan)
                for t in range(best_so_far.shape[0]):
                    v = float(best_so_far[t])
                    if not np.isfinite(v):
                        continue
                    sums[t, col] += v
                    counts[t, col] += 1
            else:
                if not hasattr(pdpa, "inner_weights"):
                    raise AttributeError(
                        "kind='neuron' requires PDPA objects to have 'inner_weights' (list of per-iteration weights)."
                    )
                inner_weights = list(getattr(pdpa, "inner_weights"))
                num_iters = min(int(best_so_far_loss.shape[0]), len(inner_weights))
                if num_iters == 0:
                    continue

                # For each t, pick best_idx = argmin(loss[:t+1]) and record neuron count at that snapshot.
                for t in range(num_iters):
                    window = best_so_far_loss[: t + 1]
                    if not np.any(np.isfinite(window)):
                        continue
                    best_idx = int(np.argmin(window))
                    w = inner_weights[best_idx]["weight"]
                    n = float(getattr(w, "shape")[0])
                    if not np.isfinite(n):
                        continue
                    sums[t, col] += n
                    counts[t, col] += 1

    mean = np.full_like(sums, np.nan, dtype=float)
    nonzero = counts > 0
    mean[nonzero] = sums[nonzero] / counts[nonzero]

    result_loss_key = "best_val_loss_mean" if loss == "valid" else "best_train_loss_mean"
    result = {
        "gammas": gammas,
        "iterations": np.arange(max_iters, dtype=int),
        # rows = iteration, cols = gamma
        result_loss_key: mean,
        "table": mean,  # alias (more explicit name)
        "counts": counts,
        "loss": loss,
        "files": [p.name for p in pkl_files],
    }

    # Optional: provide a nice DataFrame for notebook display.
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(
            mean,
            index=[f"iter={t}" for t in range(max_iters)],
            columns=[f"gamma={g:g}" for g in gammas],
        )
        # Minimal, predictable notebook output
        fmt = "{:.2e}" if kind == "loss" else "{:.2f}"
        df_fmt = df.copy()
        for c in df_fmt.columns:
            df_fmt[c] = df_fmt[c].map(lambda v: fmt.format(v) if np.isfinite(v) else "nan")
        result["table_df"] = df_fmt
    except Exception:
        pass

    return result


def summarize_best_iteration_and_loss_by_gamma(
    folder: str | os.PathLike[str],
    *,
    pdpa_key: Optional[str] = None,
    gammas_key: str = "gammas",
    best_iteration_key: Optional[str] = None,
    loss: str = "valid",
) -> dict[str, Any]:
    """
    Given a folder of duplicate experiment pickles, return a per-gamma summary table:

    - rows: gamma values
    - col 1: best_neurons (mean across runs, rounded to int)
    - col 2: best_(val|train)_loss (mean across runs)

    If the pickle contains a 'best_iteration_*' list (e.g. 'best_iteration_l2'),
    that is used when loss="valid". Otherwise, best_iteration is computed as argmin(loss_hist).
    Best loss is computed as min(loss_hist) (finite values only).
    """
    if loss not in {"valid", "train"}:
        raise ValueError(f"loss must be 'valid' or 'train', got {loss!r}")
    loss_attr = "val_loss" if loss == "valid" else "train_loss"

    def _pick_pdpa_key(d: Mapping[str, Any]) -> str:
        if pdpa_key is not None:
            if pdpa_key not in d:
                raise KeyError(f"Requested pdpa_key '{pdpa_key}' not found. Available keys: {list(d.keys())}")
            return pdpa_key
        for cand in ("pdpa_list_h1", "pdpa_list_l2", "pdpa_list"):
            if cand in d:
                return cand
        raise KeyError(
            "Could not infer PDPA list key. Expected one of "
            "('pdpa_list_h1', 'pdpa_list_l2', 'pdpa_list') in the pickle dict."
        )

    def _pick_best_iteration_key(d: Mapping[str, Any]) -> Optional[str]:
        if best_iteration_key is not None:
            return best_iteration_key if best_iteration_key in d else None
        for cand in ("best_iteration_l2", "best_iteration_h1", "best_iteration"):
            if cand in d:
                return cand
        return None

    def _get_pdpa_for_index(gammas_arr: np.ndarray, pdpa_list_or_map: Any, i: int) -> Any:
        if isinstance(pdpa_list_or_map, Mapping):
            g = gammas_arr[i].item() if hasattr(gammas_arr[i], "item") else gammas_arr[i]
            if g in pdpa_list_or_map:
                return pdpa_list_or_map[g]
            gs = str(g)
            if gs in pdpa_list_or_map:
                return pdpa_list_or_map[gs]
            raise KeyError(f"pdpa map has no entry for gamma={g!r}. Keys: {list(pdpa_list_or_map.keys())[:10]}")
        if not isinstance(pdpa_list_or_map, Sequence):
            raise TypeError(f"PDPA container must be a sequence or mapping; got {type(pdpa_list_or_map)}")
        if len(pdpa_list_or_map) != len(gammas_arr):
            raise ValueError(f"Length mismatch: len(gammas)={len(gammas_arr)} vs len(pdpa_list)={len(pdpa_list_or_map)}")
        return pdpa_list_or_map[i]

    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Folder not found or not a directory: {folder_path}")

    pkl_files = sorted([p for p in folder_path.iterdir() if p.is_file() and p.suffix == ".pkl"])
    if len(pkl_files) == 0:
        raise FileNotFoundError(f"No .pkl files found in: {folder_path}")

    gammas: Optional[np.ndarray] = None
    best_iter_lists: list[list[float]] = []
    best_neuron_lists: list[list[float]] = []
    best_loss_lists: list[list[float]] = []

    for p in pkl_files:
        with open(p, "rb") as f:
            d = pickle.load(f)
        if not isinstance(d, Mapping):
            raise TypeError(f"Pickle '{p.name}' is not a dict-like object; got {type(d)}")
        if gammas_key not in d:
            raise KeyError(f"Pickle '{p.name}' is missing key '{gammas_key}'")

        gammas_arr = np.asarray(d[gammas_key]).reshape(-1).astype(float)
        if gammas is None:
            gammas = gammas_arr.copy()
        else:
            if gammas.shape != gammas_arr.shape or not np.allclose(gammas, gammas_arr, rtol=0.0, atol=0.0):
                raise ValueError(
                    f"Gamma list mismatch in '{p.name}'.\n"
                    f"Expected: {gammas.tolist()}\n"
                    f"Got:      {gammas_arr.tolist()}"
                )

        k_pdpa = _pick_pdpa_key(d)
        pdpa_list_or_map = d[k_pdpa]
        k_best_it = _pick_best_iteration_key(d)
        best_it_list = d.get(k_best_it) if (loss == "valid" and k_best_it is not None) else None

        file_best_iters: list[float] = []
        file_best_neurons: list[float] = []
        file_best_losses: list[float] = []
        for i in range(len(gammas_arr)):
            pdpa = _get_pdpa_for_index(gammas_arr, pdpa_list_or_map, i)
            loss_hist = np.asarray(getattr(pdpa, loss_attr), dtype=float).reshape(-1)
            safe = np.where(np.isfinite(loss_hist), loss_hist, np.inf)
            if safe.size == 0 or not np.any(np.isfinite(safe)):
                file_best_losses.append(np.nan)
                file_best_iters.append(np.nan)
                file_best_neurons.append(np.nan)
                continue

            best_loss = float(np.min(safe))
            if best_it_list is not None and isinstance(best_it_list, Sequence) and len(best_it_list) == len(gammas_arr):
                best_it = float(best_it_list[i])
            else:
                best_it = float(int(np.argmin(safe)))

            file_best_losses.append(best_loss)
            file_best_iters.append(best_it)

            # Neuron count at the best iteration snapshot.
            try:
                inner_weights = list(getattr(pdpa, "inner_weights"))
                it_int = int(np.rint(best_it))
                if it_int < 0 or it_int >= len(inner_weights):
                    file_best_neurons.append(np.nan)
                else:
                    w = inner_weights[it_int]["weight"]
                    file_best_neurons.append(float(getattr(w, "shape")[0]))
            except Exception:
                file_best_neurons.append(np.nan)

        best_iter_lists.append(file_best_iters)
        best_neuron_lists.append(file_best_neurons)
        best_loss_lists.append(file_best_losses)

    assert gammas is not None
    best_iters_arr = np.asarray(best_iter_lists, dtype=float)  # (R, G)
    best_neurons_arr = np.asarray(best_neuron_lists, dtype=float)  # (R, G)
    best_losses_arr = np.asarray(best_loss_lists, dtype=float)  # (R, G)

    best_iter_mean = np.nanmean(best_iters_arr, axis=0)
    best_neuron_mean = np.nanmean(best_neurons_arr, axis=0)
    best_loss_mean = np.nanmean(best_losses_arr, axis=0)

    # "round up to integer" -> round to nearest int (NaN stays NaN)
    best_iter_mean_int = np.where(np.isfinite(best_iter_mean), np.rint(best_iter_mean), np.nan).astype(float)
    best_neuron_mean_int = np.where(np.isfinite(best_neuron_mean), np.rint(best_neuron_mean), np.nan).astype(float)

    result_loss_key = "best_val_loss_mean" if loss == "valid" else "best_train_loss_mean"
    result: dict[str, Any] = {
        "gammas": gammas,
        # Kept for debugging/back-compat, but prefer best_neurons_mean for reporting.
        "best_iteration_mean": best_iter_mean_int,
        "best_neurons_mean": best_neuron_mean_int,
        result_loss_key: best_loss_mean,
        "loss": loss,
        "files": [p.name for p in pkl_files],
    }

    try:
        import pandas as pd  # type: ignore

        loss_col = "best_val_loss" if loss == "valid" else "best_train_loss"
        df = pd.DataFrame(
            {"best_neurons": best_neuron_mean_int, loss_col: best_loss_mean},
            index=[f"gamma={g:g}" for g in gammas],
        )
        df_fmt = df.copy()
        df_fmt["best_neurons"] = df_fmt["best_neurons"].map(lambda v: f"{v:.0f}" if np.isfinite(v) else "nan")
        # Scientific notation, 2 digits after decimal (e.g. 1.23e-02).
        df_fmt[loss_col] = df_fmt[loss_col].map(lambda v: f"{v:.2e}" if np.isfinite(v) else "nan")
        result["table_df"] = df_fmt
    except Exception:
        pass

    return result


def plot_best_loss_vs_best_neurons_by_gamma(
    folder: str | os.PathLike[str],
    *,
    pdpa_key: Optional[str] = None,
    gammas_key: str = "gammas",
    gammas_include: Optional[Sequence[float]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Best-so-far loss vs best-so-far neurons (avg across runs; single file plots one run)",
    xlabel: str = "Avg best-so-far neuron count",
    ylabel: str = "Avg best-so-far validation loss",
    logy: bool = False,
    legend: bool = True,
    linewidth: float = 1.8,
    alpha: float = 0.95,
    marker: Optional[str] = "o",
    markersize: float = 4.0,
    loss: str = "valid",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    For each gamma (one line), plot:
      x-axis: average best-so-far neuron count at iteration t
      y-axis: average best-so-far validation loss at iteration t

    Input can be either:
    - a directory containing multiple '.pkl' files (duplicate runs), or
    - a single '.pkl' file (plots that single run).

    Data is computed using `build_best_val_loss_table_by_gamma(..., kind=...)`.

    If `gammas_include` is provided, only plot lines for gamma values in that list.
    """
    loss_out = build_best_val_loss_table_by_gamma(
        folder, pdpa_key=pdpa_key, gammas_key=gammas_key, kind="loss", loss=loss
    )
    neu_out = build_best_val_loss_table_by_gamma(
        folder, pdpa_key=pdpa_key, gammas_key=gammas_key, kind="neuron", loss=loss
    )

    gammas = np.asarray(loss_out["gammas"], dtype=float).reshape(-1)
    gammas2 = np.asarray(neu_out["gammas"], dtype=float).reshape(-1)
    if gammas.shape != gammas2.shape or not np.allclose(gammas, gammas2, rtol=0.0, atol=0.0):
        raise ValueError("Gamma lists differ between loss and neuron aggregation outputs.")

    loss_tbl = np.asarray(loss_out["table"], dtype=float)  # (T, G)
    neu_tbl = np.asarray(neu_out["table"], dtype=float)  # (T, G)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    include_set = None
    if gammas_include is not None:
        include_set = set(float(x) for x in gammas_include)

    plotted = 0
    for j, g in enumerate(gammas):
        if include_set is not None and float(g) not in include_set:
            continue
        xs = neu_tbl[:, j]
        ys = loss_tbl[:, j]
        finite = np.isfinite(xs) & np.isfinite(ys)
        if not np.any(finite):
            continue
        xs_f = xs[finite]
        ys_f = ys[finite]
        ax.plot(
            xs_f,
            ys_f,
            linewidth=linewidth,
            alpha=alpha,
            marker=marker,
            markersize=markersize,
            label=fr"$\gamma$={float(g):g}",
        )

        # Mark the end point (last iteration): this corresponds to the best iteration reached so far.
        end_x = float(xs_f[-1])
        end_y = float(ys_f[-1])
        ax.scatter([end_x], [end_y], s=120, marker="o", edgecolors="black", linewidths=1.0, zorder=6)
        ax.annotate(
            f"{end_x:.0f}",
            (end_x, end_y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )

        plotted += 1

    if plotted == 0:
        raise ValueError("No finite (neuron, loss) points to plot.")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Neuron counts are discrete; keep x-axis ticks as integers.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.25)
    if logy:
        ax.set_yscale("log")
    else:
        # Force plain scientific notation on y-axis (no offset like "1e-8 + 2.83e-1").
        # This keeps the standard scientific scaling (e.g. "1e-8") without rounding ticks.
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((0, 0))  # always use scientific notation
        fmt.set_useOffset(False)
        ax.yaxis.set_major_formatter(fmt)
    if legend:
        from matplotlib.lines import Line2D

        dot = Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="best iteration reached",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=10,
            linewidth=0,
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + [dot], labels + [dot.get_label()], frameon=False)
    fig.tight_layout()
    return fig, ax

