from __future__ import annotations
from typing import Any, Mapping, Optional, Sequence, Tuple
from pathlib import Path
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter


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


def plot_loss_vs_neurons_by_gamma(
    results: Sequence[dict] | str | os.PathLike[str],
    *,
    loss: str = "valid",
    metric: str = "loss",
    ax: Optional[plt.Axes] = None,
    logy: bool = False,
    legend: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot best-so-far metric vs neuron count over iterations.

    One line per gamma.  At each PDPA iteration *t* the plotted point is:
      x = neuron count at iteration t
      y = min(metric_hist[:t+1])

    Args:
        results: list of result dicts, or path to a pickle containing one.
        loss: ``"valid"`` or ``"train"`` — which split to use.
        metric: ``"loss"`` (objective), ``"err_l2"`` (relative L2), or
                ``"err_h1"`` (relative H1).
        ax: optional axes to draw on.
        logy: use log scale on y-axis.
        legend: show legend.
    """
    if loss not in {"valid", "train"}:
        raise ValueError(f"loss must be 'valid' or 'train', got {loss!r}")
    if metric not in {"loss", "err_l2", "err_h1"}:
        raise ValueError(f"metric must be 'loss', 'err_l2', or 'err_h1', got {metric!r}")

    suffix = "val" if loss == "valid" else "train"
    if metric == "loss":
        metric_key = f"{suffix}_loss" if suffix == "train" else "val_loss"
    else:
        metric_key = f"{metric}_{suffix}"

    _YLABEL = {"loss": "Best-so-far loss", "err_l2": "Relative L2 error", "err_h1": "Relative H1 error"}
    _TITLE = {"loss": "Loss vs neurons", "err_l2": "L2 error vs neurons", "err_h1": "H1 error vs neurons"}

    result_list = _load_results(results)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    for r in result_list:
        g = r["gamma"]
        hist = r.get(metric_key)
        if hist is None:
            continue
        metric_hist = np.asarray(hist, dtype=float).reshape(-1)
        safe = np.where(np.isfinite(metric_hist), metric_hist, np.inf)
        if safe.size == 0 or not np.any(np.isfinite(safe)):
            continue

        best_so_far = np.minimum.accumulate(safe)
        inner_weights = r["inner_weights"]
        T = min(len(best_so_far), len(inner_weights))

        neurons = np.array([inner_weights[t]["weight"].shape[0] for t in range(T)], dtype=float)
        values = best_so_far[:T]
        finite = np.isfinite(neurons) & np.isfinite(values)
        if not np.any(finite):
            continue

        xs, ys = neurons[finite], values[finite]
        ax.plot(xs, ys, marker="o", markersize=4, label=fr"$\gamma$={g:g}")
        ax.annotate(f"{xs[-1]:.0f}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(4, 4), fontsize=8, fontweight="bold")

    ax.set_xlabel("Neuron count")
    ax.set_ylabel(_YLABEL[metric])
    ax.set_title(_TITLE[metric])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.25)
    if logy:
        ax.set_yscale("log")
    else:
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((0, 0))
        fmt.set_useOffset(False)
        ax.yaxis.set_major_formatter(fmt)
    if legend:
        ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax
