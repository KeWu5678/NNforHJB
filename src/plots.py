#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Centralized plotting helpers for value-function / experiment visualization.

All figure-producing helpers live here. Tabular summaries live in
``src/metric.py``; the shared result loader ``_load_results`` is imported from
there to avoid duplication.
"""

import os
import logging
from typing import Any, Optional, Sequence, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_PLOT_CACHE = os.path.join(_REPO_ROOT, "rawdata", "logs", "matplotlib")
_XDG_CACHE = os.path.join(_REPO_ROOT, "rawdata", "logs", "cache")
os.makedirs(_PLOT_CACHE, exist_ok=True)
os.makedirs(_XDG_CACHE, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _PLOT_CACHE)
os.environ.setdefault("XDG_CACHE_HOME", _XDG_CACHE)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import torch

from .metric import _load_results

logger = logging.getLogger(__name__)


def plot_score_tradeoff(
    rows: Sequence[dict[str, Any]],
    *,
    x: str,
    y: str,
    label: str,
    color: str | None = None,
    title: str = "Score tradeoff",
    xlabel: str | None = None,
    ylabel: str | None = None,
    save_path: str | os.PathLike[str] | None = None,
    show_plot: bool = False,
):
    """Scatter of experiment summary rows (x vs y), labeled and optionally colored by a third column."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    if color:
        color_values = [str(row.get(color, "")) for row in rows]
        categories = list(dict.fromkeys(color_values))
        cmap = plt.get_cmap("tab10")
        palette = {cat: cmap(i % 10) for i, cat in enumerate(categories)}
        colors = [palette[value] for value in color_values]
    else:
        categories = []
        palette = {}
        colors = "#4c78a8"

    xs = [float(row[x]) for row in rows]
    ys = [float(row[y]) for row in rows]
    ax.scatter(xs, ys, s=52, c=colors, edgecolor="white", linewidth=0.7, zorder=3)
    for row, x_val, y_val in zip(rows, xs, ys):
        ax.annotate(str(row.get(label, "")), (x_val, y_val), xytext=(5, 4),
                    textcoords="offset points", fontsize=7)
    if categories:
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[cat],
                       markeredgecolor="white", markersize=7, label=f"{color}={cat}")
            for cat in categories
        ]
        ax.legend(handles=handles, fontsize=8, frameon=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    if save_path is not None:
        save_path = os.fspath(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def _get_field(dataset: Any, name: str) -> np.ndarray:
    """Extract a field from a structured array or dict-like dataset."""
    if isinstance(dataset, np.ndarray) and dataset.dtype.fields is not None:
        if name not in dataset.dtype.fields:
            raise KeyError(f"Dataset is missing field '{name}'. Available: {list(dataset.dtype.fields.keys())}")
        return dataset[name]
    if hasattr(dataset, "keys") and hasattr(dataset, "__getitem__"):
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
    """3D scatter of dataset samples (x[0], x[1], V), colored by value."""
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
    """2D scatter of x colored by V, with ∇V arrows sampled on a regular grid."""
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


def _extract_active_weights(run: dict, u_thresh: float = 1e-4) -> dict:
    """Return active (a, b, u) at the best iteration of one run.

    Active means |u| > u_thresh.  Returns a dict with keys:
        'a'     : np.ndarray (n_active, d)
        'b'     : np.ndarray (n_active,)
        'u'     : np.ndarray (n_active,)
        'gamma' : float
    """
    it = run["best_iteration"]
    iw = run["inner_weights"][it]
    a = np.asarray(iw["weight"])               # (n, d)
    b = np.asarray(iw["bias"])                 # (n,)
    u = np.asarray(run["outer_weights"][it]).flatten()  # (n,)
    mask = np.abs(u) > u_thresh
    return {"a": a[mask], "b": b[mask], "u": u[mask], "gamma": run["gamma"]}


def plot_inner_weight_3d_scatter(
    results: "Sequence[dict] | str | os.PathLike[str]",
    *,
    u_thresh: float = 1e-4,
    elev: float = 25.0,
    azim: float = 45.0,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, list]:
    """3D scatter of active inner weights (a₁, a₂, b) per gamma; size ∝ |u|, color = u."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    result_list = _load_results(results)
    n = len(result_list)
    fig = plt.figure(figsize=(4 * n, 4.5))
    axes: list = []

    for col, run in enumerate(result_list):
        rec = _extract_active_weights(run, u_thresh)
        a, b, u, gamma = rec["a"], rec["b"], rec["u"], rec["gamma"]

        ax = fig.add_subplot(1, n, col + 1, projection="3d")
        axes.append(ax)

        if len(u) == 0:
            ax.set_title(f"gamma={gamma:g}\n(no active neurons)")
            continue

        sizes = (np.abs(u) / np.abs(u).max()) * 120 + 10
        sc = ax.scatter(
            a[:, 0], a[:, 1], b,
            s=sizes, c=u, cmap="RdBu_r",
            vmin=-np.abs(u).max(), vmax=np.abs(u).max(),
            alpha=0.85, edgecolors="k", linewidths=0.3,
        )
        fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.55, label="$u$")
        ax.set_xlabel("$a_1$", labelpad=1)
        ax.set_ylabel("$a_2$", labelpad=1)
        ax.set_zlabel("$b$",   labelpad=1)
        ax.set_title(f"$\\gamma={gamma:g}$\n({len(u)} active)", fontsize=9)
        ax.view_init(elev=elev, azim=azim)

    fig.suptitle(
        "Inner weights $(a_1, a_2, b)$ — 3D scatter\n"
        "(size $\\propto |u|$, color $= u$)",
        fontsize=10,
    )
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def plot_inner_weight_pairwise_distance(
    results: "Sequence[dict] | str | os.PathLike[str]",
    *,
    u_thresh: float = 1e-4,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, list]:
    """Heatmap of pairwise ‖wᵢ - wⱼ‖ among active neurons (sorted by |u|), one subplot per gamma."""
    result_list = _load_results(results)
    n = len(result_list)
    fig, axes_list = plt.subplots(1, n, figsize=(4 * n, 4))

    if n == 1:
        axes_list = [axes_list]

    for ax, run in zip(axes_list, result_list):
        rec = _extract_active_weights(run, u_thresh)
        a, b, u, gamma = rec["a"], rec["b"], rec["u"], rec["gamma"]

        if len(u) == 0:
            ax.set_title(f"gamma={gamma:g}\n(no active neurons)")
            continue

        # sort by descending |u| so the most important neurons come first
        order = np.argsort(-np.abs(u))
        a_s, b_s, u_s = a[order], b[order], u[order]

        w = np.column_stack([a_s, b_s])                          # (n, 3)
        D = np.linalg.norm(w[:, None, :] - w[None, :, :], axis=-1)  # (n, n)

        im = ax.imshow(D, cmap="viridis_r", aspect="auto",
                       vmin=0, vmax=D.max())
        fig.colorbar(im, ax=ax, shrink=0.85, label="$\\|w_i - w_j\\|_2$")

        n_act = len(u)
        ax.set_xticks(range(n_act))
        ax.set_yticks(range(n_act))
        if n_act <= 20:
            ax.set_xticklabels([f"{v:.2f}" for v in u_s], rotation=90, fontsize=6)
            ax.set_yticklabels([f"{v:.2f}" for v in u_s], fontsize=6)
        else:
            ax.tick_params(labelsize=6)
        ax.set_xlabel("neuron index (sorted by $|u|$ desc)")
        ax.set_ylabel("neuron index")
        ax.set_title(f"$\\gamma={gamma:g}$  ({n_act} active)", fontsize=9)

    fig.suptitle(
        "Pairwise Euclidean distance $\\|w_i - w_j\\|_2$  in $(a_1, a_2, b)$ space\n"
        "(neurons sorted by $|u|$ descending — dark = close)",
        fontsize=10,
    )
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes_list


def plot_model_value_surface(
    pkl_path: str | os.PathLike[str],
    *,
    gamma_index: int = 0,
    grid_n: int = 100,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    dataset: Any = None,
    title: Optional[str] = None,
    cmap: str = "viridis",
    elev: float = 30.0,
    azim: float = 45.0,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, Any]:
    """3D surface of the learned V(x) evaluated on a 2D grid, loaded from a result pickle."""
    from src.models.net import ShallowNetwork

    result_list = _load_results(pkl_path)
    if gamma_index >= len(result_list):
        raise IndexError(f"gamma_index={gamma_index} but only {len(result_list)} runs in file")
    run = result_list[gamma_index]

    # Extract best-iteration weights
    best_it = run["best_iteration"]
    iw = run["inner_weights"][best_it]
    a = np.asarray(iw["weight"])       # (n_neurons, d)
    b = np.asarray(iw["bias"])         # (n_neurons,)
    u = np.asarray(run["outer_weights"][best_it])  # (1, n_neurons)

    n_neurons, d = a.shape
    if d != 2:
        raise ValueError(f"Expected 2D input, got d={d}. This function only supports 2D plots.")

    activation = run.get("activation", torch.relu)
    power = run.get("power", 1.0)

    # Build network and load weights
    net = ShallowNetwork(
        layer_sizes=[d, n_neurons, 1],
        activation=activation,
        p=power,
        inner_weights=a,
        inner_bias=b,
        outer_weights=u,
    )
    net.eval()

    # Determine grid range
    if x_range is None or y_range is None:
        if dataset is not None:
            x_data = np.asarray(_get_field(dataset, "x"))
            if x_range is None:
                x_range = (float(x_data[:, 0].min()), float(x_data[:, 0].max()))
            if y_range is None:
                y_range = (float(x_data[:, 1].min()), float(x_data[:, 1].max()))
        else:
            x_range = x_range or (-3.0, 3.0)
            y_range = y_range or (-3.0, 3.0)

    # Evaluate on grid
    x0 = np.linspace(x_range[0], x_range[1], grid_n)
    x1 = np.linspace(y_range[0], y_range[1], grid_n)
    X0, X1 = np.meshgrid(x0, x1)
    grid_points = np.column_stack([X0.ravel(), X1.ravel()])  # (grid_n^2, 2)

    with torch.no_grad():
        V = net(torch.tensor(grid_points, dtype=torch.float64)).numpy().reshape(grid_n, grid_n)

    # Plot
    gamma = run.get("gamma", None)
    if title is None:
        title = f"Learned V(x) — gamma={gamma:g}, {n_neurons} neurons" if gamma is not None else f"Learned V(x) — {n_neurons} neurons"

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X0, X1, V, cmap=cmap, alpha=0.9, edgecolor="none")
    ax.set_xlabel("x[0]")
    ax.set_ylabel("x[1]")
    ax.set_zlabel("V(x)")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_loss_vs_neurons_by_gamma(
    results: Sequence[dict] | str | os.PathLike[str],
    *,
    loss: str = "valid",
    metric: str = "loss",
    ax: Optional[plt.Axes] = None,
    logy: bool = False,
    legend: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Best-so-far metric vs neuron count over PDAP iterations, one line per gamma."""
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

