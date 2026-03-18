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
    """3D scatter of inner weights (a1, a2, b) — one subplot per gamma.

    Each active neuron is a point at (a1, a2, b) in R^3.  Marker size is
    proportional to |u| and color encodes the signed outer weight u.

    Args:
        results: list of result dicts, or path to a pickle file.
        u_thresh: neurons with |u| <= u_thresh are excluded.
        elev, azim: 3D viewing angles.
        save_path: optional path to save the figure.
        show: call plt.show().

    Returns:
        (fig, axes_list) where axes_list contains the 3D Axes objects.
    """
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
    """Pairwise Euclidean distance heatmap of inner weights — one subplot per gamma.

    For each gamma the active neurons are sorted by descending |u|.  The
    (i, j) cell shows ||w_i - w_j||_2  where w_n = (a_n1, a_n2, b_n).

    Near-duplicate neurons appear as dark off-diagonal squares.  The
    diagonal annotation marks each neuron's |u| value (colorbar on side).

    Args:
        results: list of result dicts, or path to a pickle file.
        u_thresh: neurons with |u| <= u_thresh are excluded.
        save_path: optional path to save the figure.
        show: call plt.show().

    Returns:
        (fig, axes_list)
    """
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
    """Plot the learned value function V(x) as a 3D surface from a saved model pickle.

    Args:
        pkl_path: path to the pickle file containing experiment results.
        gamma_index: which gamma run to use (index into the results list).
        grid_n: number of grid points per axis.
        x_range: (min, max) for x[0]. If None, inferred from dataset or defaults to [-3, 3].
        y_range: (min, max) for x[1]. If None, inferred from dataset or defaults to [-3, 3].
        dataset: optional dataset dict with 'x' key, used to infer plot range.
        title: plot title. If None, auto-generated from gamma and neuron count.
        cmap: matplotlib colormap name.
        elev, azim: 3D view angles.
        save_path: optional path to save the figure.
        show: call plt.show().

    Returns:
        (fig, ax) where ax is a 3D axis.
    """
    import torch
    from src.net import ShallowNetwork

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
