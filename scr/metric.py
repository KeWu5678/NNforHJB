from __future__ import annotations
from typing import Any, Mapping, Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator




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


def plot_pdpa_val_loss_histories_by_gamma(
    model_dict: Mapping[str, Any],
    *,
    pdpa_key: str = "pdpa_list_h1",
    gammas_key: str = "gammas",
    title: str = "PDPA validation loss history by γ",
    xlabel: str = "Iteration",
    ylabel: str = "Validation loss",
    logy: bool = False,
    ax: Optional[plt.Axes] = None,
    legend: bool = True,
    linewidth: float = 1.8,
    alpha: float = 0.95,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot PDPA validation-loss histories for each gamma on a single figure.

    Expects a dict like in `notebook/pdpa_vdp.ipynb`:
        {
            "gammas": gammas,
            "pdpa_list_h1": pdpa_list_h1,
        }

    For each gamma, this plots the corresponding `pdpa.val_loss` sequence.

    Args:
        model_dict: dict-like with keys `gammas_key` and `pdpa_key`
        pdpa_key: key containing either a list/tuple aligned with gammas, or a dict mapping gamma -> PDPA
        gammas_key: key containing gamma values
        logy: if True, use log scale on y-axis
        ax: optional axes to plot into

    Returns:
        (fig, ax)
    """

    if gammas_key not in model_dict:
        raise KeyError(f"model_dict is missing key '{gammas_key}'. Available: {list(model_dict.keys())}")
    if pdpa_key not in model_dict:
        raise KeyError(f"model_dict is missing key '{pdpa_key}'. Available: {list(model_dict.keys())}")

    gammas = model_dict[gammas_key]
    pdpa_list_or_map = model_dict[pdpa_key]

    gammas_arr = np.asarray(gammas).reshape(-1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Resolve (gamma -> pdpa) for both common formats:
    # - list/tuple aligned with gammas
    # - dict mapping gamma -> pdpa
    if isinstance(pdpa_list_or_map, Mapping):
        def get_pdpa(i: int) -> Any:
            g = gammas_arr[i].item() if hasattr(gammas_arr[i], "item") else gammas_arr[i]
            if g in pdpa_list_or_map:
                return pdpa_list_or_map[g]
            # fallback: try stringified keys (e.g. JSON saved dicts)
            gs = str(g)
            if gs in pdpa_list_or_map:
                return pdpa_list_or_map[gs]
            raise KeyError(f"pdpa map has no entry for gamma={g!r}. Keys: {list(pdpa_list_or_map.keys())[:10]}")
    else:
        if not isinstance(pdpa_list_or_map, Sequence):
            raise TypeError(
                f"model_dict['{pdpa_key}'] must be a sequence (aligned with gammas) or a mapping gamma->pdpa; "
                f"got {type(pdpa_list_or_map)}"
            )
        if len(pdpa_list_or_map) != len(gammas_arr):
            raise ValueError(
                f"Length mismatch: len(gammas)={len(gammas_arr)} but len({pdpa_key})={len(pdpa_list_or_map)}"
            )

        def get_pdpa(i: int) -> Any:
            return pdpa_list_or_map[i]

    plotted = 0
    for i in range(len(gammas_arr)):
        gamma = gammas_arr[i]
        pdpa = get_pdpa(i)
        if not hasattr(pdpa, "val_loss"):
            raise AttributeError(f"PDPA object for gamma={gamma} has no attribute 'val_loss'")

        y = np.asarray(getattr(pdpa, "val_loss"), dtype=float).reshape(-1)
        if y.size == 0:
            continue

        x = np.arange(y.size)
        ax.plot(
            x,
            y,
            linewidth=linewidth,
            alpha=alpha,
            label=fr"$\gamma$={float(gamma):g}",
        )
        plotted += 1

    if plotted == 0:
        raise ValueError("No non-empty pdpa.val_loss histories found to plot.")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.25)
    if logy:
        ax.set_yscale("log")
    if legend:
        ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax

