"""Append transient-phase diagnostics for the pendulum open-loop dataset."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat
import numpy as np
from scipy import ndimage
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator


REPO_ROOT = Path(__file__).resolve().parents[1]
TAG = "real_31x31_T3_tol1e-5_workers8"
DATA_DIR = REPO_ROOT / "rawdata" / "raw_data" / "data"
PLOT_DIR = REPO_ROOT / "rawdata" / "raw_data" / "plots" / TAG / "transient_phase_clean"
NOTEBOOK_PATH = REPO_ROOT / "notebook" / "openloop.ipynb"

DATA_PATH = DATA_DIR / f"PENDULUM_transient_openloop_{TAG}.npy"
FAILED_PATH = DATA_DIR / f"PENDULUM_transient_openloop_failed_{TAG}.json"
DIAGNOSTICS_PATH = DATA_DIR / f"PENDULUM_transient_openloop_diagnostics_{TAG}.json"
META_PATH = DATA_DIR / f"PENDULUM_transient_openloop_{TAG}_meta.json"
FIG_DPI = 145

PLOT_RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#777777",
    "axes.linewidth": 0.8,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "font.size": 9,
}


def load_inputs():
    data = np.load(DATA_PATH)
    failed = json.loads(FAILED_PATH.read_text(encoding="utf-8"))
    diagnostics = json.loads(DIAGNOSTICS_PATH.read_text(encoding="utf-8"))
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return data, failed, diagnostics, meta


def gridded_data(data: np.ndarray):
    theta_values = np.unique(data["x"][:, 0])
    omega_values = np.unique(data["x"][:, 1])
    shape = (omega_values.size, theta_values.size)

    p_theta = np.full(shape, np.nan)
    p_omega = np.full(shape, np.nan)
    value = np.full(shape, np.nan)

    theta_lookup = {round(float(theta), 12): idx for idx, theta in enumerate(theta_values)}
    omega_lookup = {round(float(omega), 12): idx for idx, omega in enumerate(omega_values)}
    for row in data:
        theta_idx = theta_lookup[round(float(row["x"][0]), 12)]
        omega_idx = omega_lookup[round(float(row["x"][1]), 12)]
        p_theta[omega_idx, theta_idx] = row["dv"][0]
        p_omega[omega_idx, theta_idx] = row["dv"][1]
        value[omega_idx, theta_idx] = row["v"]

    return theta_values, omega_values, p_theta, p_omega, value


def cell_edges(values: np.ndarray) -> np.ndarray:
    diffs = np.diff(values)
    return np.concatenate(
        ([values[0] - 0.5 * diffs[0]], values[:-1] + 0.5 * diffs, [values[-1] + 0.5 * diffs[-1]])
    )


def copy_cmap(name: str):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad("#ffffff")
    return cmap


def fill_missing_for_display(field: np.ndarray) -> np.ndarray:
    valid = np.isfinite(field)
    if valid.all() or not valid.any():
        return field
    nearest_indices = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return field[tuple(nearest_indices)]


def robust_symmetric_limits(*fields: np.ndarray, percentile: float = 98.0) -> tuple[float, float]:
    finite_parts = [field[np.isfinite(field)] for field in fields]
    finite = np.concatenate([part.ravel() for part in finite_parts if part.size])
    if not finite.size:
        return -1.0, 1.0
    limit = float(np.percentile(np.abs(finite), percentile))
    if limit <= 0.0:
        limit = float(np.max(np.abs(finite))) or 1.0
    return -limit, limit


def phase_axes(ax, theta: np.ndarray, omega: np.ndarray) -> None:
    ax.set_xlabel("theta")
    ax.set_ylabel("omega")
    ax.set_xlim(cell_edges(theta)[0], cell_edges(theta)[-1])
    ax.set_ylim(cell_edges(omega)[0], cell_edges(omega)[-1])
    ax.set_xticks([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi])
    ax.set_xticklabels(["-pi", "-pi/2", "0", "pi/2", "pi"])
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.tick_params(length=3, color="#777777")


def plot_grid_field(
    ax,
    theta: np.ndarray,
    omega: np.ndarray,
    field: np.ndarray,
    *,
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
):
    mesh = ax.pcolormesh(
        cell_edges(theta),
        cell_edges(omega),
        fill_missing_for_display(field),
        cmap=copy_cmap(cmap),
        vmin=vmin,
        vmax=vmax,
        shading="flat",
        rasterized=True,
    )
    ax.set_title(title)
    phase_axes(ax, theta, omega)
    return mesh


def save_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_paper_style_summary(data: np.ndarray) -> Path:
    path = PLOT_DIR / "finite_horizon_domain_summary.png"
    theta, omega, p_theta, p_omega, value = gridded_data(data)
    value_display = ndimage.gaussian_filter(fill_missing_for_display(value), sigma=0.55)
    p_theta_display = fill_missing_for_display(p_theta)
    p_omega_display = fill_missing_for_display(p_omega)
    control = -0.5 * p_omega_display
    jump = ndimage.gaussian_filter(fill_missing_for_display(neighbor_jump_score(data)), sigma=0.85)
    angle = ndimage.gaussian_filter(np.arctan2(p_omega_display, p_theta_display), sigma=0.7)
    control_smooth = ndimage.gaussian_filter(control, sigma=0.75)
    theta_grid, omega_grid = np.meshgrid(theta, omega)

    def interpolate_u(state: np.ndarray) -> float:
        wrapped_theta = float((state[0] + np.pi) % (2.0 * np.pi) - np.pi)
        clipped_omega = float(np.clip(state[1], omega[0], omega[-1]))
        return float(np.clip(u_interp((clipped_omega, wrapped_theta)), -12.0, 12.0))

    u_interp = RegularGridInterpolator(
        (omega, theta),
        control,
        bounds_error=False,
        fill_value=None,
    )

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        theta_state, omega_state = state
        u = interpolate_u(state)
        return np.array([omega_state, -0.1 * omega_state + 9.8 * np.sin(theta_state) + u])

    with plt.rc_context(PLOT_RC):
        fig = plt.figure(figsize=(9.6, 3.15))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.18, 1.0, 1.0])

        ax0 = fig.add_subplot(gs[0, 0], projection="3d")
        z_clip = float(np.percentile(value_display, 97.0))
        ax0.plot_surface(
            theta_grid,
            omega_grid,
            np.minimum(value_display, z_clip),
            cmap="viridis",
            linewidth=0,
            antialiased=True,
            rcount=31,
            ccount=31,
            alpha=0.96,
        )
        ax0.view_init(elev=26, azim=-58)
        ax0.set_title("Value surface on generated domain")
        ax0.set_xlabel("theta", labelpad=-1)
        ax0.set_ylabel("omega", labelpad=-1)
        ax0.set_xticks([-np.pi, 0.0, np.pi])
        ax0.set_xticklabels(["-pi", "0", "pi"])
        ax0.set_yticks([-4, 0, 4])
        ax0.set_zticks([0, round(z_clip / 2), round(z_clip)])
        ax0.tick_params(labelsize=7, pad=-2)
        ax0.grid(True, linewidth=0.3, alpha=0.35)

        ax1 = fig.add_subplot(gs[0, 1])
        ax1.contourf(theta_grid, omega_grid, angle, levels=12, cmap="Pastel1")
        ax1.contour(theta_grid, omega_grid, control_smooth, levels=[0.0], colors="#4b5563", linewidths=0.7)
        finite_jump = jump[np.isfinite(jump)]
        if finite_jump.size:
            levels = np.percentile(finite_jump, [88.0, 95.0])
            ax1.contour(theta_grid, omega_grid, jump, levels=levels, colors="#111827", linewidths=[0.85, 1.25])
        ax1.set_title("Inferred front on generated domain")
        phase_axes(ax1, theta, omega)
        ax1.set_aspect("auto")

        ax2 = fig.add_subplot(gs[0, 2])
        theta_initial = np.linspace(theta[0], theta[-1], 11)
        omega_initial = np.linspace(-3.5, 3.5, 7)
        colors = plt.get_cmap("turbo")(np.linspace(0.02, 0.95, theta_initial.size * omega_initial.size))
        color_idx = 0
        for theta0 in theta_initial:
            for omega0 in omega_initial:
                sol = solve_ivp(
                    rhs,
                    (0.0, 3.0),
                    np.array([theta0, omega0], dtype=float),
                    t_eval=np.linspace(0.0, 3.0, 100),
                    rtol=1e-5,
                    atol=1e-7,
                    max_step=0.04,
                )
                if sol.y.shape[1] > 1:
                    ax2.plot(sol.y[0], sol.y[1], color=colors[color_idx], linewidth=0.62, alpha=0.78)
                color_idx += 1
        ax2.set_title("Induced trajectories")
        phase_axes(ax2, theta, omega)
        ax2.grid(True, linewidth=0.35, alpha=0.28)
        ax2.set_aspect("auto")

        fig.subplots_adjust(left=0.035, right=0.99, bottom=0.19, top=0.86, wspace=0.34)
        save_figure(fig, path)
    return path


def save_gradient_components(data: np.ndarray) -> Path:
    path = PLOT_DIR / "gradient_components.png"
    theta, omega, p_theta, p_omega, _ = gridded_data(data)
    vmin, vmax = robust_symmetric_limits(p_theta, p_omega)

    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.05), constrained_layout=True)
        for ax, values, title in [
            (axes[0], p_theta, "p_theta = dV/dtheta"),
            (axes[1], p_omega, "p_omega = dV/domega"),
        ]:
            mesh = plot_grid_field(ax, theta, omega, values, title=title, cmap="RdBu_r", vmin=vmin, vmax=vmax)
            fig.colorbar(mesh, ax=ax, shrink=0.82, pad=0.02)
        save_figure(fig, path)
    return path


def save_feedback_control(data: np.ndarray) -> Path:
    path = PLOT_DIR / "feedback_control.png"
    theta, omega, _, p_omega, _ = gridded_data(data)
    control = -0.5 * p_omega
    vmin, vmax = robust_symmetric_limits(control)

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(4.65, 3.45), constrained_layout=True)
        mesh = plot_grid_field(
            ax,
            theta,
            omega,
            control,
            title="Initial feedback u0 = -0.5 p_omega",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )
        finite = control[np.isfinite(control)]
        fig.colorbar(mesh, ax=ax, label="u0", shrink=0.86, pad=0.02)
        save_figure(fig, path)
    return path


def save_gradient_direction(data: np.ndarray) -> Path:
    path = PLOT_DIR / "gradient_direction.png"
    theta, omega, p_theta, p_omega, _ = gridded_data(data)
    angle = np.arctan2(p_omega, p_theta)

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(4.65, 3.45), constrained_layout=True)
        mesh = plot_grid_field(
            ax,
            theta,
            omega,
            angle,
            title="Gradient direction angle",
            cmap="twilight",
            vmin=-np.pi,
            vmax=np.pi,
        )
        cbar = fig.colorbar(mesh, ax=ax, label="angle of ∇V", shrink=0.86, pad=0.02)
        cbar.set_ticks([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi])
        cbar.set_ticklabels(["-pi", "-pi/2", "0", "pi/2", "pi"])
        save_figure(fig, path)
    return path


def neighbor_jump_score(data: np.ndarray) -> np.ndarray:
    theta, omega, p_theta, p_omega, _ = gridded_data(data)
    score = np.full(p_theta.shape, np.nan)

    def update(omega_a: int, theta_a: int, omega_b: int, theta_b: int, dx: float) -> None:
        grad_a = np.array([p_theta[omega_a, theta_a], p_omega[omega_a, theta_a]])
        grad_b = np.array([p_theta[omega_b, theta_b], p_omega[omega_b, theta_b]])
        if not (np.all(np.isfinite(grad_a)) and np.all(np.isfinite(grad_b))):
            return
        local_score = float(np.linalg.norm(grad_a - grad_b) / max(dx, 1e-12))
        for omega_idx, theta_idx in [(omega_a, theta_a), (omega_b, theta_b)]:
            score[omega_idx, theta_idx] = np.nanmax([score[omega_idx, theta_idx], local_score])

    for omega_idx in range(omega.size):
        for theta_idx in range(theta.size):
            if theta_idx + 1 < theta.size:
                update(omega_idx, theta_idx, omega_idx, theta_idx + 1, theta[theta_idx + 1] - theta[theta_idx])
            if omega_idx + 1 < omega.size:
                update(omega_idx, theta_idx, omega_idx + 1, theta_idx, omega[omega_idx + 1] - omega[omega_idx])

    return score


def save_gradient_jump_indicator(data: np.ndarray) -> Path:
    path = PLOT_DIR / "gradient_jump_indicator.png"
    score = neighbor_jump_score(data)
    theta, omega, _, _, _ = gridded_data(data)
    log_score = np.full_like(score, np.nan)
    valid = np.isfinite(score) & (score > 0.0)
    log_score[valid] = np.log10(score[valid])
    finite = log_score[np.isfinite(log_score)]
    vmin, vmax = np.percentile(finite, [4.0, 98.0]) if finite.size else (-1.0, 1.0)

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(4.65, 3.45), constrained_layout=True)
        mesh = plot_grid_field(
            ax,
            theta,
            omega,
            log_score,
            title="Local grid jump indicator for ∇V",
            cmap="magma",
            vmin=float(vmin),
            vmax=float(vmax),
        )
        fig.colorbar(mesh, ax=ax, label="log10 local ||Delta ∇V|| / ||Delta x||", shrink=0.86, pad=0.02)
        save_figure(fig, path)
    return path


def save_omega_slices(data: np.ndarray) -> Path:
    path = PLOT_DIR / "omega_slices.png"
    theta, omega, _, p_omega, _ = gridded_data(data)
    selected = np.unique(np.round(np.linspace(0, omega.size - 1, 7)).astype(int))

    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(2, 1, figsize=(6.2, 4.15), sharex=True, constrained_layout=True)
        colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, selected.size))

        for color, omega_idx in zip(colors, selected):
            control = -0.5 * p_omega[omega_idx]
            label = f"omega={omega[omega_idx]:.2g}"
            axes[0].plot(theta, p_omega[omega_idx], color=color, linewidth=1.45, label=label)
            axes[1].plot(theta, control, color=color, linewidth=1.45)

        for ax in axes:
            ax.axhline(0.0, color="#444444", linewidth=0.75)
            ax.set_xlim(theta[0], theta[-1])
            ax.set_xticks([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi])
            ax.set_xticklabels(["-pi", "-pi/2", "0", "pi/2", "pi"])
            ax.tick_params(length=3, color="#777777")

        axes[0].set_title("Selected theta slices of p_omega")
        axes[1].set_title("Selected theta slices of u0")
        axes[1].set_xlabel("theta")
        axes[0].set_ylabel("p_omega")
        axes[1].set_ylabel("u0")
        axes[0].legend(ncol=4, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.32))
        save_figure(fig, path)
    return path


def append_notebook(
    plot_paths: list[Path],
    data: np.ndarray,
    failed: list[dict],
    diagnostics: list[dict],
    meta: dict,
) -> None:
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)

    old_titles = (
        "## Pendulum Transient Open-Loop Pilot Diagnostics",
        "## Pendulum Transient Phase / ∇V Diagnostics",
    )

    def is_old_pendulum_diagnostic_cell(cell) -> bool:
        source = str(cell.source)
        if cell.cell_type == "markdown":
            return any(source.startswith(title) for title in old_titles)
        if cell.cell_type != "code":
            return False
        return (
            "PENDULUM_transient_openloop_" in source
            and "pendulum_diagnostics" in source
            and (
                "pendulum_data = np.load(data_path)" in source
                or 'tag = "pendulum_transient_pilot_7x7_T3_tol1e-5"' in source
                or f'tag = "{TAG}"' in source
            )
        )

    nb.cells = [cell for cell in nb.cells if not is_old_pendulum_diagnostic_cell(cell)]

    relative_paths = [
        Path(os.path.relpath(path, NOTEBOOK_PATH.parent)).as_posix()
        for path in plot_paths
    ]
    accepted_attempts = [row for row in diagnostics if row["accepted"]]
    max_accepted_residual = max(row["residual_l2_squared"] for row in accepted_attempts)
    jump_score = neighbor_jump_score(data)
    grid_total = int(meta.get("accepted", data.shape[0])) + int(meta.get("failed", len(failed)))
    max_jump_score = float(np.nanmax(jump_score))

    markdown = f"""## Pendulum Transient Phase / ∇V Diagnostics

Data: `{DATA_PATH.relative_to(REPO_ROOT)}`.

This section is about the transient phase and possible discontinuity in `∇V`, not about data-generation failures. This is not the paper-domain Fig. 2 surface: the generated dataset only covers `theta in [-pi, pi]` and `omega in [-4, 4]`, while `on_nonsmooth_geometry.pdf` plots the PMP value geometry on approximately `[-10, 10] x [-10, 10]`. Rejected grid cells are filled from the nearest accepted cell for display only.

Accepted samples: `{data.shape[0]}` out of `{grid_total}`. Failed grid points: `{len(failed)}`. Max accepted `integral G(t)^2 dt`: `{max_accepted_residual:.6e}`.

<div style="max-width: 980px; margin: 0 0 14px 0;">
  <img src="{relative_paths[0]}" style="width: 100%; border: 1px solid #d8dde3;">
</div>

Largest local grid-neighbor jump score on this dataset: `{max_jump_score:.6e}`.

To reproduce the paper's Fig. 2 value surface, we need a new backward-PMP dataset on the larger `[-10, 10] x [-10, 10]` domain with nonsmooth curves computed from trajectory intersections; the current 31x31 finite-horizon training grid is insufficient for that plot.
"""

    code = f"""from pathlib import Path
import json
import numpy as np

repo = Path("..").resolve()
tag = "{TAG}"
data_path = repo / "rawdata/data" / f"PENDULUM_transient_openloop_{{tag}}.npy"
failed_path = repo / "rawdata/data" / f"PENDULUM_transient_openloop_failed_{{tag}}.json"
diagnostics_path = repo / "rawdata/data" / f"PENDULUM_transient_openloop_diagnostics_{{tag}}.json"
meta_path = repo / "rawdata/data" / f"PENDULUM_transient_openloop_{{tag}}_meta.json"

pendulum_data = np.load(data_path)
pendulum_failed = json.loads(failed_path.read_text())
pendulum_diagnostics = json.loads(diagnostics_path.read_text())
pendulum_meta = json.loads(meta_path.read_text())
accepted_attempts = [row for row in pendulum_diagnostics if row["accepted"]]

print("samples:", pendulum_data.shape[0])
print("failed grid points:", len(pendulum_failed))
print("value range:", float(pendulum_data["v"].min()), float(pendulum_data["v"].max()))
print("max accepted residual_l2_squared:", max(row["residual_l2_squared"] for row in accepted_attempts))
print("theta range:", float(pendulum_data["x"][:, 0].min()), float(pendulum_data["x"][:, 0].max()))
print("omega range:", float(pendulum_data["x"][:, 1].min()), float(pendulum_data["x"][:, 1].max()))
print("optimizer:", pendulum_meta["optimizer_method"])
"""

    nb.cells.append(nbformat.v4.new_markdown_cell(markdown))
    nb.cells.append(nbformat.v4.new_code_cell(code))
    nbformat.write(nb, NOTEBOOK_PATH)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    data, failed, diagnostics, meta = load_inputs()
    plot_paths = [
        save_paper_style_summary(data),
    ]
    append_notebook(plot_paths, data, failed, diagnostics, meta)
    print(f"wrote transient-phase plots to {PLOT_DIR}")
    print(f"updated diagnostics in {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
