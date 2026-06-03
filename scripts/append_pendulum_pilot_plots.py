"""Append pendulum pilot diagnostic plots to notebook/openloop.ipynb."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.paths import DATA_DIR, PLOTS_DIR  # noqa: E402

TAG = "pendulum_transient_pilot_7x7_T3_tol1e-5"
PLOT_DIR = PLOTS_DIR / TAG
NOTEBOOK_PATH = REPO_ROOT / "notebook" / "openloop.ipynb"

DATA_PATH = DATA_DIR / f"PENDULUM_transient_openloop_{TAG}.npy"
FAILED_PATH = DATA_DIR / f"PENDULUM_transient_openloop_failed_{TAG}.json"
DIAGNOSTICS_PATH = DATA_DIR / f"PENDULUM_transient_openloop_diagnostics_{TAG}.json"


def load_inputs():
    data = np.load(DATA_PATH)
    failed = json.loads(FAILED_PATH.read_text(encoding="utf-8"))
    diagnostics = json.loads(DIAGNOSTICS_PATH.read_text(encoding="utf-8"))
    return data, failed, diagnostics


def accepted_attempts_by_index(diagnostics: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for row in diagnostics:
        grouped.setdefault(int(row["index"]), []).append(row)
    return grouped


def best_attempt_rows(diagnostics: list[dict]) -> list[dict]:
    rows = []
    for _index, attempts in accepted_attempts_by_index(diagnostics).items():
        rows.append(min(attempts, key=lambda row: row["residual_l2_squared"]))
    return rows


def save_coverage_plot(data: np.ndarray, failed: list[dict]) -> Path:
    path = PLOT_DIR / "coverage.png"
    failed_x = np.array([row["x"] for row in failed], dtype=float) if failed else np.empty((0, 2))

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    sc = ax.scatter(data["x"][:, 0], data["x"][:, 1], c=data["v"], s=75, cmap="viridis")
    if failed_x.size:
        ax.scatter(
            failed_x[:, 0],
            failed_x[:, 1],
            marker="x",
            s=90,
            linewidths=2.0,
            color="crimson",
            label="failed tolerance",
        )
        ax.legend(loc="best")
    fig.colorbar(sc, ax=ax, label="value")
    ax.set_title("Pendulum pilot: accepted samples and failures")
    ax.set_xlabel("theta")
    ax.set_ylabel("omega")
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_value_plot(data: np.ndarray) -> Path:
    path = PLOT_DIR / "value_tricontour.png"
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    if data.shape[0] >= 4:
        contour = ax.tricontourf(data["x"][:, 0], data["x"][:, 1], data["v"], levels=18, cmap="viridis")
        fig.colorbar(contour, ax=ax, label="value")
    ax.scatter(data["x"][:, 0], data["x"][:, 1], s=35, color="black", alpha=0.65)
    ax.set_title("Pendulum pilot: value over accepted points")
    ax.set_xlabel("theta")
    ax.set_ylabel("omega")
    ax.grid(True, alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_gradient_plot(data: np.ndarray) -> Path:
    path = PLOT_DIR / "gradient_quiver.png"
    gradient = data["dv"]
    norm = np.linalg.norm(gradient, axis=1)
    scale = np.maximum(norm, 1.0)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    sc = ax.scatter(data["x"][:, 0], data["x"][:, 1], c=data["v"], s=55, cmap="viridis", alpha=0.85)
    ax.quiver(
        data["x"][:, 0],
        data["x"][:, 1],
        gradient[:, 0] / scale,
        gradient[:, 1] / scale,
        angles="xy",
        scale_units="xy",
        scale=2.8,
        width=0.004,
        color="black",
        alpha=0.8,
    )
    fig.colorbar(sc, ax=ax, label="value")
    ax.set_title("Pendulum pilot: normalized value-gradient arrows")
    ax.set_xlabel("theta")
    ax.set_ylabel("omega")
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_residual_plot(diagnostics: list[dict]) -> Path:
    path = PLOT_DIR / "best_residual_by_point.png"
    best_rows = best_attempt_rows(diagnostics)
    x = np.array([row["x"] for row in best_rows], dtype=float)
    residual = np.array([row["residual_l2_squared"] for row in best_rows], dtype=float)
    accepted = np.array([row["accepted"] for row in best_rows], dtype=bool)
    colors = np.log10(np.maximum(residual, 1e-16))

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    sc = ax.scatter(x[:, 0], x[:, 1], c=colors, s=75, cmap="magma_r")
    if np.any(~accepted):
        ax.scatter(
            x[~accepted, 0],
            x[~accepted, 1],
            marker="x",
            s=90,
            linewidths=2.0,
            color="cyan",
            label="best seed still failed",
        )
        ax.legend(loc="best")
    fig.colorbar(sc, ax=ax, label="log10 best integral G(t)^2 dt")
    ax.set_title("Pendulum pilot: best residual over all seeds")
    ax.set_xlabel("theta")
    ax.set_ylabel("omega")
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def append_notebook(plot_paths: list[Path], data: np.ndarray, failed: list[dict], diagnostics: list[dict]) -> None:
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)

    section_title = "## Pendulum Transient Open-Loop Pilot Diagnostics"
    nb.cells = [
        cell for cell in nb.cells
        if not (cell.cell_type == "markdown" and str(cell.source).startswith(section_title))
    ]

    accepted_attempts = [row for row in diagnostics if row["accepted"]]
    max_accepted_residual = max(row["residual_l2_squared"] for row in accepted_attempts)
    relative_paths = [
        Path(os.path.relpath(path, NOTEBOOK_PATH.parent)).as_posix()
        for path in plot_paths
    ]

    markdown = f"""{section_title}

Pilot data: `{DATA_PATH.relative_to(REPO_ROOT)}`.

Accepted samples: `{data.shape[0]}`. Failed grid points: `{len(failed)}`. Attempts: `{len(diagnostics)}`.

Acceptance rule: `integral G(t)^2 dt <= 1e-5`. Max accepted residual: `{max_accepted_residual:.6e}`.

![Accepted and failed samples]({relative_paths[0]})

![Value contour]({relative_paths[1]})

![Gradient arrows]({relative_paths[2]})

![Best residual by point]({relative_paths[3]})
"""

    code = f"""from pathlib import Path
import json
import numpy as np

repo = Path("..").resolve()
tag = "{TAG}"
data_path = repo / "rawdata/data" / f"PENDULUM_transient_openloop_{{tag}}.npy"
failed_path = repo / "rawdata/data" / f"PENDULUM_transient_openloop_failed_{{tag}}.json"
diagnostics_path = repo / "rawdata/data" / f"PENDULUM_transient_openloop_diagnostics_{{tag}}.json"

pendulum_data = np.load(data_path)
pendulum_failed = json.loads(failed_path.read_text())
pendulum_diagnostics = json.loads(diagnostics_path.read_text())

accepted_attempts = [row for row in pendulum_diagnostics if row["accepted"]]
print("accepted samples:", pendulum_data.shape[0])
print("failed grid points:", len(pendulum_failed))
print("attempts:", len(pendulum_diagnostics))
print("max accepted residual_l2_squared:", max(row["residual_l2_squared"] for row in accepted_attempts))
print("value range:", float(pendulum_data["v"].min()), float(pendulum_data["v"].max()))
"""

    nb.cells.append(nbformat.v4.new_markdown_cell(markdown))
    nb.cells.append(nbformat.v4.new_code_cell(code))
    nbformat.write(nb, NOTEBOOK_PATH)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    data, failed, diagnostics = load_inputs()
    plot_paths = [
        save_coverage_plot(data, failed),
        save_value_plot(data),
        save_gradient_plot(data),
        save_residual_plot(diagnostics),
    ]
    append_notebook(plot_paths, data, failed, diagnostics)
    print(f"wrote plots to {PLOT_DIR}")
    print(f"appended diagnostics to {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
