#!/usr/bin/env python3
"""Plot discontinuity-focused activation rankings, consolidated by family."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TSV = ROOT / "results_near.tsv"

if not TSV.exists():
    raise SystemExit(f"missing {TSV}; run rank_discontinuity.py first")

with TSV.open() as file:
    rows = list(csv.DictReader(file, delimiter="\t"))


def decimal_suffix(text: str) -> str:
    return "0." + text.replace("_", "")


def family_name(name: str) -> str:
    if re.fullmatch(r"leaky_relu2(?:_a0_[0-9_]+)?_sphere", name):
        return "Leaky ReLU2 sphere"
    if re.fullmatch(r"leaky_relu2(?:_a0_[0-9_]+)?", name):
        return "Leaky ReLU2 no sphere"
    if name == "relu2":
        return "ReLU2 sphere"
    if name == "x_absx":
        return "x|x| sphere"
    if name.startswith("smoothy_relu") or name.startswith("smooth_relu"):
        return "SmoothReLU"
    if name.startswith("sp2_") or name.startswith("softplus_squared"):
        return "Softplus^2"
    if name.startswith("qr_"):
        return "Quadratic residual"
    if name.startswith("gelu_b"):
        return "GELU beta"
    if name.startswith("softplus_b"):
        return "Softplus beta"
    if name.startswith("mish_b"):
        return "Mish beta"
    if name.startswith("swish_b") or name.startswith("silu"):
        return "Swish/SiLU"
    if name.startswith("elu2"):
        return "ELU^2"
    if name.startswith("elu") or name == "celu":
        return "ELU/CELU"
    return name.replace("_", " ")


def variant_note(name: str) -> str:
    match = re.fullmatch(r"leaky_relu2_a0_([0-9_]+)_sphere", name)
    if match:
        return f"a={decimal_suffix(match.group(1))}"
    if name == "leaky_relu2_sphere":
        return "a=0.01"
    match = re.fullmatch(r"leaky_relu2_a0_([0-9_]+)", name)
    if match:
        return f"a={decimal_suffix(match.group(1))}"
    match = re.fullmatch(r"smoothy_relu_w(.+)", name)
    if match:
        return "w=" + match.group(1).replace("_", ".")
    match = re.fullmatch(r"gelu_b(.+)", name)
    if match:
        return "b=" + match.group(1).replace("_", ".")
    match = re.fullmatch(r"softplus_b(.+)", name)
    if match:
        return "b=" + match.group(1).replace("_", ".")
    match = re.fullmatch(r"mish_b(.+)", name)
    if match:
        return "b=" + match.group(1).replace("_", ".")
    return ""


def plot_label(row: dict[str, str]) -> str:
    family = row["family"]
    note = row["variant"]
    return f"{family} ({note})" if note else family


def frontier_label(label: str) -> str:
    replacements = {
        "Leaky ReLU2 sphere": "LReLU2",
        "Leaky ReLU2 no sphere": "LReLU2 no sphere",
        "ReLU2 sphere": "ReLU2",
        "Quadratic residual": "QR",
        "Softplus beta": "Softplus",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label.replace("Softplus^2", "SP^2")


typed_rows = []
for row in rows:
    item = dict(row)
    item["family"] = family_name(row["activation"])
    item["variant"] = variant_note(row["activation"])
    typed_rows.append(item)

family_reps: dict[str, dict[str, str]] = {}
for row in typed_rows:
    family = row["family"]
    if family not in family_reps or float(row["mean_near_grad"]) < float(
        family_reps[family]["mean_near_grad"]
    ):
        family_reps[family] = row

points = [
    (
        float(row["mean_near_grad"]),
        float(row["mean_neurons"]),
        row["activation"],
        plot_label(row),
        float(row["mean_near_score"]),
        float(row["std_near_score"]),
        float(row["near_far_ratio"]),
        float(row["std_near_grad"]),
    )
    for row in family_reps.values()
]

display_points = [point for point in points if point[0] <= 0.42]
frontier = []
best_near = float("inf")
for near, n, name, label, score, score_std, ratio, near_std in sorted(
    display_points, key=lambda point: point[1]
):
    if near < best_near:
        frontier.append((near, n, name, label, score, score_std, ratio, near_std))
        best_near = near

fig, axes = plt.subplots(
    1,
    2,
    figsize=(17, 7),
    gridspec_kw={"width_ratios": [1.05, 1.35], "wspace": 0.48},
)

ax = axes[0]
ratios = np.array([point[6] for point in display_points])
scatter = ax.scatter(
    [point[1] for point in display_points],
    [point[0] for point in display_points],
    c=ratios,
    cmap="magma",
    s=52,
    edgecolors="black",
    linewidths=0.4,
    alpha=0.9,
    zorder=3,
)
ax.step(
    [point[1] for point in frontier],
    [point[0] for point in frontier],
    where="post",
    color="black",
    linestyle="--",
    linewidth=1.3,
    label="family Pareto frontier",
    zorder=2,
)
ax.scatter(
    [point[1] for point in frontier],
    [point[0] for point in frontier],
    c="black",
    s=28,
    zorder=4,
)
for near, n, name, label, score, score_std, ratio, near_std in display_points:
    if any(name == point[2] for point in frontier):
        xytext = (5, 4)
        if name == "leaky_relu2_a0_02_sphere":
            xytext = (-80, 8)
        ax.annotate(
            frontier_label(label),
            (n, near),
            fontsize=8,
            xytext=xytext,
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.72},
        )
ax.set_xlabel("mean neurons")
ax.set_ylabel("mean near-discontinuity gradient error")
ax.set_title("Family representatives: near error vs sparsity")
ax.set_ylim(0.145, 0.405)
ns = np.linspace(5, max(point[1] for point in display_points) + 10, 220)
for score_level in [10, 16, 24, 36]:
    near_curve = score_level / ns
    mask = (near_curve >= 0.145) & (near_curve <= 0.405)
    if mask.any():
        ax.plot(ns[mask], near_curve[mask], "b:", lw=0.7, alpha=0.35)
        idxs = np.where(mask)[0]
        idx = idxs[len(idxs) // 2]
        ax.text(ns[idx], near_curve[idx], f"s={score_level}", fontsize=7, color="blue", alpha=0.6)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=8)
fig.colorbar(scatter, ax=ax, label="near/far gradient-error ratio")

ax2 = axes[1]
top = sorted(points, key=lambda point: point[0])[:12]
names = [point[3] for point in top]
near_values = [point[0] for point in top]
near_stds = [point[7] for point in top]
ypos = list(range(len(top)))
ax2.barh(
    ypos,
    near_values,
    xerr=near_stds,
    color="salmon",
    edgecolor="black",
    linewidth=0.5,
    capsize=3,
)
ax2.set_yticks(ypos)
ax2.set_yticklabels(names, fontsize=9)
ax2.invert_yaxis()
ax2.set_xlabel("mean near-discontinuity gradient error")
ax2.set_title("Top 12 activation families")
ax2.set_xlim(0.145, max(value + std for value, std in zip(near_values, near_stds)) + 0.006)
ax2.grid(True, axis="x", alpha=0.3)

out = ROOT / "near_pareto.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved {out}")
plt.close()
