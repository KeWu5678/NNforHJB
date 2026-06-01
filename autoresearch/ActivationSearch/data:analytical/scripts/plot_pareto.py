#!/usr/bin/env python3
"""Plot eval H1 vs neurons for the discontinuous-gradient activation search."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TSV = ROOT / "results.tsv"

if not TSV.exists():
    raise SystemExit(f"missing {TSV}")

with TSV.open() as file:
    rows = list(csv.DictReader(file, delimiter="\t"))

if not rows:
    raise SystemExit(f"no rows in {TSV}")

points = [
    (
        float(row["mean_eval_h1"]),
        float(row["mean_neurons"]),
        row["activation"],
        float(row["mean_score"]),
        float(row["std_score"]),
        float(row["near_far_ratio"]),
    )
    for row in rows
]

frontier = []
best_n = float("inf")
for h1, n, name, score, std, ratio in sorted(points):
    if n < best_n:
        frontier.append((h1, n, name, score, std, ratio))
        best_n = n

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ratios = np.array([point[5] for point in points])
scatter = ax.scatter(
    [point[1] for point in points],
    [point[0] for point in points],
    c=ratios,
    cmap="viridis",
    s=42,
    edgecolors="black",
    linewidths=0.4,
)
ax.plot([point[1] for point in frontier], [point[0] for point in frontier], "k--", lw=1.2)
for h1, n, name, score, std, ratio in frontier:
    ax.annotate(name, (n, h1), fontsize=8, xytext=(4, 3), textcoords="offset points")
for const in [3, 6, 10, 20, 50]:
    ns = np.linspace(5, max(160, max(point[1] for point in points) * 1.05), 200)
    h1s = const / ns
    ax.plot(ns, h1s, "b:", lw=0.6, alpha=0.35)
ax.set_xlabel("mean neurons")
ax.set_ylabel("mean analytic eval H1")
ax.set_title("Discontinuous-gradient activation search")
ax.grid(True, alpha=0.3)
fig.colorbar(scatter, ax=ax, label="near/far gradient-error ratio")

ax2 = axes[1]
top = sorted(points, key=lambda point: point[3])[:15]
names = [point[2] for point in top]
scores = [point[3] for point in top]
stds = [point[4] for point in top]
ypos = list(range(len(top)))
ax2.barh(ypos, scores, xerr=stds, color="lightblue", edgecolor="black", linewidth=0.5, capsize=3)
ax2.set_yticks(ypos)
ax2.set_yticklabels(names, fontsize=9)
ax2.invert_yaxis()
ax2.set_xlabel("score = eval H1 * neurons")
ax2.set_title("Top 15 by score")
ax2.grid(True, axis="x", alpha=0.3)

plt.tight_layout()
out = ROOT / "pareto.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved {out}")
plt.close()
