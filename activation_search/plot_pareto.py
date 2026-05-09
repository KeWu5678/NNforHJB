#!/usr/bin/env python3
"""Plot h1 vs neurons for all evaluated activations, with Pareto frontier
computed on the meaningful-fit subset (h1 < FLOOR). Polynomial-floor cluster
(h1 >= FLOOR) is shown but shaded, and not used for the frontier or top-15."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
TSV  = ROOT / "results.tsv"
FLOOR = 0.30  # h1 above this is the polynomial-floor cluster

rows = {}
with open(TSV) as f:
    for r in csv.DictReader(f, delimiter='\t'):
        rows[r['activation']] = r
acts = list(rows.values())

points = [(float(r['mean_h1']), float(r['mean_neurons']), r['activation'],
           float(r['mean_score']), float(r['std_score'])) for r in acts]

# Pareto frontier on the meaningful subset only
meaningful = [p for p in points if p[0] < FLOOR]
points_sorted = sorted(meaningful)
frontier, best_n = [], float('inf')
for h1, n, name, s, std in points_sorted:
    if n < best_n:
        frontier.append((h1, n, name, s, std))
        best_n = n

highlights = {
    'smoothy_relu_w0_25': ('lowest h1', 'tab:red'),
    'gelu_b0_25':         ('best low-h1 + sparse', 'tab:orange'),
    'softplus_b0_25':     ('best balance', 'tab:green'),
    'relu':               ('relu baseline', 'tab:gray'),
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ---- Left: scatter with shaded floor zone ----
ax = axes[0]

# Shade polynomial-floor region
ax.axhspan(FLOOR, 1.0, color='tab:purple', alpha=0.08, zorder=0)
ax.axhline(FLOOR, color='tab:purple', linestyle=':', linewidth=1, alpha=0.5)
ax.text(155, FLOOR + 0.005, f'polynomial-floor zone (h1 ≥ {FLOOR})',
        fontsize=8, color='tab:purple', ha='right', va='bottom')

# All points
floor_pts = [p for p in points if p[0] >= FLOOR]
mean_pts  = [p for p in points if p[0] <  FLOOR]
ax.scatter([p[1] for p in floor_pts], [p[0] for p in floor_pts],
           c='tab:purple', s=18, alpha=0.45, label=f'floor cluster (n={len(floor_pts)}, e.g. qr_*, sp2_*)')
ax.scatter([p[1] for p in mean_pts],  [p[0] for p in mean_pts],
           c='lightgray', s=18, alpha=0.7, label=f'meaningful fits (n={len(mean_pts)})')

# Pareto frontier (on meaningful subset only)
fr_h1 = [p[0] for p in frontier]
fr_n  = [p[1] for p in frontier]
ax.plot(fr_n, fr_h1, 'k--', lw=1.2, alpha=0.7, label=f'Pareto frontier (h1<{FLOOR})')
ax.scatter(fr_n, fr_h1, c='black', s=30, zorder=3)

# Highlight points
for name, (label, color) in highlights.items():
    if name in rows:
        h1 = float(rows[name]['mean_h1'])
        n  = float(rows[name]['mean_neurons'])
        s  = float(rows[name]['mean_score'])
        ax.scatter(n, h1, c=color, s=120, edgecolors='black', linewidths=1.2,
                   zorder=5, label=f'{name} (score={s:.2f})')

# Score isocontours: h1 * n = const
ns = np.linspace(5, 160, 200)
for c in [3, 6, 10, 20, 50]:
    h1s = c / ns
    mask = (h1s >= 0.05) & (h1s <= 1.0)
    ax.plot(ns[mask], h1s[mask], 'b:', lw=0.6, alpha=0.4)
    if mask.any():
        idx = np.where(mask)[0][len(np.where(mask)[0])//2]
        ax.text(ns[idx], h1s[idx], f' s={c}', fontsize=7, color='blue', alpha=0.6)

ax.set_xlabel('mean neurons')
ax.set_ylabel('mean H1 loss')
ax.set_title(f'Activation search ({len(points)} activations) — Pareto on h1<{FLOOR}')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

# ---- Right: top-15 by score, filtered ----
ax2 = axes[1]
top15 = sorted(meaningful, key=lambda p: p[3])[:15]
names = [p[2] for p in top15]
scores = [p[3] for p in top15]
stds   = [p[4] for p in top15]
colors = ['tab:red' if 'smoothy' in n else
          'tab:orange' if 'gelu_b0_25' == n else
          'tab:green' if 'softplus_b0_25' == n else
          'lightblue' for n in names]
ypos = list(range(len(names)))
ax2.barh(ypos, scores, xerr=stds, color=colors, edgecolor='black', linewidth=0.5, capsize=3)
ax2.set_yticks(ypos)
ax2.set_yticklabels(names, fontsize=9)
ax2.invert_yaxis()
ax2.set_xlabel('score = H1 × neurons (mean ± std)')
ax2.set_title(f'Top 15 by score among meaningful fits (h1 < {FLOOR})')
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
out = ROOT / "pareto.png"
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f"saved {out}")
plt.close()
