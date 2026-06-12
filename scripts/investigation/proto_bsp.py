#!/usr/bin/env python3
"""[DEBUG] Faithful transcription of github_contour_line_nosat.m bsp assembly.

Iterated in the fast loop (cached trajectories). Once the grade passes, this logic
is ported into src/OpenLoop/pendulum/nonsmooth.py. Throwaway.
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.affinity import translate
from shapely.prepared import prep

REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from scripts.investigation.diagnose_restriction_loop import CACHE
from src.OpenLoop.pendulum.nonsmooth import equal_value_contour

TWO_PI = 2 * np.pi


def polygon_crossings(contour: np.ndarray) -> np.ndarray:
    """SID==0 vertices: where the contour polygon boundary meets its +2pi shift."""
    p1 = Polygon(contour)
    if not p1.is_valid:
        p1 = p1.buffer(0)
    b = p1.boundary.intersection(translate(p1, xoff=TWO_PI).boundary)
    geoms = getattr(b, "geoms", [b]); out = []
    for g in geoms:
        if g.geom_type == "Point":
            out.append([g.x, g.y])
        elif g.geom_type == "MultiPoint":
            out.extend([[q.x, q.y] for q in g.geoms])
    return np.array(out) if out else np.empty((0, 2))


def track_arms(raw, levels):
    """onpoints tracking via optimal (Hungarian) 4->4 assignment, robust to swaps."""
    from scipy.optimize import linear_sum_assignment
    # Order trajectories by boundary angle so the contour is a proper ring.
    raw = sorted(raw, key=lambda t: t.boundary_angle)
    arms = None
    A = [[], [], [], []]
    for L in levels:
        contour = equal_value_contour(tuple(raw), L)
        if contour.shape[0] < 3:
            continue
        cv = polygon_crossings(contour)
        if len(cv) < 4:
            continue
        # Keep the 4 crossings closest to the (pi,0) collision (drop higher-order
        # spiral wraps that appear at large value).
        cv = cv[np.argsort(np.hypot(cv[:, 0] - np.pi, cv[:, 1]))][:4]
        if arms is None:
            ang = np.arctan2(cv[:, 1], cv[:, 0] - np.pi)
            arms = cv[np.argsort(ang)].copy()
        else:
            cost = np.linalg.norm(arms[:, None, :] - cv[None, :, :], axis=2)
            r, c = linear_sum_assignment(cost)
            na = arms.copy()
            for ri, ci in zip(r, c):
                na[ri] = cv[ci]
            arms = na
        for i in range(4):
            A[i].append(arms[i])
    return [np.array(a) for a in A]


def trim_erratic(arm: np.ndarray, factor: float = 6.0) -> np.ndarray:
    """Data-driven replacement for the reference's hard-coded boundary_spiral(1:1252)."""
    if len(arm) < 5:
        return arm
    step = np.linalg.norm(np.diff(arm, axis=0), axis=1)
    med = np.median(step)
    bad = np.where(step > factor * med)[0]
    return arm[: bad[0] + 1] if len(bad) else arm


def refl(P):  # reflect through (pi, 0): (theta,omega) -> (2pi-theta, -omega)
    return np.column_stack([TWO_PI - P[:, 0], -P[:, 1]])


def build_bsp(a_arm: np.ndarray, c_arm: np.ndarray) -> Polygon:
    part1 = np.vstack([refl(c_arm)[::-1], a_arm])
    part2 = np.vstack([part1, refl(part1)[::-1]])
    bsp = np.vstack([part2, np.column_stack([part2[:, 0] - TWO_PI, part2[:, 1]])[::-1]])
    poly = Polygon(bsp)
    return poly if poly.is_valid else poly.buffer(0)


def grade(raw, poly: Polygon):
    pp = prep(poly); kx = []; kv = []
    for t in raw:
        inside = np.fromiter((pp.contains(Point(p)) for p in t.state), bool, len(t.state))
        out = np.where(~inside)[0]
        cut = int(out[0]) if len(out) else len(t.state)
        kx.append(t.state[: max(cut, 1)]); kv.append(t.value[: max(cut, 1)])
    x = np.vstack(kx); v = np.concatenate(kv)
    return x, v


def main():
    raw = pickle.load(open(CACHE, "rb"))
    # Cap at value <= 35 (matches the reference's value-31 trim intent) instead of
    # the erratic-step trim, which over-cut the outer arms.
    arms = track_arms(raw, np.arange(26.0, 35.0, 0.25))
    print("arm lengths:", [len(a) for a in arms])

    # Try all ordered arm pairs. The correct origin basin is a SIMPLE polygon
    # (not MultiPolygon) containing the origin, whose boundary uses arm points up
    # to ~value 31 (reference trim) -> trajectories cut there have value_max ~26-35.
    # Leaky/degenerate basins let trajectories escape to value ~80-98. So select:
    # simple polygon, contains origin, value_max in [26,35], then largest area.
    best = None
    for ai in range(4):
        for ci in range(4):
            if ai == ci:
                continue
            poly = build_bsp(arms[ai], arms[ci])
            simple = poly.geom_type == "Polygon"
            if poly.is_empty or not poly.is_valid or not poly.contains(Point(0, 0)):
                continue
            x, v = grade(raw, poly)
            ok = 26.0 <= v.max() <= 35.0  # value-31 boundary; MultiPolygon allowed
            print(f"  a={ai} c={ci}: {'Polygon ' if simple else 'MULTI   '}"
                  f"area={poly.area:7.2f} retained={len(v):6d} "
                  f"value_max={v.max():6.2f} omega[{x[:,1].min():5.1f},{x[:,1].max():5.1f}]"
                  f"{'  <- candidate' if ok else ''}")
            if ok and (best is None or poly.area > best[3].area):
                best = (v.max(), ai, ci, poly, x, v)
    if best is None:
        print("NO valid origin-containing basin in the target value band")
        return 1
    score, ai, ci, poly, x, v = best
    # Restrict to the origin's own component (drop the +-2pi neighbor copies).
    if poly.geom_type == "MultiPolygon":
        poly = next(g for g in poly.geoms if g.contains(Point(0, 0)))
        x, v = grade(raw, poly)
    # Ridge-distance check: retained samples should now reach the switching set.
    from scipy.spatial import cKDTree
    ridge = np.vstack(arms)
    dmin = cKDTree(ridge).query(x, k=1)[0].min()
    print(f"  origin-component: area={poly.area:.2f} retained={len(v)} "
          f"value_max={v.max():.2f} min-dist-to-ridge={dmin:.3f}")
    print(f"  PASS (value>20 and reaches ridge): {v.max() > 20 and dmin < 0.3}")
    print(f"\nBEST a={ai} c={ci}: value_max={v.max():.2f} (want ~26-31), "
          f"retained={len(v)}, theta[{x[:,0].min():.1f},{x[:,0].max():.1f}] "
          f"omega[{x[:,1].min():.1f},{x[:,1].max():.1f}]")

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.scatter(x[::20, 0], x[::20, 1], s=2, c=v[::20], cmap="viridis")
    for part in getattr(poly, "geoms", [poly]):
        xb, yb = part.exterior.xy; ax.plot(xb, yb, "r-", lw=1.2)
    ax.scatter([0], [0], c="k", marker="*", s=200)
    ax.set_xlabel("theta"); ax.set_ylabel("omega"); ax.legend()
    ax.set_title(f"faithful bsp (a={ai},c={ci}) + inpolygon-retained samples")
    out = REPO / "rawdata" / "plots" / "debug_bsp_faithful.png"
    plt.savefig(out, dpi=110, bbox_inches="tight"); print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
