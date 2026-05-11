#!/usr/bin/env python3
"""Aggregate per-seed JSON files for an activation; append a row to results.tsv."""
from __future__ import annotations
import argparse, json, statistics, subprocess, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
TSV  = ROOT / "results.tsv"

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--activation", required=True)
    p.add_argument("--seeds", default="42,43,44,45,46")
    p.add_argument("--description", default="")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    rows, used = [], []
    for s in seeds:
        f = RUNS / f"{args.activation}_seed{s}.json"
        if not f.exists():
            continue
        try:
            line = f.read_text().strip().splitlines()[-1]
            rows.append(json.loads(line))
            used.append(s)
        except Exception as e:
            print(f"skip {f.name}: {e}", file=sys.stderr)

    if not rows:
        print(f"no valid runs for {args.activation}", file=sys.stderr)
        return 1

    scores = [r["best_score"] for r in rows]
    h1s    = [r["best_h1"]    for r in rows]
    ns     = [r["best_n"]     for r in rows]
    gammas = [r["best_gamma"] for r in rows]
    mode_g = Counter(gammas).most_common(1)[0][0]

    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT.parent
    ).decode().strip()

    status = "done" if len(rows) == len(seeds) else "partial"
    row = "\t".join([
        commit,
        args.activation,
        f"{rows[0]['power']}",
        rows[0]["loss"],
        ",".join(str(s) for s in used),
        f"{statistics.mean(scores):.4f}",
        f"{statistics.stdev(scores):.4f}" if len(scores) > 1 else "0.0000",
        f"{statistics.mean(h1s):.6f}",
        f"{statistics.mean(ns):.2f}",
        str(mode_g),
        status,
        args.description,
    ])
    with TSV.open("a") as f:
        f.write(row + "\n")
    print(row)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
