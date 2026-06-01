# Autoresearch

This directory contains experiment notes, analysis scripts, curated summaries,
and local generated outputs for activation/model studies.

## Layout

- `ActivationSearch/` - activation-function searches over multiple datasets:
  `data:VDP/` for smooth VDP HJB and `data:analytical/` for the analytic
  nonsmooth target.
- `SemiconcaveFittingComparison/` - semiconcave fitting comparisons across VDP
  reference and pendulum swing-up datasets.
- `meta_analysis/gamma/` - cross-experiment gamma-effect summaries and scripts.

## Artifact Policy

Track durable experiment materials: Markdown summaries, analysis scripts,
curated TSV summaries, and final figures. Keep bulk generated outputs local:
per-seed JSON runs, append-only scoreboards, logs, and regenerated caches are
ignored by Git.

Top-level runner entry points remain in `scripts/`. Experiment-local
aggregation and plotting helpers live in each experiment's `scripts/`
subdirectory.
