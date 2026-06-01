# VDP Reference

This folder stores VDP reference activation and model comparison outputs for
the semiconcave fitting comparison.

Key summary tables:

- `v1_top_activation_vs_baselines_42_56.tsv` compares the strongest activation
  candidates with baselines over seeds 42-56.
- `same_activation_model_comparison_42_56.tsv` compares model families while
  controlling for activation choice.
- `model_comparison/results_42_56.tsv` contains per-seed model-comparison
  details.

Per-seed JSON files under `runs/` subdirectories are generated artifacts and
are ignored by Git.
