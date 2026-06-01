# VDP Semiconcave Reference - Summary

This dataset is the VDP reference part of the semiconcave fitting comparison.
It compares semiconcave and signed model families and checks which activations
work well for the semiconcave model.

## Main Findings

The controlled same-activation comparison over seeds 42-56 favors
`PDPA_v1_semiconcave` for `leaky_relu` and `relu`, and is close for `abs_act`.

| comparison | H1 | neurons | score | near grad |
|:--|--:|--:|--:|--:|
| `PDPA_v1_semiconcave`, `leaky_relu` | 0.1117 | 111.0 | 12.42 | 0.0785 |
| `PDPA_v2_signed`, `leaky_relu` | 0.1261 | 131.4 | 16.54 | 0.0817 |
| `PDPA_v1_semiconcave`, `relu` | 0.1174 | 112.1 | 13.12 | 0.0827 |
| `PDPA_v2_signed`, `relu` | 0.1267 | 123.3 | 15.31 | 0.0817 |
| `PDPA_v1_semiconcave`, `abs_act` | 0.1138 | 116.2 | 13.23 | 0.0827 |
| `PDPA_v2_signed`, `abs_act` | 0.1110 | 128.1 | 14.15 | 0.0719 |

The best semiconcave activation among the top baseline comparison is
`leaky_relu`:

| label | H1 | neurons | score |
|:--|--:|--:|--:|
| `v1_leaky_relu` | 0.1117 | 111.0 | 12.42 |
| `v1_abs_act` | 0.1138 | 116.2 | 13.23 |
| `v1_relu` | 0.1174 | 112.1 | 13.12 |
| `v2_relu` | 0.1267 | 123.3 | 15.31 |

## Files To Read

- `v1_top_activation_vs_baselines_42_56.tsv`: concise top-activation
  comparison against the `PDPA_v2` baseline.
- `same_activation_model_comparison_42_56.tsv`: controlled model comparison for
  `abs_act`, `leaky_relu`, and `relu`.
- `model_comparison/results_42_56.tsv`: per-seed model-comparison details.
- `v1_convex_activation_sweep_aggregate.tsv`: aggregate activation sweep for
  the semiconcave family.
