# Pendulum Semiconcave Fitting - Summary

This dataset is the pendulum swing-up part of the semiconcave fitting
comparison. It evaluates PMP and transient datasets using `leaky_relu`.

## Direct `PDPA_v1` vs `PDPA_v2`

From `pdpa_v1_vs_pdpa_v2/results.tsv`, averaged over seeds 42-44:

| dataset | model | H1 | neurons | score |
|:--|:--|--:|--:|--:|
| PMP | `PDPA_v1` | 0.7226 | 99.3 | 71.78 |
| PMP | `PDPA_v2` | 0.5011 | 126.0 | 63.06 |
| transient | `PDPA_v1` | 0.3058 | 95.7 | 29.22 |
| transient | `PDPA_v2` | 0.3255 | 116.7 | 37.97 |

PMP favors `PDPA_v2`; transient favors `PDPA_v1` by the sparsity-aware score.

## Extended Semiconcave-Labeled Runs

From `extended_semiconcave_runs/results.tsv`, averaged over seeds 42-44:

| dataset | model | H1 | neurons | score |
|:--|:--|--:|--:|--:|
| PMP | `PDPA_v2` | 0.5011 | 126.0 | 63.06 |
| PMP | `PDPA_v2_semiconcave` | 0.6641 | 121.0 | 80.10 |
| transient | `PDPA_v2` | 0.3255 | 116.7 | 37.97 |
| transient | `PDPA_v2_semiconcave` | 0.3183 | 111.3 | 35.42 |

The semiconcave-labeled run is worse on PMP and slightly better on transient.
Keep this slice separate because it duplicates the `PDPA_v2` baseline rows and
uses legacy model labels.

## Files To Read

- `pdpa_v1_vs_pdpa_v2/results.tsv`: direct comparison rows.
- `extended_semiconcave_runs/results.tsv`: later semiconcave-labeled rerun.
- `pmp/results.tsv` and `transient/results.tsv` inside each slice: dataset-level
  views.
