# Semiconcave Fitting Comparison - Summary

This experiment asks whether a semiconcavity-aware model family improves sparse
HJB/value-function fitting compared with the signed model family.

Terminology in these results:

- `PDPA_v1` / `PDPA_v1_semiconcave`: semiconcave model family.
- `PDPA_v2` / `PDPA_v2_signed`: signed model family.
- `PDPA_v2_semiconcave`: legacy label from the pendulum rerun set; keep it as
  historical output text unless the runner is renamed separately.

## Datasets

- `data: VDP/`: VDP reference study. It includes activation sweeps for
  `PDPA_v1_semiconcave` and direct model comparisons against `PDPA_v2_signed`.
- `data: pendulum/`: pendulum swing-up study. It includes PMP and transient
  datasets, with a direct `PDPA_v1` vs `PDPA_v2` slice and a later
  semiconcave-labeled rerun slice.

## Main Read

On the VDP reference data, `PDPA_v1_semiconcave` is competitive and usually
more sparse-score efficient when activation is controlled. With `leaky_relu`,
the semiconcave model has lower mean H1 and fewer neurons than the signed model
(`score_mean=12.42` vs `16.54`).

On pendulum swing-up, semiconcavity is not uniformly helpful. Direct
`PDPA_v1` vs `PDPA_v2` favors `PDPA_v2` on PMP, while transient favors
`PDPA_v1` by score. The later semiconcave-labeled rerun is worse on PMP and
slightly better on transient.

Use the dataset summaries for details:

- `data: VDP/SUMMARY.md`
- `data: pendulum/SUMMARY.md`
