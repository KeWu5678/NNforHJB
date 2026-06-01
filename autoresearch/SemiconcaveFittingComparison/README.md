# Semiconcave Fitting Comparison

This experiment studies semiconcave fitting behavior across datasets.

Terminology:

- `PDPA_v1` is the semiconcave model family.
- `PDPA_v2` is the signed model family.
- Legacy outputs may contain the label `PDPA_v2_semiconcave`; keep that label
  as historical output text unless the producing code is renamed separately.

## Layout

- `data: VDP/` contains VDP reference activation sweeps and model
  comparisons.
- `data: pendulum/` contains PMP and transient pendulum swing-up comparisons.
  The previous separate pendulum folders now live here as study slices:
  `pdpa_v1_vs_pdpa_v2/` and `extended_semiconcave_runs/`.

Per-run JSON files, append-only scoreboards, and logs are local generated
artifacts and are ignored by Git.
