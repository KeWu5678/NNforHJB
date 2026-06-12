# region_split_pendulum

Profiles how well sparse PDAP models fit the pendulum value function in the
**smooth region** vs the **switching region** (near the nonsmooth swing-up
switching set). Built on the corrected pendulum dataset (the branch-restriction
fix, issue #18 / `experiments/pendulum_branch_restriction/`); without that fix the
samples never reached the switching set.

## Method

- **Data**: `data=pendulum`, which auto-selects `eval=region_split`
  (`conf/data/pendulum.yaml`). Each sample carries a precomputed distance to the
  switching set (`scripts/investigation/precompute_region_distances.py`).
- **Region split**: `near` = the lowest `near_percentile` (default 10) % of
  validation samples by distance to the switching set; `far` = the rest.
- **Sweep** (`make region_split_pendulum`): `model.kind ∈ {signed, semiconcave}` ×
  `activation ∈ {tanh, softplus, matern52, gaussian, gelu_squared}` ×
  `gamma ∈ {0, 1}`, profile insertion, H1 loss, α=1e-4, seed 42.
- **Output**: `results.md` (two tables + a diagnostic plot), via `analysis.py`.

## Choice of error metric

Measuring approximation error *near a discontinuity* and *where the reference
field vanishes* are both known to break the default metric. We validated the
choice against the literature rather than picking what fit our hypothesis.

**The default (region-local relative H1) is confounded here.** Relative error
`‖pred−true‖ / ‖true‖` is computed per region with that region's own denominator.
The `far` (smooth interior) region contains the upright equilibrium where `V→0`
and `∇V→0`, so its small `‖true‖` denominator inflates the relative error. Result:
every model looks *better* near the switching set — an artifact of where the small
values live, not of switching-set skill. Relative/percentage error is documented
as unreliable when the reference approaches zero (it can report huge error for an
accurate near-zero prediction); the standard remedies are absolute error there, or
excluding the near-zero region.¹

**Near a discontinuity, L1 is the appropriate norm.** For shock/discontinuity
problems the L1 norm gives the error "without significant contamination from
localized deviations," whereas L2 worst-case bounds "cannot be generalized … near
a shock."² So we report the region's **mean per-sample L1 (absolute) error,
normalized by the global mean `‖true‖`**: L1 is discontinuity-appropriate; the
*mean* (not the sum) keeps `near`/`far` **count-fair** — `near` is only ~10% of
samples, so a summed L1 would trivially favour it (a sum-based version reported
near/far ≈ 0.12 ≈ the 10/90 split, an artifact); and the shared global denominator
removes the per-region `V→0` confound. This is the **primary** table.

**The primary diagnostic is error vs distance to the switching set.** The standard
way to see behaviour near a kink is the spatial error profile — Gibbs-type error is
"most severe on nodes around the discontinuity, with error decreasing" with
distance.³ `analysis.py` plots per-sample absolute error (value + gradient) in
distance quantile bins, normalized by the model's own mean, so a ratio > 1 marks
worse-than-average accuracy at that distance and the curve is comparable across
activations.

We keep the relative-H1 table too, clearly labelled as confounded, so the
metric-choice effect is visible rather than hidden.

## Findings

Scored over the full dataset on the live (complete) model, with the count-fair
mean-L1 metric and a random train/val split — i.e. after correcting the data bug
(#18), the metric confounds, the sequential-split spatial bias, and the semiconcave
reconstruction bug (#19).

- **Sparse models are markedly less accurate near the switching set.**
  `near/far ≈ 2.8–5.8` across all activations and both model kinds (signed and
  semiconcave): the lowest-10%-distance band carries 3–6× the error of the rest.
- **The error profile is U-shaped in distance** (`figures/error_vs_distance.png`):
  high at the switching set (~2–3× the model mean), minimal in the mid-basin
  (~0.3–0.5), and rising again at the largest distances (~2.5–3×). The near/far
  ratio is large because `near` sits in the switching-set band while `far` (90%) is
  dominated by the well-fit mid-basin.
- **The result is metric- and pipeline-sensitive — every confound flipped it.**
  Relative H1 gave `near/far ≈ 0.3–0.6` and a sum-based L1 `≈ 0.12` (both said models
  are *better* at the switching set — artifacts of the V→0 interior denominator and
  the 10/90 count imbalance); scoring only on the sequential-split val tail (a
  clustered wedge spanning distance [0.6, 1.0]) erased the structure entirely; and
  reconstructing semiconcave models from the lossy `History` gave garbage. Only the
  corrected pipeline shows the true, consistent picture.
- **Second error rise at large distance** is a separate feature (the corridor
  far from the arms; likely the low-value LQR-bowl interior or basin extent) — worth
  a follow-up but not the switching-set question.

### Sources
1. [Forecast Evaluation for Data Scientists: Pitfalls and Best Practices](https://arxiv.org/pdf/2203.10716);
   [Regression Evaluation Metrics](https://programming-ocean.com/knowledge-hub/regression-evaluation-metrics-ai-atlas.php).
2. [Numerical smoothness and error analysis for RKDG on scalar nonlinear conservation laws](https://arxiv.org/pdf/1105.1393).
3. [Battling Gibbs Phenomenon: On Finite Element approximation of discontinuities](https://arxiv.org/pdf/1907.03429).
