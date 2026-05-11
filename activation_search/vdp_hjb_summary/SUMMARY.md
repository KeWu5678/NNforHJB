# Activation Search - VDP HJB Summary

This folder summarizes the activation search on the smooth VDP HJB dataset
with `PDPA_v2` (`power=1`, `loss=h1`, gamma sweep
`[0, 0.01, 0.1, 1, 10]`). The original search evaluated 132 activation
variants. Since many adjacent ranks were just beta/width variants of the same
activation, the results below are consolidated by activation family.

The full old variant-level TSV was not present in this workspace copy, so the
new plot and softplus diagnostics are generated from reruns of the families
needed for the conclusion:

```text
softplus_b0_1, softplus_b0_15, softplus_b0_25, softplus_b0_5,
gelu_b0_25, mish_b0_15, smoothy_relu_w0_25, relu
```

The run JSONs are in `runs/`, and derived tables are in `analysis/`.

## Consolidated Family Ranking

One representative is shown per family, chosen by the best sparsity-aware score
`score = H1 x neurons` among the rerun variants.

| family | best variant | H1 | neurons | score | interpretation |
|:-------|:-------------|---:|--------:|------:|:---------------|
| Softplus | `beta=0.25` | 0.2916 | 22.6 | 6.57 | best sparsity/accuracy score |
| Mish | `beta=0.15` | 0.2112 | 36.4 | 7.62 | lower H1, more neurons and higher variance |
| GELU | `beta=0.25` | 0.1967 | 47.4 | 9.32 | better H1, roughly 2x neurons vs softplus |
| SmoothReLU | `w=0.25` | 0.1163 | 140.8 | 16.43 | best H1, poor sparsity and high variance |
| ReLU | baseline | 0.2117 | 110.8 | 23.21 | not competitive in sparsity |

This changes the headline from "many softplus/GELU variants occupy adjacent
ranks" to the family-level statement:

**Softplus is the best sparse family; GELU/Mish improve accuracy at higher
neuron count; SmoothReLU improves H1 most but is not sparse.**

See `pareto.png` for the consolidated family plot.

## Why Softplus Is Sparse

The softplus advantage is not just a guess about smoothness. I compared
measured activation-shape properties on `z in [-6, 6]` against the rerun
performance.

| activation | H1 | neurons | score | slope range | near-zero slope | negative slope | curvature shape |
|:-----------|---:|--------:|------:|:------------|----------------:|---------------:|:----------------|
| `softplus_b0_25` | 0.2916 | 22.6 | 6.57 | [0.182, 0.818] | 0.0% | 0.0% | broad, bounded |
| `gelu_b0_25` | 0.1967 | 47.4 | 9.32 | [-0.129, 1.129] | 0.15% | 24.9% | stronger, sign-changing tail |
| `mish_b0_15` | 0.2112 | 36.4 | 7.62 | [0.907, 1.000] | 0.0% | 0.0% | very weak curvature, almost linear |
| `relu` | 0.2117 | 110.8 | 23.21 | [0, 1] | 50.0% | 0.0% | hard kink, no smooth curvature |
| `smoothy_relu_w0_25` | 0.1163 | 140.8 | 16.43 | [0, 1.996] | 47.9% | 0.0% | very sharp local curvature |

What this verifies:

1. **No dead side.** Softplus has no near-zero slope on this range. ReLU and
   SmoothReLU have about half the range inactive, which helps sharp fitting but
   costs many neurons.

2. **Monotone, nonnegative derivative.** Softplus keeps positive slope
   everywhere. GELU has negative slope on about 25% of the range, so some
   inserted neurons can oppose the monotone gradient channel; it gets better H1
   but needs more neurons.

3. **Broad bounded curvature.** `softplus_b0_25` has moderate curvature
   (`curvature_peak=0.0625`) spread over the whole tested range. SmoothReLU has
   much sharper curvature (`curvature_peak=4.0`) and gets better H1, but the
   fit is expensive in neuron count.

4. **Not just smoothness.** Mish is smooth and monotone here, but its derivative
   is almost constant (`0.907-1.000`) and curvature is weak, so it behaves more
   like a mildly nonlinear identity. It improves H1 over softplus but loses the
   score comparison.

So the verified mechanism is:

**Softplus gives every neuron a usable gradient channel, avoids negative-slope
tails, and has enough smooth curvature to fit the smooth VDP value function
without requiring many inserted neurons.**

## Gamma Effect On Softplus

The penalty parameter `gamma` changes the selected sparsity/accuracy point. The
table below averages seeds 42-46.

### Softplus `beta=0.25`

| gamma | H1 | neurons | score |
|------:|---:|--------:|------:|
| 0 | 0.3031 | 27.8 | 8.21 |
| 0.01 | 0.3113 | 24.2 | 7.48 |
| 0.1 | 0.3092 | 23.0 | 7.09 |
| 1 | 0.2933 | 23.0 | 6.71 |
| 10 | 0.3105 | 23.4 | 7.22 |

For `beta=0.25`, increasing `gamma` from 0 to 1 removes about 5 neurons and
also improves H1 in this rerun. Past that, `gamma=10` keeps sparsity but loses
accuracy, so the score worsens.

### Softplus beta sweep, best gamma by score

| variant | best gamma | H1 | neurons | score |
|:--------|-----------:|---:|--------:|------:|
| `softplus_b0_1` | 1 | 0.2414 | 30.8 | 6.93 |
| `softplus_b0_15` | 0.01 | 0.2387 | 31.6 | 7.21 |
| `softplus_b0_25` | 1 | 0.2933 | 23.0 | 6.71 |
| `softplus_b0_5` | 0.01 | 0.2314 | 44.8 | 10.01 |

Interpretation:

- Smaller beta (`0.1-0.15`) improves H1 but uses more neurons.
- `beta=0.25` is the best sparse score because it uses far fewer neurons.
- Larger beta (`0.5`) improves H1 relative to `0.25`, but the neuron count
  nearly doubles, so score is worse.
- The optimal gamma is not universal; it interacts with beta. For the sparse
  `beta=0.25` point, `gamma=1` is best in this rerun.

## When Gamma Matters Most

In this small rerun set, gamma has the largest effect for high-neuron,
sharp/inactive activations (`SmoothReLU`, `ReLU`). This should be read as a
local observation for the VDP reruns, not a general rule across all activation
searches. The broader discontinuous-gradient experiment has clear
counterexamples; see `../gamma_pattern_check.md`.

| activation | accuracy range | neuron range | score range | largest step | interpretation |
|:-----------|---------------:|-------------:|------------:|:-------------|:---------------|
| `smoothy_relu_w0_25` | 0.1475 | 1.8 | 20.82 | `0.01 -> 0.1` | largest accuracy/score sensitivity; sparsity barely changes |
| `relu` | 0.0759 | 19.2 | 4.20 | `0 -> 0.01` | largest joint sparsity/accuracy change among baselines |
| `softplus_b0_1` | 0.0380 | 12.6 | 2.18 | `0.1 -> 1` | gamma can prune neurons for this narrower softplus |
| `softplus_b0_25` | 0.0179 | 4.8 | 1.50 | `0 -> 0.01` | winning sparse point; gamma only refines the result |

So, for the smooth VDP HJB experiment, the largest gamma effect is not on the
best sparse family. It appears most strongly in `smoothy_relu_w0_25` for
accuracy and in `relu`/`softplus_b0_1` for sparsity. For `softplus_b0_25`,
changing gamma from `0` to `1` improves the score, but the effect is smaller
than the activation-family and beta effects.

## Recommendation

Use the activation family depending on the objective:

| objective | choice |
|:----------|:-------|
| Best sparsity/accuracy score | `softplus_b0_25`, with gamma near `1` |
| Better H1 while still moderately sparse | `gelu_b0_25` or `mish_b0_15` |
| Lowest H1 regardless of sparsity | `smoothy_relu_w0_25` |
| Baseline comparison | `relu` |

The key caveat is that the old full 132-variant TSV is not available in this
workspace copy. The family consolidation in `pareto.png` is therefore based on
the rerun representative families above, while the earlier historical numbers
remain useful as context.
