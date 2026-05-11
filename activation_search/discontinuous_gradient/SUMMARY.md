# Discontinuous-Gradient Activation Search - Summary

This folder repeats the activation search on the analytic Experiment 3 dataset
from `notebook/pdpa_vdp.ipynb`, where the value function is continuous but its
gradient jumps across

```text
h(x1, x2) = x1 + x2 * abs(x2) / 2 = 0.
```

The primary local metric is `near_grad`: relative gradient error on evaluation
points near the discontinuity. The notebook also uses
`near_grad / far_grad` as a localization diagnostic. The masks match the
notebook: near means distance to the switching curve below the 20th percentile,
and far means distance above the 50th percentile.

The detailed variant-level table is `results_near.tsv`. The summary below is
consolidated by activation family: each family is represented by its best
tested variant under `near_grad`.

## Family Leaders

| family | best variant | near grad | std | far grad | near/far | neurons |
|:-------|:-------------|----------:|----:|---------:|---------:|--------:|
| Leaky ReLU2 sphere | `alpha=0.02` | 0.151223 | 0.003271 | 0.017083 | 8.8520 | 120.10 |
| ReLU2 sphere | baseline | 0.157911 | 0.004934 | 0.017340 | 9.1068 | 106.50 |
| x\|x\| sphere | baseline | 0.164566 | 0.004347 | 0.021366 | 7.7024 | 128.20 |
| SmoothReLU | `w=0.05` | 0.168244 | 0.009455 | 0.055722 | 3.0193 | 149.40 |
| abs activation | baseline | 0.177244 | 0.007285 | 0.054753 | 3.2371 | 142.00 |
| ReLU | baseline | 0.182863 | 0.007877 | 0.057389 | 3.1864 | 144.00 |
| Leaky ReLU | baseline | 0.186197 | 0.003840 | 0.056432 | 3.2995 | 143.00 |
| Matérn 5/2 | baseline | 0.214069 | 0.008288 | 0.043474 | 4.9241 | 131.40 |
| ELU2 | `beta=0.5` | 0.222943 | 0.008986 | 0.039858 | 5.5935 | 55.20 |
| Gaussian | notebook form | 0.248449 | 0.009634 | 0.031426 | 7.9060 | 99.00 |

## Main Finding

After consolidation, the same conclusion remains: the best family for behavior
at the discontinuous gradient is **spherical leaky squared ReLU**,

```text
phi_alpha(x) = max(x, alpha*x)^2,
```

with the best tested value `alpha=0.02`.

The improvement is not just from using any kinked quadratic. The antisymmetric
quadratic `x|x|` is worse (`near_grad=0.164566`), and the same leaky squared
ReLU without spherical parameterization is much worse
(`near_grad=0.228469` for `alpha=0.05`).

## Variant Sweep

The leaky ReLU2 sphere family was tested over several leak values. The leading
region is narrow:

| variant | near grad | neurons | near score |
|:--------|----------:|--------:|-----------:|
| `alpha=0.02` | 0.151223 | 120.10 | 18.1617 |
| `alpha=0.025` | 0.151872 | 120.90 | 18.3463 |
| `alpha=0.05` | 0.152113 | 120.20 | 18.2869 |
| `alpha=0.075` | 0.154104 | 110.40 | 17.0111 |
| `alpha=0.10` | 0.155808 | 110.10 | 17.1509 |
| `alpha=0.01` | 0.156104 | 115.20 | 17.9812 |
| ReLU2, no leak | 0.157911 | 106.50 | 16.8143 |

So `alpha=0.02` is best for discontinuity accuracy, but `alpha=0.075` and
plain ReLU2 are stronger sparsity-aware choices.

## Near/Far Ratio

The near/far ratio is diagnostic, not an objective by itself. It measures where
the remaining gradient error lives. A high ratio can mean either:

- the activation fits smooth regions very well, so `far_grad` is tiny;
- or the activation fails badly near the discontinuity.

For the best squared-ReLU families, the high ratio is mostly the first case:
far-field errors are very small, and the remaining error is localized at the
jump. For example, `Leaky ReLU2 sphere (alpha=0.02)` has
`near_grad=0.151223`, `far_grad=0.017083`, and `near/far=8.8520`.

This differs from the notebook's Gaussian example, where the ratio was also
high but the near error stayed poor. In the expanded search, squared ReLU
variants reduce both near and far errors compared with Matérn and Gaussian.

## Sparsity Tradeoff

The best-accuracy family uses about 120 neurons. If sparsity matters, the
choice changes:

| choice | near grad | neurons | near score | comment |
|:-------|----------:|--------:|-----------:|:--------|
| Leaky ReLU2 sphere, `alpha=0.02` | 0.151223 | 120.10 | 18.1617 | best discontinuity accuracy |
| Leaky ReLU2 sphere, `alpha=0.075` | 0.154104 | 110.40 | 17.0111 | best leaky sparse compromise |
| ReLU2 sphere | 0.157911 | 106.50 | 16.8143 | sparsest good baseline |

The unconstrained `near_score = near_grad * neurons` is not reliable by itself:
very sparse low-rank activations such as `sp2_b0_2` and `qr_0_05` get low
scores only because they use 9-11 neurons, but their near-gradient errors are
around `0.39`, which is poor behavior at the discontinuity.

## When Gamma Matters Most

For this experiment, the main accuracy metric is `near_grad`, and the sparsity
metric is the mean selected neuron count. Gamma has the largest absolute effect
on noncompetitive smooth/saturating activations, not on the leading squared-ReLU
families.

| activation | near-grad range | neuron range | near-score range | largest step | interpretation |
|:-----------|----------------:|-------------:|-----------------:|:-------------|:---------------|
| `asinh` | 0.4484 | 4.2 | 25.30 | `0.01 -> 0.1` | largest absolute effect, but the fit is poor |
| `gelu_b0_2` | 0.0607 | 21.6 | 5.29 | `0 -> 0.01` | gamma strongly changes sparsity and accuracy |
| `smoothy_relu_w0_05` | 0.0440 | 4.2 | 5.66 | `1 -> 10` | gamma matters for near-discontinuity accuracy |
| `mish_b0_15` | 0.0320 | 16.8 | 7.05 | `0 -> 0.01` | strong sparsity/score effect, but not a top near-gradient fit |
| `softplus_b0_25` | 0.0274 | 10.2 | 2.68 | `1 -> 10` | gamma changes sparsity, still not competitive near the jump |
| Leaky ReLU2 sphere, `alpha=0.02` | 0.0051 | 5.5 | 0.64 | `1 -> 10` | best near accuracy is robust to gamma |
| Leaky ReLU2 sphere, `alpha=0.075` | 0.0081 | 3.2 | 1.37 | `1 -> 10` | sparse compromise is also robust |
| ReLU2 sphere | 0.0047 | 4.5 | 0.63 | `0 -> 0.01` | sparsest good baseline is robust |

This is the important distinction: gamma can move some weak activations a lot,
but it does not rescue them into the leading group. Among the useful
near-discontinuity fits, gamma changes the exact sparsity/accuracy point only
slightly; the activation shape, especially spherical squared ReLU with a small
leak, dominates the result.

This also means the VDP-side heuristic "gamma matters most for high-neuron or
sharp/inactive activations" does not generalize here. The generated
`../gamma_pattern_check.md` file lists counterexamples such as `asinh`,
`mish_b0_15`, `atan`, `swish_b0_25`, and `gelu_b0_2`, which have large
gamma-induced score changes without being high-neuron sharp/inactive fits.

Use `near_pareto.png` for the consolidated family plot with error bars and
`results_near.tsv` for the complete variant-level table.
