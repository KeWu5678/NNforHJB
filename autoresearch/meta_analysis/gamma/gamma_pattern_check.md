# Gamma Pattern Check

This checks the hypothesis that gamma has the largest effect when an activation
already uses many neurons or has a sharp inactive/active transition. The check
uses the generated gamma-effect tables and simple name-based activation classes.

Conclusion: the hypothesis is only locally true for the small VDP rerun set.
It is not a general pattern in the discontinuous-gradient experiment.

## VDP HJB

| effect metric | Pearson vs mean neurons | Spearman vs mean neurons |
|:--------------|------------------------:|-------------------------:|
| `accuracy_range` | 0.915 | 0.476 |
| `neuron_range` | 0.276 | 0.190 |
| `score_range` | 0.817 | 0.571 |

Median mean neurons: `45.0`. 75th percentile score range: `2.69`.

### Largest score-range effects

| activation | score range | accuracy range | neuron range | mean neurons | class |
|:-----------|------------:|---------------:|-------------:|-------------:|:------|
| `smoothy_relu_w0_25` | 20.8 | 0.147 | 1.8 | 141.3 | sharp_or_inactive |
| `relu` | 4.2 | 0.0759 | 19.2 | 121.0 | sharp_or_inactive |
| `softplus_b0_1` | 2.18 | 0.038 | 12.6 | 38.8 | other |
| `softplus_b0_5` | 1.52 | 0.0281 | 4.4 | 46.4 | other |
| `softplus_b0_25` | 1.5 | 0.0179 | 4.8 | 24.3 | other |
| `mish_b0_15` | 1.26 | 0.00645 | 5.2 | 43.6 | smooth_gated |
| `gelu_b0_25` | 0.916 | 0.0123 | 6.8 | 51.8 | smooth_gated |
| `softplus_b0_15` | 0.599 | 0.0167 | 1.2 | 32.0 | other |

### Counterexamples: high gamma effect without high neurons or sharp/inactive class

| activation | score range | accuracy range | neuron range | mean neurons | class |
|:-----------|------------:|---------------:|-------------:|-------------:|:------|
| none in this rerun set | | | | | |

### Counterexamples: high neurons or sharp/inactive class with low gamma effect

| activation | score range | accuracy range | neuron range | mean neurons | class |
|:-----------|------------:|---------------:|-------------:|-------------:|:------|
| `gelu_b0_25` | 0.916 | 0.0123 | 6.8 | 51.8 | smooth_gated |

## Discontinuous Gradient

| effect metric | Pearson vs mean neurons | Spearman vs mean neurons |
|:--------------|------------------------:|-------------------------:|
| `accuracy_range` | -0.135 | -0.103 |
| `neuron_range` | -0.237 | -0.088 |
| `score_range` | -0.070 | -0.074 |

Median mean neurons: `59.1`. 75th percentile score range: `2.17`.

### Largest score-range effects

| activation | score range | accuracy range | neuron range | mean neurons | class |
|:-----------|------------:|---------------:|-------------:|-------------:|:------|
| `asinh` | 25.3 | 0.448 | 4.2 | 53.6 | saturating |
| `softsign` | 15.7 | 0.0846 | 10.4 | 109.7 | saturating |
| `mish_b0_15` | 7.05 | 0.032 | 16.8 | 52.8 | smooth_gated |
| `atan` | 6.39 | 0.0823 | 14.6 | 55.8 | saturating |
| `cubic` | 6.38 | 4.27e-06 | 16.2 | 26.7 | polynomial_or_squared |
| `smoothy_relu_w0_05` | 5.66 | 0.044 | 4.2 | 147.0 | sharp_or_inactive |
| `swish_b0_25` | 5.32 | 0.0327 | 15.8 | 31.9 | smooth_gated |
| `gelu_b0_2` | 5.29 | 0.0607 | 21.6 | 40.3 | smooth_gated |
| `tanh` | 5.12 | 0.079 | 5.8 | 62.9 | saturating |
| `hardswish` | 4.61 | 0.00562 | 12.2 | 48.6 | smooth_gated |
| `mish_b0_1` | 3.9 | 0.0405 | 10.6 | 34.6 | smooth_gated |
| `sigmoid` | 3.58 | 0.0275 | 9 | 60.1 | saturating |

### Counterexamples: high gamma effect without high neurons or sharp/inactive class

| activation | score range | accuracy range | neuron range | mean neurons | class |
|:-----------|------------:|---------------:|-------------:|-------------:|:------|
| `asinh` | 25.3 | 0.448 | 4.2 | 53.6 | saturating |
| `mish_b0_15` | 7.05 | 0.032 | 16.8 | 52.8 | smooth_gated |
| `atan` | 6.39 | 0.0823 | 14.6 | 55.8 | saturating |
| `cubic` | 6.38 | 4.27e-06 | 16.2 | 26.7 | polynomial_or_squared |
| `swish_b0_25` | 5.32 | 0.0327 | 15.8 | 31.9 | smooth_gated |
| `gelu_b0_2` | 5.29 | 0.0607 | 21.6 | 40.3 | smooth_gated |
| `hardswish` | 4.61 | 0.00562 | 12.2 | 48.6 | smooth_gated |
| `mish_b0_1` | 3.9 | 0.0405 | 10.6 | 34.6 | smooth_gated |
| `silu` | 3.57 | 0.13 | 5.6 | 33.3 | smooth_gated |
| `softplus_b0_2` | 2.85 | 0.0257 | 9.4 | 48.8 | other |
| `softplus_b0_25` | 2.68 | 0.0274 | 10.2 | 44.9 | other |
| `elu2` | 2.34 | 0.00794 | 7 | 46.4 | polynomial_or_squared |

### Counterexamples: high neurons or sharp/inactive class with low gamma effect

| activation | score range | accuracy range | neuron range | mean neurons | class |
|:-----------|------------:|---------------:|-------------:|-------------:|:------|
| `x_absx` | 0.2 | 0.00262 | 2.3 | 125.4 | sharp_or_inactive |
| `leaky_relu2_a0_001_sphere` | 0.284 | 0.0031 | 3.4 | 104.6 | polynomial_or_squared, sharp_or_inactive |
| `leaky_relu2_a0_1_sphere` | 0.397 | 0.00339 | 3.5 | 108.7 | polynomial_or_squared, sharp_or_inactive |
| `smoothy_relu_w0_25` | 0.41 | 0.00521 | 2 | 147.7 | sharp_or_inactive |
| `leaky_relu2_a0_0375_sphere` | 0.484 | 0.00558 | 5.4 | 114.1 | polynomial_or_squared, sharp_or_inactive |
| `quartic` | 0.495 | 2.9e-06 | 0.8 | 79.2 | polynomial_or_squared |
| `leaky_relu2_a0_015_sphere` | 0.529 | 0.00586 | 2.5 | 115.1 | polynomial_or_squared, sharp_or_inactive |
| `leaky_relu2_a0_0625_sphere` | 0.611 | 0.00607 | 6 | 108.3 | polynomial_or_squared, sharp_or_inactive |
| `smoothy_relu_w0_125` | 0.626 | 0.00527 | 1 | 146.8 | sharp_or_inactive |
| `relu2` | 0.63 | 0.0047 | 4.5 | 104.9 | polynomial_or_squared, sharp_or_inactive |
| `leaky_relu2_a0_02_sphere` | 0.644 | 0.00512 | 5.5 | 114.9 | polynomial_or_squared, sharp_or_inactive |
| `leaky_relu2_a0_05_sphere` | 0.693 | 0.00368 | 3.8 | 117.8 | polynomial_or_squared, sharp_or_inactive |
