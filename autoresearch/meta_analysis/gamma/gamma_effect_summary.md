# Gamma Effect Summary

Gamma effect is measured as the range of the mean metric across the fixed
gamma sweep `[0, 0.01, 0.1, 1, 10]`, averaged over available seeds.

## Main Interpretation

The largest gamma effects are not the same as the most useful gamma effects.
The pattern "gamma matters most for high-neuron or sharp/inactive activations"
is only supported in the small VDP rerun set. It is not a general rule: the
discontinuous-gradient experiment has high-effect counterexamples among
saturating and smooth-gated activations with below-median neuron counts.

For the smooth VDP HJB experiment, gamma matters most for `smoothy_relu_w0_25`
in accuracy/score and for `relu` in sparsity. The best sparse softplus variant
(`softplus_b0_25`) changes only modestly: H1 range `0.0179`, neuron range
`4.8`, and score range `1.50`.

For the discontinuous-gradient experiment, gamma has the largest absolute
near-gradient effect on weak smooth/saturating activations such as `asinh`,
`gelu_b0_2`, and `smoothy_relu_w0_05`. The leading squared-ReLU family is
stable: `leaky_relu2_a0_02_sphere` has near-gradient range `0.0051`, neuron
range `5.5`, and near-score range `0.64` across the full gamma sweep.

See `gamma_pattern_check.md` for the explicit correlation and counterexample
check.

## vdp_hjb

### Largest absolute effect on `h1`

| activation | min gamma/value | max gamma/value | range | largest adjacent |
|:-----------|:----------------|:----------------|------:|:-----------------|
| smoothy_relu_w0_25 | 0.01 / 0.19 | 0.1 / 0.3374 | 0.1475 | 0.01->0.1 |
| relu | 0.01 / 0.2243 | 0 / 0.3002 | 0.07591 | 0->0.01 |
| softplus_b0_1 | 0.1 / 0.2034 | 1 / 0.2414 | 0.03797 | 0.1->1 |
| softplus_b0_5 | 0.01 / 0.2314 | 0 / 0.2595 | 0.02809 | 0->0.01 |
| softplus_b0_25 | 1 / 0.2933 | 0.01 / 0.3113 | 0.01793 | 1->10 |
| softplus_b0_15 | 10 / 0.233 | 0 / 0.2497 | 0.01667 | 0->0.01 |
| gelu_b0_25 | 1 / 0.1956 | 0 / 0.2078 | 0.01227 | 0->0.01 |
| mish_b0_15 | 10 / 0.205 | 0.01 / 0.2115 | 0.006447 | 0.1->1 |

### Largest absolute effect on `neurons`

| activation | min gamma/value | max gamma/value | range | largest adjacent |
|:-----------|:----------------|:----------------|------:|:-----------------|
| relu | 0 / 107.8 | 0.1 / 127 | 19.2 | 0->0.01 |
| softplus_b0_1 | 1 / 30.8 | 0 / 43.4 | 12.6 | 0.1->1 |
| gelu_b0_25 | 0 / 48.8 | 0.1 / 55.6 | 6.8 | 0.01->0.1 |
| mish_b0_15 | 10 / 41.8 | 0 / 47 | 5.2 | 0->0.01 |
| softplus_b0_25 | 0.1 / 23 | 0 / 27.8 | 4.8 | 0->0.01 |
| softplus_b0_5 | 0.01 / 44.8 | 10 / 49.2 | 4.4 | 1->10 |
| smoothy_relu_w0_25 | 0 / 140.2 | 10 / 142 | 1.8 | 0->0.01 |
| softplus_b0_15 | 0.01 / 31.6 | 10 / 32.8 | 1.2 | 1->10 |

### Largest absolute effect on `score`

| activation | min gamma/value | max gamma/value | range | largest adjacent |
|:-----------|:----------------|:----------------|------:|:-----------------|
| smoothy_relu_w0_25 | 0.01 / 26.97 | 0.1 / 47.79 | 20.82 | 0.01->0.1 |
| relu | 10 / 28.18 | 0 / 32.38 | 4.203 | 0->0.01 |
| softplus_b0_1 | 1 / 6.926 | 0 / 9.111 | 2.185 | 0.1->1 |
| softplus_b0_5 | 0.01 / 10.01 | 10 / 11.53 | 1.522 | 0->0.01 |
| softplus_b0_25 | 1 / 6.715 | 0 / 8.214 | 1.499 | 0->0.01 |
| mish_b0_15 | 10 / 8.426 | 0 / 9.688 | 1.262 | 0.01->0.1 |
| gelu_b0_25 | 10 / 9.991 | 0.1 / 10.91 | 0.9159 | 0.01->0.1 |
| softplus_b0_15 | 0.01 / 7.215 | 0 / 7.814 | 0.5995 | 0->0.01 |

## discontinuous_gradient

### Largest absolute effect on `eval_h1`

| activation | min gamma/value | max gamma/value | range | largest adjacent |
|:-----------|:----------------|:----------------|------:|:-----------------|
| asinh | 0.01 / 0.4466 | 0.1 / 0.6474 | 0.2007 | 0.01->0.1 |
| atan | 10 / 0.2988 | 0.1 / 0.4417 | 0.1429 | 0.01->0.1 |
| tanh | 1 / 0.1769 | 0 / 0.275 | 0.09816 | 0.1->1 |
| softsign | 10 / 0.5333 | 0 / 0.6048 | 0.07157 | 1->10 |
| gelu_b0_2 | 0.01 / 0.09913 | 0 / 0.1443 | 0.04519 | 0->0.01 |
| smoothy_relu_w0_05 | 0.01 / 0.06028 | 10 / 0.1029 | 0.04261 | 1->10 |
| silu | 1 / 0.1368 | 0.01 / 0.1787 | 0.04185 | 0.01->0.1 |
| swish_b0_25 | 0 / 0.1108 | 10 / 0.1491 | 0.03826 | 0->0.01 |

### Largest absolute effect on `far_grad`

| activation | min gamma/value | max gamma/value | range | largest adjacent |
|:-----------|:----------------|:----------------|------:|:-----------------|
| atan | 10 / 0.4058 | 0.1 / 0.648 | 0.2421 | 0.01->0.1 |
| asinh | 0.01 / 0.6303 | 0.1 / 0.8332 | 0.2029 | 0.01->0.1 |
| tanh | 1 / 0.2141 | 0 / 0.3821 | 0.168 | 0.1->1 |
| softsign | 10 / 0.7184 | 0 / 0.835 | 0.1167 | 1->10 |
| swish_b0_25 | 0 / 0.07605 | 10 / 0.1673 | 0.09123 | 0->0.01 |
| gelu_b0_2 | 0.01 / 0.05878 | 0 / 0.1381 | 0.07928 | 0->0.01 |
| smoothy_relu_w0_05 | 0.01 / 0.05712 | 10 / 0.13 | 0.0729 | 1->10 |
| mish_b0_15 | 10 / 0.06813 | 1 / 0.1223 | 0.05414 | 1->10 |

### Largest absolute effect on `near_far_ratio`

| activation | min gamma/value | max gamma/value | range | largest adjacent |
|:-----------|:----------------|:----------------|------:|:-----------------|
| softplus_b0_2 | 0.01 / 5.432 | 0.1 / 8.5 | 3.068 | 0.01->0.1 |
| gelu_b0_2 | 0 / 3.253 | 0.01 / 6.113 | 2.86 | 0->0.01 |
| swish_b0_25 | 10 / 2.136 | 0 / 4.969 | 2.833 | 0->0.01 |
| leaky_relu2_a0_05 | 1 / 8.9 | 10 / 11.36 | 2.461 | 1->10 |
| softplus_b0_3 | 0 / 3.53 | 10 / 5.713 | 2.182 | 0->0.01 |
| elu2_b0_5 | 10 / 5.464 | 0 / 7.338 | 1.874 | 1->10 |
| gaussian | 10 / 6.623 | 1 / 8.371 | 1.749 | 1->10 |
| softplus_b0_25 | 0.01 / 4.856 | 1 / 6.527 | 1.67 | 1->10 |

### Largest absolute effect on `near_grad`

| activation | min gamma/value | max gamma/value | range | largest adjacent |
|:-----------|:----------------|:----------------|------:|:-----------------|
| asinh | 0.01 / 0.5816 | 0.1 / 1.03 | 0.4484 | 0.01->0.1 |
| silu | 1 / 0.3964 | 0.01 / 0.526 | 0.1296 | 0.01->0.1 |
| softsign | 10 / 0.6866 | 0.01 / 0.7712 | 0.0846 | 1->10 |
| atan | 1 / 0.5159 | 0.01 / 0.5982 | 0.08229 | 0.1->1 |
| tanh | 1 / 0.3813 | 10 / 0.4603 | 0.07904 | 1->10 |
| sra_b0_25 | 10 / 0.379 | 0.1 / 0.4507 | 0.07164 | 0.01->0.1 |
| gelu_b0_2 | 0.01 / 0.3199 | 0 / 0.3806 | 0.0607 | 0->0.01 |
| elu2_b0_5 | 10 / 0.226 | 0.01 / 0.2752 | 0.04916 | 1->10 |

### Largest absolute effect on `near_score`

| activation | min gamma/value | max gamma/value | range | largest adjacent |
|:-----------|:----------------|:----------------|------:|:-----------------|
| asinh | 0.01 / 31.4 | 0.1 / 56.7 | 25.3 | 0.01->0.1 |
| softsign | 10 / 72.62 | 0.01 / 88.37 | 15.75 | 1->10 |
| mish_b0_15 | 1 / 16.57 | 0 / 23.63 | 7.054 | 0->0.01 |
| atan | 0.1 / 27.57 | 0.01 / 33.95 | 6.387 | 0.01->0.1 |
| cubic | 10 / 7.556 | 0 / 13.93 | 6.375 | 0.01->0.1 |
| smoothy_relu_w0_05 | 0.01 / 25.53 | 10 / 31.19 | 5.664 | 1->10 |
| swish_b0_25 | 0.01 / 9.358 | 0 / 14.68 | 5.317 | 0->0.01 |
| gelu_b0_2 | 0 / 9.859 | 0.01 / 15.15 | 5.292 | 0->0.01 |

### Largest absolute effect on `neurons`

| activation | min gamma/value | max gamma/value | range | largest adjacent |
|:-----------|:----------------|:----------------|------:|:-----------------|
| gelu_b0_2 | 0 / 26.2 | 0.01 / 47.8 | 21.6 | 0->0.01 |
| mish_b0_15 | 1 / 45.4 | 0 / 62.2 | 16.8 | 0->0.01 |
| cubic | 10 / 19.2 | 0 / 35.4 | 16.2 | 0.01->0.1 |
| swish_b0_25 | 10 / 28.2 | 0 / 44 | 15.8 | 0->0.01 |
| atan | 0.1 / 47.6 | 10 / 62.2 | 14.6 | 0.01->0.1 |
| hardswish | 10 / 42 | 0 / 54.2 | 12.2 | 1->10 |
| elu2_b0_5 | 0.1 / 48.6 | 10 / 59.4 | 10.8 | 1->10 |
| mish_b0_1 | 0 / 29.6 | 0.01 / 40.2 | 10.6 | 0->0.01 |
