# activationsearch

Sweeps activation functions across both datasets (VDP, pendulum) and all model
configurations to identify which activation families best balance H1 accuracy
and neuron-count sparsity under the PDAP framework.

## Sweep axes

| axis | values |
|---|---|
| `data` | vdp, pendulum |
| `model.kind` | signed, semiconcave |
| `model.insertion` | profile, finite\_step |
| `model.activation` | see table below |
| `model.alpha` | 1e-2, 1e-3, 1e-4, 1e-5 |
| `model.gamma` | 0, 0.1, 1, 10 |
| `model.loss_weights` | [1,0] (H1 only), [1,1] (H1+L2) |

Fixed: `model.power=1.0`, `data.normalize=true`, `env.seed=42`.

Score metric: `H1 × neurons` (lower is better); primary ranking uses H1-only
loss rows.

## Activation functions

| activation | genre | rationale |
|---|---|---|
| `tanh` | S-shaped saturating | Classical baseline. Smooth, bounded, antisymmetric. Its derivative is concentrated near 0, giving each neuron a localized gradient channel. Competitive on pendulum in prior runs. |
| `softplus` | smooth monotone one-sided | Best sparse family in the VDP autoresearch (132-variant search): `softplus_b0_25` scored 6.57 (H1×neurons), better than GELU, Mish, and SmoothReLU. Key properties: no dead side, monotone nonneg derivative, broad bounded curvature. The canonical sparse-accuracy reference. |
| `matern52` | smooth radial decay | Matérn 5/2 kernel function: (1+√5\|z\|+5/3 z²)exp(-√5\|z\|). Encodes the RKHS smoothness prior that matches the regularity assumed by the gradient-augmented regression theory. Non-monotone, non-homogeneous — the only kernel-theoretic choice in the sweep. |
| `gaussian` | localized radial bump | exp(-z²): localized, non-monotone, symmetric. Provides a benchmark for localized basis functions. Underperformed in the analytical discontinuous-gradient search (near\_grad 0.248) but is competitive on smooth problems; kept to diagnose dataset-dependent behavior. |
| `gelu_squared` | squared-smooth | GELU(x)² inherits a one-sided near-kinked structure from GELU while doubling the polynomial degree at the origin. ELU² was tested at one β in the analytical search (near\_grad 0.223, not competitive); GELU²/SiLU² use a smoother base and may give the ReLU²-sphere sparsity advantage on smooth problems without requiring sphere sampling. |
| `silu_squared` | squared-smooth | SiLU(x)² — same genre as `gelu_squared`. SiLU has a slightly different negative-slope tail than GELU; testing both identifies whether the advantage (if any) is base-function-specific or a general property of squaring a smooth one-sided activation. |
| `rcip_2` | rational bounded | x/(1+x²): the only bounded, two-sided, non-saturating family untested in any prior search. Rational activations have algebraic rather than exponential asymptotic decay. Bounded derivative prevents the large-activation blow-up that can hurt sparsity under the H1 norm. |
| `snake_b0_25` | periodic | snake(x) = x + sin²(0.25x)/0.25: monotone trend with periodic curvature modulation. Completely untested. Motivated by the hypothesis that HJB value functions with oscillatory structure near boundaries may benefit from a periodic inductive bias in the basis functions. |
| `lisht` | composite additive | x·tanh(x): an odd function that grows quadratically near the origin and linearly far from it. Symmetric, never flat, curvature spread evenly across both half-axes. Tests whether the gradient-augmented loss rewards symmetric curvature or one-sided structure. |
| `gausscent_1` | localized curvature | 1-exp(-x²): zero at the origin, curvature concentrated near 0, saturates to 1. Complementary geometry to `gaussian`: tests whether the Gaussian underperformance is due to radial/localized shape or its non-monotone-in-\|x\| profile. |
