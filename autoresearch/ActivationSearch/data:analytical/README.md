# Discontinuous-Gradient Activation Search

Autonomous activation-function search for the analytic dataset from
`notebook/pdpa_vdp.ipynb` Experiment 3.

The target value function is continuous, but its gradient jumps across the
curve

```text
h(x1, x2) = x1 + x2 * abs(x2) / 2 = 0
```

with

```text
V(x1, x2) = x1^2 + x2^2 + c * abs(h(x1, x2)),  c = 1
```

Training uses the same 30x30 grid as the notebook. The runner imports the
existing activation registry, with `gaussian` overridden to the notebook's
`exp(-z^2/2)` definition. Scoring is done on a denser analytic evaluation grid,
so the result is not tied to the one-point validation split inside the PDAP run.

## Layout

```text
autoresearch/ActivationSearch/data:analytical/
  README.md
  SUMMARY.md
  program.md
  scripts/
    aggregate.py
    plot_pareto.py
    rank_discontinuity.py
    plot_near.py
  results.tsv    # generated, ignored
  results_near.tsv # generated, ignored
  near_pareto.png  # generated, ignored
  runs/          # generated per-seed JSON, ignored
scripts/
  run_discontinuous_activation_experiment.py
```

## Smoke Test

```bash
mkdir -p autoresearch/ActivationSearch/data:analytical/runs

uv run python scripts/run_discontinuous_activation_experiment.py \
  --activation softplus_b0_25 --seed 42 \
  --num-iterations 2 --num-insertion 10
```

If `uv` is unavailable in the shell, use:

```bash
mkdir -p autoresearch/ActivationSearch/data:analytical/runs

.venv/bin/python scripts/run_discontinuous_activation_experiment.py \
  --activation softplus_b0_25 --seed 42 \
  --num-iterations 2 --num-insertion 10
```

The script prints one JSON object. The old global score is

```text
score = eval_h1 * best_neurons
```

where `eval_h1` is the relative H1 error on the dense analytic evaluation grid.
The JSON also reports gradient error near and far from the discontinuity.

For Experiment 3, the primary analysis is the discontinuity-focused view:

```bash
.venv/bin/python autoresearch/ActivationSearch/data:analytical/scripts/rank_discontinuity.py
.venv/bin/python autoresearch/ActivationSearch/data:analytical/scripts/plot_near.py
```

This writes `results_near.tsv` and `near_pareto.png`, selecting each seed/gamma
by the smallest `near_grad`.

## Current discontinuity-focused result

The current best behavior at the discontinuous gradient is from spherical
leaky squared ReLU:

```text
phi_alpha(x) = max(x, alpha*x)^2
```

The best tested leak is `alpha=0.02`:

```text
activation                    mean_near_grad  std_near_grad  mean_eval_h1  mean_neurons
leaky_relu2_a0_02_sphere      0.151223        0.003271       0.041791      120.10
leaky_relu2_a0_025_sphere     0.151872        0.004625       0.041918      120.90
leaky_relu2_a0_05_sphere      0.152113        0.002337       0.042039      120.20
leaky_relu2_a0_015_sphere     0.153173        0.004042       0.042512      120.10
leaky_relu2_sphere            0.156104        0.003910       0.043305      115.20
relu2                         0.157911        0.004934       0.043606      106.50
```

Rounded/smooth ReLU variants are consistently worse near the jump. For example,
`smoothy_relu_w0_05` gives `mean_near_grad=0.168244`, and wider smoothing bands
continue to degrade. This supports the Experiment 3 point: fitting the
discontinuous part benefits from an activation with a kink/squared-kink
structure rather than a globally smooth activation.

The same `alpha=0.05` activation without the spherical basis gives
`mean_near_grad=0.228469`, so both the squared-kink activation and spherical
parameterization are important in this setup.
