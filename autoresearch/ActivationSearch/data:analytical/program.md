# autoresearch/ActivationSearch/data:analytical

Autonomous search for activation functions on the analytic value function with
a discontinuous gradient from `notebook/pdpa_vdp.ipynb` Experiment 3.

## Fixed Study

Use `PDPA_v2` with:

- `power=1`
- `loss="h1"`
- gamma list `[0, 0.01, 0.1, 1, 10]`
- `alpha=1e-5`
- `num_iterations=10`
- `num_insertion=50`
- `pruning_threshold=1e-5`
- training grid: 30x30 points on `[-2, 2]^2`
- evaluation grid: 61x61 points on `[-2, 2]^2`
- target: `V = x1^2 + x2^2 + abs(x1 + x2*abs(x2)/2)`

The global score is

```text
score = eval_h1 * best_neurons
```

where `eval_h1` is computed analytically on the dense evaluation grid at the
PDPA-selected best iteration for each gamma.

For this Experiment 3 task, the primary result is discontinuity behavior:

```text
near_grad = relative gradient error on points near h(x1, x2)=0
```

Use `rank_discontinuity.py` to reselect the best gamma per seed by smallest
`near_grad`. Use `near_score = near_grad * neurons` only as a sparsity-aware
secondary metric.

## What You Do

```text
LOOP FOREVER:
  0. Ensure the run directory exists:
       mkdir -p autoresearch/ActivationSearch/data:analytical/runs

  1. Read autoresearch/ActivationSearch/data:analytical/results.tsv. Pick an
     activation that has not been run yet, or a promising activation that has
     only partial seed coverage.

     Start with:
       tanh, matern52, softplus, gaussian,
       relu, gelu_b0_25, softplus_b0_25, softplus_b0_15,
       smoothy_relu_w0_25, mish_b0_25, mish_b0_15, celu, elu,
       qr_0_1, sp2_b0_25

     You may use any activation already registered in
     scripts/run_activation_experiment.py, because the discontinuous runner
     imports the same ACTIVATIONS registry. For this task, `gaussian` is
     overridden to the Experiment 3 notebook definition `exp(-z^2/2)`.

  2. For each seed in {42, 43, 44, 45, 46}:
       uv run python scripts/run_discontinuous_activation_experiment.py \
         --activation <name> --seed <seed> \
         > autoresearch/ActivationSearch/data:analytical/runs/<name>_seed<seed>.json 2>&1

     Each call prints one JSON line with per-gamma results, best gamma,
     eval metrics, near/far gradient errors, and elapsed time.
     If `uv` is unavailable, use `.venv/bin/python` with the same script and
     arguments.

  3. Aggregate:
       uv run python autoresearch/ActivationSearch/data:analytical/scripts/aggregate.py \
         --activation <name> \
         --description "<short note>"

  4. Append one row to:
       autoresearch/ActivationSearch/data:analytical/results.tsv

     Columns:
       commit, activation, power, loss, seeds, mean_score, std_score,
       mean_eval_h1, mean_eval_grad, mean_neurons, mean_near_grad,
       mean_far_grad, near_far_ratio, best_gamma_mode, status, description

  5. Continue with the next activation.
```

## Constraints

- Do not modify `src/` for this study.
- Keep the algorithm settings above fixed.
- Use `use_sphere=True` only where the shared activation registry already marks
  it as true.
- If a run crashes, inspect the run file. Fix runner-level issues if needed;
  otherwise aggregate the valid seeds as `partial` and move on.

## Plotting

After several rows exist:

```bash
uv run python autoresearch/ActivationSearch/data:analytical/scripts/plot_pareto.py
uv run python autoresearch/ActivationSearch/data:analytical/scripts/rank_discontinuity.py
uv run python autoresearch/ActivationSearch/data:analytical/scripts/plot_near.py
```

This writes `autoresearch/ActivationSearch/data:analytical/pareto.png`,
`results_near.tsv`, and `near_pareto.png`. For this dataset, read
`near_pareto.png` first.
