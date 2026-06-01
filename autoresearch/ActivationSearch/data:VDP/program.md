# autoresearch/ActivationSearch/data:VDP

Autonomous search for the activation function that minimizes a sparsity-aware
score on the VDP HJB problem. Pattern lifted from `~/autoresearch`: human edits
this file, agent runs the loop.

## The study

For each activation we train `PDPA_v2` with `power=1`, `loss="h1"`, the gamma
list `[0, 1e-2, 1e-1, 1, 10]`, and the rest of the hyperparameters from
`notebook/pdpa_v3_vdp.ipynb` (`alpha=1e-5`, `num_iterations=10`,
`num_insertion=50`, `pruning_threshold=1e-5`). For each `(activation, seed)` we
take the gamma that minimizes

    score = err_h1_val[best_iteration] * best_neurons

Per activation we run **5 seeds** (`42, 43, 44, 45, 46`) and report the mean
score across seeds. Lower is better. No keep/discard hill-climbing — this is an
exhaustive ranking, not a hill-climb on `train.py`.

## What you (the agent) do

```
LOOP FOREVER:
  1. Read autoresearch/ActivationSearch/data:VDP/results.tsv. Pick an activation that has not been
     run yet, or one that looks promising and was only run with a subset of
     seeds. The initial activation list is:
       relu, tanh, gelu, silu, sin, softplus, matern52
     You may add more activations to scripts/run_activation_experiment.py
     (extend the ACTIVATIONS dict). Use use_sphere=True only for positively
     homogeneous activations (relu and similar); False otherwise.

  2. For each seed in {42, 43, 44, 45, 46}:
       uv run python scripts/run_activation_experiment.py \
           --activation <name> --seed <seed> \
           > autoresearch/ActivationSearch/data:VDP/runs/<name>_seed<seed>.json 2>&1
     Each call prints one JSON line with per_gamma, best_gamma, best_score,
     best_h1, best_n, elapsed_s. If a run crashes (no JSON on the last line),
     read the file and decide: fix-and-retry if it's something dumb, otherwise
     log the activation as `crash` in results.tsv and move on.

  3. Aggregate the 5 JSON files:
       - mean_score, std_score over best_score
       - mean_h1, mean_neurons over best_h1, best_n
       - best_gamma_mode = the gamma chosen most often across seeds (ties: any)

  4. Append one row to autoresearch/ActivationSearch/data:VDP/results.tsv:
       commit  activation  power  loss  seeds  mean_score  std_score
       mean_h1  mean_neurons  best_gamma_mode  status  description
     Where:
       commit = `git rev-parse --short HEAD`
       seeds  = "42,43,44,45,46" (or a subset if some crashed)
       status = `done` | `partial` | `crash`
       description = a short note (e.g. "baseline relu", "tried matern52")
     TSV is tab-separated, NOT comma-separated. Do NOT git-track results.tsv.

  5. Loop.
```

## Constraints

- You may edit `scripts/run_activation_experiment.py` to add activations to the
  registry, but do NOT change the algorithm settings (gammas, alpha, power,
  loss_weights, num_iterations, num_insertion, pruning_threshold). Those are
  fixed for this study.
- Do not modify `src/`. The algorithm is the dependent variable here, not the
  search target.
- Do not cap run time. The algorithm completes on its own. Per-seed wall time
  for the full sweep with the production config (`num_iterations=10`,
  `num_insertion=50`) is on the order of minutes; 5 seeds is on the order of
  tens of minutes per activation.

## Sanity check before kicking off

The smoke test (already verified by the human):

```
uv run python scripts/run_activation_experiment.py \
    --activation relu --seed 42 --num-iterations 2 --num-insertion 10
```

prints a valid JSON line on stdout in a few seconds.

## Never stop

Once the loop has begun, do NOT pause to ask the human "should I continue?".
The human may be asleep. Run until interrupted. If you run out of activations,
extend the registry: try `elu`, `leaky_relu`, `mish`, `selu`, `hardswish`,
shifted/scaled variants, smooth approximations of relu, etc. Each new
activation costs ~5 seeds of compute and the result is logged either way.
