# VDP HJB Activation Search

Autonomous search for the activation function that minimizes
`err_h1_val * num_neurons` on the VDP HJB problem.

## Layout

```
autoresearch/ActivationSearch/data:VDP/
  README.md      — this file
  SUMMARY.md     — archived result summary
  program.md     — agent loop instructions
  scripts/       — aggregation and plotting helpers
  results.tsv    — append-only scoreboard, tab-separated, NOT git-tracked
  runs/          — per-(activation, seed) JSON outputs from the script
scripts/
  run_activation_experiment.py  — one (activation, seed) experiment
  run_discontinuous_activation_experiment.py  — one discontinuous-gradient experiment
```

## How to launch

1. Open a fresh Claude Code session in this repo with permissions wide open
   (this is what makes it run autonomously without asking you for each tool
   call). The `--dangerously-skip-permissions` flag, or accepting the
   "always allow" prompts, both work.
2. Prompt:

   ```
   Read autoresearch/ActivationSearch/data:VDP/program.md and start the loop. Don't stop.
   ```

3. Walk away. Claude will iterate activations, run 5 seeds each, and append
   to `autoresearch/ActivationSearch/data:VDP/results.tsv`. Per-run JSON detail lands in
   `autoresearch/ActivationSearch/data:VDP/runs/`.
4. To stop, interrupt the session.

## Reading the results

```
column -t -s $'\t' autoresearch/ActivationSearch/data:VDP/results.tsv | sort -k6 -n
```

The activation with the lowest `mean_score` wins. Cross-check `mean_h1` and
`mean_neurons` — the score is the product, but you may want to know whether a
winner won by being accurate or by being sparse.

## Adding activations

Edit the `ACTIVATIONS` dict at the top of
`scripts/run_activation_experiment.py`. Each entry is `name: (callable,
use_sphere)`. Set `use_sphere=True` only for positively homogeneous activations
(relu, leaky_relu, prelu — anything where `f(c*x) = c*f(x)` for `c > 0`).
Everything else gets `False`.

## Caveats

- The algorithm is fixed at `power=1` (the only fully working configuration of
  PDPA_v2 — the `power != 1` SSN bug is documented in `CLAUDE.md`).
- Training loss is `loss_weights="h1"`. The scoring loss is `err_h1_val`. These
  are aligned by design.
- This is a *search*, not a *hill-climb*. There is no keep/discard branching.
  Every activation is logged, the lowest mean_score wins.
