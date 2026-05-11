# activation_search — how to launch

Autonomous search for the activation function that minimizes
`err_h1_val * num_neurons` on the VDP HJB problem.

## Layout

```
activation_search/
  program.md     — agent loop instructions (read by Claude Code)
  results.tsv    — append-only scoreboard, tab-separated, NOT git-tracked
  runs/          — per-(activation, seed) JSON outputs from the script
  README.md      — this file
  vdp_hjb_summary/          — archived summary of the previous VDP HJB results
  discontinuous_gradient/   — new autoresearch task for the analytic discontinuous-gradient dataset
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
   Read activation_search/program.md and start the loop. Don't stop.
   ```

3. Walk away. Claude will iterate activations, run 5 seeds each, and append
   to `activation_search/results.tsv`. Per-run JSON detail lands in
   `activation_search/runs/`.
4. To stop, interrupt the session.

## Reading the results

```
column -t -s $'\t' activation_search/results.tsv | sort -k6 -n
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
