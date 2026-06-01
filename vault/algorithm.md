# Algorithm & Implementation Details

Sub-level disclosure for the "Code — core algorithm" section of `../CLAUDE.md`.
Covers the full PDAP/SSN pipeline, the MATLAB mapping, known differences,
critical parameters, implementation gotchas, and a script index.

## PDAP loop (one `PDAP` class, configured by `model=` / `insertion=`)

```
PDAP.fit() loop (15-20 iterations):
  1. Insertion: sample S^d -> L-BFGS maximize dual profile -> merge duplicates -> accept atoms
       - model="signed":      accept if |p(omega)| > alpha   (two-sided, signed atoms)
       - model="semiconcave": accept if  p(omega)  > alpha   (one-sided, convex atoms)
       - insertion="finite_step": accept if min_c ΔJ(c;omega) < 0  (q<1; returns c*)
  2. Warm-start: coordinate descent gives new neurons non-zero outer weights
       (1D proximal along the combined new-atom direction; matches MATLAB
        PDAPmultisemidiscrete.m lines 104-147).  finite_step skips this (uses c*).
  3. SSN: semismooth Newton on outer weights (inner weights frozen), ~20 iterations
       - signed:      src/SSN (default masks: signed prox, all coords penalized)
       - semiconcave: src/SSN with penalized_mask/nonneg_mask (nonneg prox on c,
         unpenalized C/affine)
  4. Prune: merge near-duplicate neurons (cosine similarity on S^d), remove
       proximal zeros (MATLAB postprocess + lines 176-179)
```

The single `PDAP` loop is model-agnostic: it drives the model via the uniform
interface (`set_atoms`/`get_atoms`/`warm_start`/`fit_outer_weights`/`predict_tensors`)
and the insertion strategy from `src/PDAP/insertion.py`.  Aliases `from_alias("v2")`
= signed+profile, `"v1"` = semiconcave+profile, `"v3"` = signed+finite_step.  See
`semiconcave_model.md` for the semiconcave model.  Note: the insertion-candidate
merge tolerance (1e-2) is distinct from the prune tolerance (`fit`'s `merge_tol`).

## MATLAB reference

Reference implementation: `/Users/ruizhechao/Documents/NonConvexSparseNN/`

| MATLAB | Python | Notes |
|--------|--------|-------|
| `PDAPmultisemidiscrete.m` | `src/PDAP/pdap.py` | Main PDAP loop (NOT `PDAPsemidiscrete.m`) |
| `SSN.m` | `src/SSN/optimizer.py` | Semismooth Newton optimizer (package) |
| `SSN_TR.m` | `src/SSN/strategies.py` (`steihaug_cg`) | Trust-region globalization |
| `setup_problem_NN_2d_from_xhat.m` | `src/models/signed.py` + `src/net.py` | Kernel, loss, find_max |
| `run_vdp_2d_from_mat.m` | `notebook/pdpa_vdp.ipynb` | Experiment runner |

### Known differences from MATLAB
- **Kernel**: MATLAB uses smoothed ReLU (`delta=0.001`), Python uses exact `torch.relu`.
- **Parameterization**: MATLAB optimizes in stereographic R^d; Python on sphere S^d (`use_sphere=True`).
- **Postprocess**: MATLAB merges by Euclidean distance in stereographic space; Python by cosine similarity on S^d.
- **Gamma sweep**: MATLAB warm-starts each gamma from previous solution; Python runs independently.
- **General power p**: Python supports `ReLU^p` with power-transformed penalty `phi(|u|^q)`, `q=2/(p+1)`. MATLAB only supports `p=1`. See `power_q_penalty.md`.

## Critical parameters (matching MATLAB)

| Parameter | Value | Source |
|-----------|-------|--------|
| `alpha` | `1e-5` | `run_vdp_2d_from_mat.m:105` |
| `th` | `0.5` | `setup_problem_NN_2d_from_xhat.m:183` |
| `delta` (MATLAB only) | `0.001` | `setup_problem_NN_2d_from_xhat.m:147` |
| Loss normalization | `Nx = d * N_points` | `setup_problem_NN_2d_from_xhat.m:198` |
| Max PDAP iterations | `15` | `run_vdp_2d_from_mat.m:12` |
| Max insert per step | `15` | `PDAPmultisemidiscrete.m:95` |
| SSN iterations | `20` | Converges in 3-4 steps |

## Critical implementation notes

1. **SSN requires non-zero initial weights**: `_initialize_q` assumes NOC -> `G(q)=0` at `u=0` -> `dq=0`. Must warm-start before SSN.
2. **SSN gradient must be data-only**: Autograd gives the full objective gradient; SSN adds `alpha*dphi` separately. Double-counting breaks convergence. (The closure returns the full objective; SSN subtracts the penalty gradient internally.)
3. **SSN line search uses full objective**: Data loss + regularization. Without regularization, sparsifying steps get rejected.
4. **Loss normalization**: Must use `Nx = N * d`, not `N`. Mismatch weakens regularization relative to the data term.
5. **Insertion deduplication**: L-BFGS candidates converge to the same maximum with exact ReLU. Must merge within insertion by cosine similarity and use an iterative merge+re-optimize loop.
6. **Sphere parameterization (`use_sphere`)**: For positively homogeneous activations (ReLU), inner weights must lie on the unit sphere S^d (`use_sphere=True`). Without this, L-BFGS pushes weights to extreme norms (1e+15), causing kernel-matrix overflow (`what=nan`) and coordinate-descent failure.
7. **Data scale / normalization (samples, not analytic targets)**: external datasets (e.g. pendulum, value O(100) over a large domain) must be normalized so `alpha`/`gamma` regularize as on the O(1) analytic targets. The data loss grows quadratically in the value scale while the log-penalty grows at most linearly, so an unscaled large value makes `alpha=1e-5` effectively zero. Normalize `x -> [-1,1]^d`, `V -> ~[0,1]`, and rescale `dV` by `s_x/s_v` consistently; un-normalize for any physical HJB residual.

## Modules (`src/`)

- `PDAP/` — the unified outer-loop package:
  - `pdap.py` — `PDAP` class + `fit()` (matches `PDAPmultisemidiscrete.m`); configured
    by `model=` ("signed"|"semiconcave") and `insertion=` ("profile"|"finite_step").
    Holds the shared `sample_uniform_sphere_points` / `prune_small_weights` /
    `check_linearity_neurons` helpers.
  - `insertion.py` — `profile_threshold` / `finite_step` strategies (shared
    `_generate_candidates`: sample -> maximize -> merge -> rescale) + `solve_insertion_weight`.
  - `registry.py` — `v1/v2/v3` (and descriptive) aliases -> (model, insertion) + `from_alias`.
- `models/` — parametric value-function models behind a `Model` protocol (`base.py`):
  - `signed.py` — `SignedModel` (pure network; matches `setup_problem_NN_2d_from_xhat.m`).
  - `semiconcave.py` — `SemiconcaveModel` ansatz + augmented data Hessian. See `semiconcave_model.md`.
  Both expose `set_atoms`/`get_atoms`/`warm_start`/`fit_outer_weights`/`predict_tensors`.
- `net.py` — `ShallowNetwork`: `input -> hidden (ReLU^p) -> output`; `forward_network_matrix()` and `forward_gradient_kernel()` build the SSN data Hessian.
- `SSN/` — the semismooth-Newton optimizer package (one configurable class):
  - `optimizer.py` — `SSN`. Stores `q=2/(p+1)`; assumes NOC; data-only gradient.
    Signed (default masks) and semiconcave (`penalized_mask`/`nonneg_mask`) configs.
  - `strategies.py` — `levenberg_marquardt` (damped Newton) and `steihaug_cg`
    (trust-region MPCG) globalizations, selected by `method=`.
  - `prox.py`, `penalty.py` — proximal / penalty kernels (re-exported by `utils.py`).
  - `mpcg.py` — projected/trust-region CG inner solve for `steihaug_cg`.
- `utils.py` — re-exports the `SSN` penalty/proximal kernels (`_phi`/`_dphi`/`_ddphi`, `_compute_prox`/`_compute_dprox`/`_phi_prox`, `_penalty_grad`, `_nonconvex_correction*`), plus stereographic projection and misc helpers.
- `metric.py` — experiment-analysis utilities (per-gamma neuron/loss tables, plots).
- Open-loop data subsystem: `src/OpenLoop/` — `openloop_optimizer.py`, `transient_openloop_optimizer.py`, `pendulum_pmp_sampler.py` (solvers/sampler) and `data_generation_VDP.py`, `data_generation_pendulum_bb.py`, `data_generation_pendulum_transient.py` (dataset generators).

## Script index (`scripts/`)

| script | role |
|--------|------|
| `run_activation_experiment.py` | base activation registry + VDP-HJB activation search |
| `run_discontinuous_activation_experiment.py` | discontinuous-gradient activation search (extends `ACTIVATIONS`, `set_seed`) |
| `run_pendulum_pmp_openloop_example.py` | generate PMP backward-sampler pendulum dataset (infinite-horizon) |
| `run_pendulum_transient_openloop_dataset.py` | generate transient reduced-gradient BFGS pendulum dataset (finite-horizon T=3) |
| `run_pendulum_bb_openloop_example.py` | pendulum BB open-loop TPBVP example (see `pendulum_bb_tpbvp.md`) |
| `run_pendulum_model_comparison.py` | PDAP semiconcave vs signed model on pendulum data |
| `compare_openloop_methods.py` | compare open-loop data-generation methods |
| `visualize_proximal_deadzone.py` | 4-panel proximal dead-zone figure (see `power_q_penalty.md`) |
| `append_pendulum_pilot_plots.py`, `append_pendulum_transient_phase_plots.py` | plotting helpers |

## Data flow

```
VDP / pendulum data generator -> dict {x, v, dv} -> PDAP(model=, insertion=).fit() -> results
```
VDP experiments: `notebook/pdpa_vdp.ipynb` (pickles in `models/experiment_N/`).
Activation/model studies: `scripts/*` -> `autoresearch/*`.
