# PDPA_v1 Debugging & Fixes — 2025-02-15

## Summary

Continued from a prior session where 6 inconsistencies between the Python PDPA_v1 and MATLAB `PDAPmultisemidiscrete.m` were identified. This session focused on implementing the coordinate descent warm-start, insertion robustness, pruning/merging, and fixing the SSN line search objective.

---

## 1. Correct MATLAB Reference File

The previous session incorrectly referenced `PDAPsemidiscrete.m` (single-point insertion). The correct reference for our 2D VDP case is **`PDAPmultisemidiscrete.m`** (multi-point insertion, up to 15 candidates per iteration), as confirmed by `test_run_2d.m` line 63.

## 2. Coordinate Descent Warm-Start

### Problem
SSN's `_initialize_q` assumes the NOC (necessary optimality condition) holds. When outer weights are zero (newly inserted neurons), the NOC is violated: `_initialize_q` produces `G(q) = 0` by construction, so `dq = 0` and SSN cannot move.

MATLAB avoids this via a coordinate descent step (`PDAPmultisemidiscrete.m:135-147`) that gives new neurons non-zero initial weights before SSN runs.

### How MATLAB's Coordinate Descent Works
1. **`coeff`**: For each new neuron, the sign/direction for its outer weight (`-grad/|grad|`, essentially ±1 for scalar output)
2. **`vhat`**: Combined direction vector (zeros for existing neurons, `coeff` for new ones)
3. **`phat`, `what`**: Slope and curvature of the objective along `vhat` (reduces to a scalar)
4. **`tau = phi.prox(alpha/what, -phat/what)`**: Optimal step size from a 1D proximal problem
5. Each new neuron gets initial weight `tau * coeff_i` before SSN

### Solution Implemented
Instead of replicating the exact MATLAB 1D proximal step, we use an Adam warm-start (500 iterations) on outer weights before SSN. This achieves the same goal — moving weights away from zero — and is more general (not restricted to a 1D subspace).

Added `self.model_warmstart` (Adam, `train_outerweights=True`, `lr=1e-3`) in `PDPA_v1.__init__`.

**Files changed**: `scr/PDPA_v1.py`

## 3. Insertion Hard Cap

### Problem
Python's insertion accepted all candidates where `|profile| > alpha`. With `alpha=1e-3`, nearly all 50 optimized candidates passed this weak filter. MATLAB caps at 15 (`PDAPmultisemidiscrete.m:95-96`).

### Solution
Added `max_insert` parameter (default 15) to `insertion()`. After accepting candidates, sorts by profile value descending and caps at `max_insert`.

**Files changed**: `scr/PDPA_v1.py` — `insertion()` and `retrain()` signatures

## 4. Neuron Merging (Postprocess)

### Problem
Many inserted candidates converge to the same local maximum during L-BFGS optimization, producing near-duplicate neurons. MATLAB's `postprocess` (`setup_problem_NN_2d.m:204-216`) merges nearby candidates using connected-component clustering.

Without merging, all 15 capped candidates could be duplicates. SSN assigns them equal weights (~0.16 each), none below the sparsity threshold, so no sparsification occurs.

### Solution
Rewrote `prune_small_weights()` to include merging:
1. **Merge duplicates**: Compute pairwise cosine similarity of normalized `(w, b)` vectors. Union-find clusters neurons with similarity `> 1 - merge_tol`. Sum outer weights within each cluster, keep one representative.
2. **Remove exact zeros**: Prune neurons where `|outer_weight| == 0` (from SSN's proximal operator).

Parameter `merge_tol` (default `1e-3`) controls the merging sensitivity.

**Files changed**: `scr/PDPA_v1.py` — `prune_small_weights()` and `retrain()` signatures

### Result
Before: 15→15 neurons per iteration (no reduction).
After: 15→1, 16→2, 17→3, ... (duplicates properly merged).

## 5. SSN Line Search Objective Fix

### Problem
MATLAB's SSN (`SSN.m:34`) uses the **full objective** in its line search:
```matlab
obj = @(u) F.F(Sred*u - ref) + alpha*sum(phi.phi(computeNorm(u, NQ)));
```

Python's `_compute_loss` returned **only data loss** for SSN:
```python
if isinstance(self.optimizer, (SSN, SSN_TR)):
    total_loss = data_loss  # missing regularization!
```

This caused SSN to reject sparsifying Newton steps (which reduce regularization but may slightly increase data loss), preventing SSN from ever producing zero weights.

### Solution
Changed `_compute_loss` to always return the full objective (data loss + regularization) for all optimizers.

**Files changed**: `scr/model.py` — `_compute_loss()`

### Result
Before fix: `removed 0 zeros` in every iteration.
After fix: `removed 3 zeros` in iteration 4 — SSN now correctly sparsifies.

## 6. SSN Iteration Count

### Finding
SSN converges in 3-4 steps (quadratic convergence, expected for Newton methods). The remaining iterations are wasted — loss changes by ~1e-8 per step after convergence.

### Solution
Reduced SSN iterations from 1000 to 20 in `retrain()`.

**Files changed**: `scr/PDPA_v1.py` — `retrain()`

## 7. MATLAB-Style Logging

Added per-iteration summary matching MATLAB's output format:
```
PDAP:   1, supp: 15->1,  train: 1.74e+01, val: 2.36e+01
PDAP:   2, supp: 16->3,  train: 1.04e+01, val: 1.01e+01
```

**Files changed**: `scr/PDPA_v1.py` — `retrain()`

---

## Files Modified

| File | Changes |
|------|---------|
| `scr/PDPA_v1.py` | Warm-start model, insertion cap, neuron merging, SSN iterations, logging |
| `scr/model.py` | Full objective in `_compute_loss` for all optimizers |

## Final Test Results (5 iterations, alpha=1e-3, gamma=1.0)

```
PDAP:   1, supp: 15->1,  train: 1.74e+01, val: 2.36e+01
PDAP:   2, supp: 16->3,  train: 1.04e+01, val: 1.01e+01
PDAP:   3, supp: 18->9,  train: 7.45e+00, val: 8.86e+00
PDAP:   4, supp: 24->20, train: 5.79e+00, val: 8.70e+00  (3 zeros removed by SSN)
PDAP:   5, supp: 35->33, train: 2.51e+00, val: 4.51e+00
```

## Remaining Items (from 2025-02-15)
- **Reason 3 (not implemented)**: MATLAB reuses existing support positions as starting points for `find_max`, concentrating search near known good positions. Python samples purely randomly.
- **SSN convergence criterion**: MATLAB stops SSN when `norm(Gq) < 1e-10 * normGQ0` and no damping was needed. Python uses a fixed iteration count. Could add early stopping.
- **Warm-start tuning**: The 500 Adam iterations and `lr=1e-3` are reasonable defaults but may need tuning per problem.

---

# PDPA_v2 — Matching MATLAB Exactly — 2026-02-16

## Summary

Created `PDPA_v2.py` replacing Adam warm-start with MATLAB-style coordinate descent. Then systematically investigated and fixed discrepancies between Python and MATLAB neuron counts. Changes span `model.py`, `PDPA_v2.py`, and `utils.py`.

---

## 1. PDPA_v2 Created (Coordinate Descent Warm-Start)

### Problem
Adam warm-start (PDPA_v1) produced outer weights too large for SSN's proximal operator to zero out. MATLAB uses a 1D coordinate descent step (`PDAPmultisemidiscrete.m:104-147`) that initializes new neurons at the characteristic scale where the proximal operator is effective.

### Solution
Created `src/PDPA_v2.py` with `_coordinate_descent_init()` matching MATLAB exactly:
1. Compute signed profiles (data-loss gradient per outer weight)
2. Build descent direction `coeff`: magnitude 1 for best neuron, `sqrt(eps)` for rest
3. Compute slope `phat` and curvature `what` along combined direction
4. Solve 1D proximal: `tau = _phi_prox(alpha/what, -phat/what, th, gamma)`
5. Return `tau * coeff` as initial outer weights

Also added `_phi_prox` to `src/utils.py` — the nonconvex proximal operator with closed-form for gamma > 0:
```python
gam = gamma / (1 - th)
tau = 0.5 * max(a + sqrt(a^2 + 4*(g - sigma)/gam), 0)
```
where `a = g - sigma*th - 1/gam`.

**Files created**: `src/PDPA_v2.py`
**Files changed**: `src/utils.py` (added `_phi_prox`)

## 2. Parameter Mismatch Fix (th, alpha)

### Problem
Python used `th=0.01` and notebook used `alpha=1e-15`. MATLAB uses `th=0.5` (`setup_problem_NN_2d_from_xhat.m:183`) and `alpha=1e-5` (`run_vdp_2d_from_mat.m:105`).

### Solution
Changed `th` default from `0.01` to `0.5` in `model.py:32`.

**Files changed**: `src/model.py`

## 3. `final_neurons` Return Value

### Problem
MATLAB reports neuron count at the final iteration (after postprocess). Python's `retrain()` only returned `best_iteration` and `best_neurons` (at best training loss iteration).

### Solution
Added `final_neurons = int(W_hidden.shape[0])` as a third return value from `retrain()` in both `PDPA_v1.py` and `PDPA_v2.py`.

**Files changed**: `src/PDPA_v1.py`, `src/PDPA_v2.py`

## 4. Loss Normalization Fix

### Problem
MATLAB normalizes the data loss by `Nx = numel(p.xhat)` (`setup_problem_NN_2d_from_xhat.m:198`), where `numel(xhat) = d * N_points` (e.g., 1800 for 2D with 900 points). Python normalized by `N = x_input.shape[0]` (number of points only, e.g., 810 with 90% train split).

This made Python's data gradient ~2.2x stronger than MATLAB's, weakening the relative effect of regularization. SSN's proximal operator never produced zeros because the data term dominated.

### Solution
Changed all normalizations in `model.py` and `PDPA_v2.py` from `N` to `Nx = N * d`:

- `model.py:_compute_loss`: `value_loss = ||...||^2 / (2*Nx)`, `grad_loss = ||...||^2 / (2*Nx)`
- `model.py:train()` data_hessian: `(1/Nx) * (w1*S'S + w2*S_grad'S_grad)`
- `PDPA_v2.py:_coordinate_descent_init`: profiles, phat, what all use `Nx`
- `PDPA_v2.py:insertion:profile`: normalization uses `Kx = K * d`

**Files changed**: `src/model.py`, `src/PDPA_v2.py`

## 5. Training Split — Match MATLAB (Use All Data)

### Problem
MATLAB uses all 900 data points for training. Python used 90% (810 points) with 10% validation.

### Solution
In `PDPA_v2.__init__`, compute `training_pct = (N_total - 1) / N_total` to leave just 1 point for validation (preventing errors from empty validation set).

**Files changed**: `src/PDPA_v2.py`

## 6. Insertion Function Rewrite

### Problem
All 50 L-BFGS candidates converged to the same global maximum, producing 14-15 duplicates per iteration. After merging in `prune_small_weights`, only ~1 new neuron survived per PDAP step. This is why Python got ~20-35 neurons after 15 iterations instead of MATLAB's ~99.

Root causes:
- L-BFGS with `max_iter=1000` + `strong_wolfe` was too aggressive — every starting point converged to the single dominant maximum
- No deduplication within insertion (duplicates only caught after SSN)
- No existing support as starting candidates
- MATLAB normalizes the residual before optimization (`y_norm = y/norm(y)`, `setup_problem_NN_2d.m:385`)

### Solution — Matching MATLAB `find_max` (setup_problem_NN_2d.m:361-427)

Rewrote `insertion()` with:

1. **Residual normalization**: `residual_v_n = residual_v / norm(residual)` before optimization (matches MATLAB line 385)
2. **Existing support as starting candidates**: Include current network's hidden weights (normalized to S^d) alongside random samples (matches MATLAB lines 368-383)
3. **Iterative merge+re-optimize loop** (up to 5 rounds): Optimize all → merge by cosine similarity → re-optimize merged set → repeat until stable (matches MATLAB lines 390-414)
4. **Reduced L-BFGS**: `max_iter=200` (from 1000), matching MATLAB's `fminunc` with `MaxIter=500` but fewer effective iterations
5. **Cosine similarity merge within insertion**: `merge_tol=1e-2` threshold, keeping only distinct candidates
6. **Removed aggressive buffer cutoff**: MATLAB's `cutoff = alpha + 0.5*(max-alpha)` is effectively a no-op (selects all). Now just takes top `max_insert` above `alpha`.

**Files changed**: `src/PDPA_v2.py` — complete rewrite of `insertion()`

### Result
Before: `merged 14 duplicates` every iteration → ~1 new neuron/step → 20-35 total
After: `after merge 60-70` distinct candidates → 2-15 accepted/step → 138-194 total

---

## Files Modified (2026-02-16)

| File | Changes |
|------|---------|
| `src/PDPA_v2.py` | Created: coordinate descent, rewritten insertion, Nx normalization, training split |
| `src/model.py` | th=0.5 default, Nx normalization in `_compute_loss` and data_hessian |
| `src/utils.py` | Added `_phi_prox` (nonconvex proximal operator) |
| `src/PDPA_v1.py` | Added `final_neurons` return value |

## Test Results (gamma sweep, alpha=1e-5, th=0.5, loss_weights=[1,0], 15 iterations)

```
Python:   gamma=0→179, 0.01→168, 0.1→168, 1→164, 10→138  (monotonically decreasing)
MATLAB:   gamma=0→99,  0.01→188, 0.1→188, 1→119, 10→91   (non-monotone due to warm-start)
```

Python's trend is monotonically decreasing with gamma, which is the correct theoretical behavior for independent (non-warm-started) runs.

## Remaining Differences from MATLAB

1. **MATLAB counts after final `postprocess`**: `u_pp = postprocess(u_sol, 1e-3)` merges by Euclidean distance in stereographic coordinates. Python's `final_neurons` is before this step.
2. **MATLAB gamma sweep uses warm-start**: Each gamma starts from previous gamma's solution. This explains the non-monotone MATLAB pattern (gamma=0.01 has 188 > gamma=0's 99 because nonconvex relaxation from L1 solution preserves more neurons).
3. **MATLAB kernel uses smoothed ReLU**: `max_delta(0, y)` with `delta=0.001` (`setup_problem_NN_2d_from_xhat.m:150`), not exact ReLU. This affects the optimization landscape in `find_max` and the SSN data matrices.
4. **MATLAB parameterizes neurons in stereographic R^d**: `find_max` optimizes position `x ∈ R^2` via unconstrained `fminunc`. Python optimizes `(w,b) ∈ S^d` via sphere-constrained L-BFGS. Same representational power but different optimization landscapes.

---

# Semiconcavity-Aware PDPA_v1 and Reference Discontinuity Test — 2026-05-25

## Summary

Reworked the legacy `PDPA_v1` path into a semiconcavity-aware shallow network:

```text
V_theta(x) = (C / 2) ||x||^2 - g_theta(x)
```

Here `g_theta` is constrained to be convex by using a shallow ReLU network with nonnegative output weights, plus an affine term. The scalar `C` is trainable, but is deliberately not included in the sparsity penalty. The existing signed `PDPA_v2` algorithm was not changed; it is used only as a baseline in the new reference experiment script.

## 1. Semiconcavity Enforcement in `PDPA_v1`

### Motivation

For a `C`-semiconcave value function, `x -> V(x) - (C/2)||x||^2` is concave. Equivalently,

```text
g(x) = (C/2)||x||^2 - V(x)
```

should be convex. This motivates fitting `g_theta` as a convex shallow network and recovering `V_theta` from the quadratic-minus-convex representation.

### Implementation

Replaced the old warm-start/SSN legacy implementation in `src/PDPA_v1.py` with:

1. **Trainable semiconcavity constant**
   - `C = c_min + softplus(raw_C)` keeps `C > 0`.
   - `C` participates in data fitting.
   - `C` is not included in the nonconvex sparsity penalty.

2. **Convex shallow network for `g_theta`**
   - Reuses the existing `ShallowNetwork`.
   - Inner atoms are inserted and then frozen, consistent with the shallow PDPA setting.
   - Output weights are projected to be nonnegative after each optimizer step.
   - An affine term is included in `g_theta`; affine terms preserve convexity and are not penalized.

3. **Penalty**
   - The existing nonconvex log penalty `_phi` is applied only to the nonnegative output weights of `g_theta`.
   - The same exponent convention is used: `q = 2 / (power + 1)`.

4. **Insertion**
   - Candidate atoms are inserted one-sided for the convex network `g_theta`.
   - The insertion profile no longer uses `abs(profile)` because only positive atoms are admissible.
   - Candidates are accepted when the one-sided profile exceeds `alpha + threshold`.

5. **Prediction helpers**
   - Added `predict_value_tensor`, `predict_gradient_tensor`, and `predict` for experiment scripts.

**Files changed**: `src/PDPA_v1.py`

## 2. Reference Paper Test Case

Added a script implementing the 2D exponential distance function from `semiconcave_approximaton.pdf`:

```text
v(x) = min_i exp(-0.5 ||x - c_i||^2),
c_i in {e_1, e_2, -e_1, -e_2}, Omega = [-1, 1]^2.
```

This target is semiconcave, Lipschitz, not `C^1`, and has gradient discontinuities along the diagonals of the square.

The script reports:

- Relative value, gradient, and H1 errors.
- Near-switching vs far-from-switching gradient error.
- HJB residuals for the paper Hamiltonian:

```text
H(p, a) = |p|^2 + 2 log(|a|) a^2
```

- Paper-style `Omega_delta` metrics:
  - `DC_delta`: max value error.
  - `DW1_delta`: mean gradient error.
  - `DWinf_delta`: max gradient error.
  - `DH1_delta`: mean absolute HJB residual.
  - `DHinf_delta`: max absolute HJB residual.

The same script can optionally run unchanged `PDPA_v2` as the signed shallow-network baseline with `--run-signed-baseline`.

**Files added**: `scripts/run_semiconcave_reference_experiment.py`

## 3. Verification

Syntax checks:

```bash
./.venv/bin/python -m py_compile src/PDPA_v1.py scripts/run_semiconcave_reference_experiment.py
```

Tiny semiconcave smoke test:

```bash
./.venv/bin/python scripts/run_semiconcave_reference_experiment.py \
  --train-grid-size 7 --eval-grid-size 11 \
  --num-iterations 1 --num-insertion 3 --max-insert 2 \
  --fit-steps 3 --insertion-steps 2 \
  --display-every 1 --quiet
```

Result:

```text
semiconcave_PDPA   neurons=   2 C=  0.9812 rel_h1=2.703e+00 rel_grad=3.085e+00 near_grad=1.132e+00 far_grad=1.180e+00 HJB_mean=5.389e-01 HJB_max=1.931e+00
```

Tiny baseline smoke test:

```bash
./.venv/bin/python scripts/run_semiconcave_reference_experiment.py \
  --train-grid-size 5 --eval-grid-size 7 \
  --num-iterations 1 --num-insertion 1 --max-insert 1 \
  --fit-steps 2 --insertion-steps 1 \
  --display-every 1 --quiet --run-signed-baseline
```

Result:

```text
semiconcave_PDPA   neurons=   1 C=  0.9874 rel_h1=2.961e+00 rel_grad=3.344e+00 near_grad=1.207e+00 far_grad=1.212e+00 HJB_mean=6.484e-01 HJB_max=1.938e+00
signed_PDPA_v2     neurons=   1 C=       - rel_h1=9.788e-01 rel_grad=9.407e-01 near_grad=3.574e-01 far_grad=2.849e-01 HJB_mean=5.949e-03 HJB_max=4.488e-02
```

Moderate comparison run:

```bash
./.venv/bin/python scripts/run_semiconcave_reference_experiment.py \
  --train-grid-size 15 --eval-grid-size 41 \
  --num-iterations 3 --num-insertion 15 --max-insert 5 \
  --fit-steps 100 --insertion-steps 10 \
  --display-every 1 --quiet --run-signed-baseline
```

Result:

```text
semiconcave_PDPA   neurons=  13 C=  0.2727 rel_h1=3.111e-01 rel_grad=3.686e-01 near_grad=2.014e-01 far_grad=7.391e-02 HJB_mean=3.399e-02 HJB_max=3.169e-01
signed_PDPA_v2     neurons=  15 C=       - rel_h1=3.579e-01 rel_grad=4.238e-01 near_grad=2.274e-01 far_grad=8.375e-02 HJB_mean=7.119e-02 HJB_max=3.255e-01
```

The moderate run suggests the semiconcavity-aware model improves relative H1 error, gradient error, near-discontinuity gradient error, and mean HJB residual compared with the unchanged signed `PDPA_v2` baseline. The max HJB residual is similar.

## Remaining Items

- Run a larger grid and longer training schedule before drawing strong conclusions.
- Tune `alpha`, `gamma`, `fit_steps`, and insertion count for the semiconcave model.
- Consider adding the reference-paper metrics to `src/metric.py` after the script stabilizes.
- The current implementation uses projected Adam for the positive outer-weight/C/affine subproblem, not SSN.

## 4. Semiconcave Activation Sweep

Added an activation sweep runner that reuses the previous discontinuous-gradient activation registry and evaluates all activations with `PDPA_v1` on the new exponential-distance example.

**Files added**: `scripts/run_semiconcave_activation_experiment.py`

The runner supports:

- `--activation all` or comma-separated activation names.
- one or more seeds via `--seeds`.
- full gamma sweeps via `--gammas`, or a single gamma via `--gamma`.
- saved per-run JSON files under `autoresearch/SemiconcaveFittingComparison/VDPReference/runs`.
- a TSV summary at `autoresearch/SemiconcaveFittingComparison/VDPReference/results.tsv`.

Compact pilot command:

```bash
./.venv/bin/python scripts/run_semiconcave_activation_experiment.py \
  --activation all --seeds 42 --gamma 1 \
  --train-grid-size 11 --eval-grid-size 21 \
  --num-iterations 2 --num-insertion 8 --max-insert 3 \
  --fit-steps 30 --insertion-steps 5 \
  --overwrite-summary --quiet
```

Pilot result: all 150 activations completed successfully.

Top activations by relative H1 error in this compact pilot:

```text
cubic                         rel_h1=0.7311  rel_grad=0.8301  neurons=6  HJB_mean=0.0962
sin                           rel_h1=0.7408  rel_grad=0.8774  neurons=6  HJB_mean=0.1043
leaky_relu2_a0_0375_sphere    rel_h1=0.7443  rel_grad=0.7489  neurons=5  HJB_mean=0.0359
x_absx                        rel_h1=0.7499  rel_grad=0.8311  neurons=6  HJB_mean=0.0506
leaky_relu2_a0_025_sphere     rel_h1=0.7551  rel_grad=0.7621  neurons=5  HJB_mean=0.0348
leaky_relu2_a0_02_sphere      rel_h1=0.7558  rel_grad=0.7611  neurons=5  HJB_mean=0.0332
```

Top activations by mean HJB residual:

```text
leaky_relu                    HJB_mean=0.03316  rel_h1=1.0756
leaky_relu2_a0_02_sphere      HJB_mean=0.03320  rel_h1=0.7558
leaky_relu2_a0_025_sphere     HJB_mean=0.03478  rel_h1=0.7551
leaky_relu2_a0_0375_sphere    HJB_mean=0.03591  rel_h1=0.7443
relu                          HJB_mean=0.04009  rel_h1=0.9943
```

Interpretation: in the compact pilot, the leaky squared-ReLU family gives the strongest combined behavior for the semiconcavity-aware model because it is near the best H1 group and also dominates the HJB residual ranking. The `cubic`, `sin`, and `x_absx` entries are numerically competitive in H1, but they do not preserve convexity of `g_theta` in the same structural sense, so they should be treated as empirical baselines rather than semiconcavity-guaranteed choices.

## 5. PDPA_v1 vs PDPA_v2 With Autoresearch Hyperparameters

Added a model-comparison runner that fixes the activation to ReLU and compares the model class:

- `PDPA_v1_semiconcave`: constrained ansatz `V = (C/2)||x||^2 - g`.
- `PDPA_v2_signed`: existing signed shallow-network PDPA_v2 baseline.

**Files added**: `scripts/run_semiconcave_model_comparison.py`

The comparison uses the same common hyperparameters as the previous activation autoresearch:

```text
alpha = 1e-5
gammas = [0, 1e-2, 1e-1, 1, 10]
power = 1
loss = h1
num_iterations = 10
num_insertion = 50
threshold = 1e-5
train_grid_size = 30
eval_grid_size = 61
seeds = [42, 43, 44, 45, 46]
activation = ReLU
```

`PDPA_v1` additionally uses projected Adam for its positive-weight/C/affine subproblem with `fit_steps=200`; this is model-specific because `PDPA_v2` still uses its SSN/coordinate-descent path.

Command:

```bash
./.venv/bin/python scripts/run_semiconcave_model_comparison.py \
  --models v1,v2 \
  --seeds 42,43,44,45,46 \
  --gammas 0,0.01,0.1,1,10 \
  --train-grid-size 30 --eval-grid-size 61 \
  --num-iterations 10 --num-insertion 50 --max-insert 15 \
  --threshold 1e-5 --alpha 1e-5 --power 1 --th 0.5 \
  --v1-fit-steps 200 --insertion-steps 20 \
  --quiet
```

Results are saved under:

```text
autoresearch/SemiconcaveFittingComparison/VDPReference/model_comparison/
```

Five-seed aggregate:

```text
PDPA_v1_semiconcave:
  mean rel_h1       = 0.12220 +/- 0.01296
  mean rel_grad     = 0.14608 +/- 0.01552
  mean neurons      = 114.8   +/- 7.46
  mean score        = 13.96   +/- 0.82
  mean HJB residual = 0.01456 +/- 0.00052
  mean HJB max      = 0.31110 +/- 0.02949
  mean C            = 0.53100 +/- 0.00985

PDPA_v2_signed:
  mean rel_h1       = 0.14152 +/- 0.03151
  mean rel_grad     = 0.16899 +/- 0.03757
  mean neurons      = 117.0   +/- 19.96
  mean score        = 16.16   +/- 2.50
  mean HJB residual = 0.01604 +/- 0.00271
  mean HJB max      = 0.29471 +/- 0.02963
```

Per-seed best results:

```text
seed 42: v1 h1=0.1306 n=112 gamma=10   | v2 h1=0.1873 n=82  gamma=1
seed 43: v1 h1=0.1110 n=121 gamma=10   | v2 h1=0.1611 n=128 gamma=0
seed 44: v1 h1=0.1391 n=103 gamma=0    | v2 h1=0.1159 n=129 gamma=10
seed 45: v1 h1=0.1219 n=120 gamma=0    | v2 h1=0.1168 n=127 gamma=0
seed 46: v1 h1=0.1084 n=118 gamma=0    | v2 h1=0.1265 n=119 gamma=0
```

Interpretation: under the old autoresearch hyperparameter scale, neuron counts are no longer artificially small. `PDPA_v1_semiconcave` has better mean H1, mean gradient error, mean score, and mean HJB residual across the five seeds. `PDPA_v2_signed` has a slightly better mean max HJB residual and wins two individual seeds by H1, so the result is favorable to the semiconcavity-aware ansatz but not uniformly dominant.

### Extended 15-Seed Update

Ran 10 additional seeds with the same settings:

```text
additional seeds = [47, 48, 49, 50, 51, 52, 53, 54, 55, 56]
combined seeds   = [42, ..., 56]
```

The combined table is saved at:

```text
autoresearch/SemiconcaveFittingComparison/VDPReference/model_comparison/results_42_56.tsv
```

Combined 15-seed aggregate:

```text
PDPA_v1_semiconcave:
  mean rel_h1       = 0.11737 +/- 0.01198
  mean rel_grad     = 0.14030 +/- 0.01433
  mean neurons      = 112.07  +/- 6.84
  mean score        = 13.12   +/- 1.23
  mean HJB residual = 0.01437 +/- 0.00054
  mean HJB max      = 0.30792 +/- 0.02294
  mean C            = 0.53251 +/- 0.01615

PDPA_v2_signed:
  mean rel_h1       = 0.12665 +/- 0.03127
  mean rel_grad     = 0.15138 +/- 0.03726
  mean neurons      = 123.27  +/- 14.10
  mean score        = 15.31   +/- 2.83
  mean HJB residual = 0.01526 +/- 0.00237
  mean HJB max      = 0.30157 +/- 0.03623
```

Paired seed-level comparison:

```text
H1:           v1 wins 7,  v2 wins 8,  mean(v1-v2) = -0.00928, 95% CI approx [-0.02778, 0.00921]
grad:         v1 wins 7,  v2 wins 8,  mean(v1-v2) = -0.01108, 95% CI approx [-0.03313, 0.01097]
score:        v1 wins 12, v2 wins 3,  mean(v1-v2) = -2.19046, 95% CI approx [-4.07386, -0.30706]
HJB mean:     v1 wins 9,  v2 wins 6,  mean(v1-v2) = -0.00089, 95% CI approx [-0.00213, 0.00035]
far grad err: v1 wins 14, v2 wins 1,  mean(v1-v2) = -0.00337, 95% CI approx [-0.00533, -0.00141]
neurons:      v1 wins 13, v2 wins 2,  mean(v1-v2) = -11.2 neurons, 95% CI approx [-19.36, -3.04]
```

Updated interpretation: after 15 seeds, the evidence is stronger that `PDPA_v1_semiconcave` is more compact and has a better H1-times-neuron score. It also has slightly better mean H1, gradient error, and mean HJB residual. However, the paired H1 and paired gradient differences are not decisive because `PDPA_v2_signed` wins 8 of 15 seeds by H1. The strongest claim is therefore not "v1 is uniformly more accurate", but rather:

```text
On the Kunisch semiconcave reference example with the autoresearch hyperparameters,
PDPA_v1_semiconcave is consistently more compact and better by sparsity-adjusted score,
while pure H1 accuracy is competitive but not uniformly better than PDPA_v2_signed.
```

## 6. Convex-Safe Activation Sweep for PDPA_v1

Because semiconcavity preservation requires `g_theta` to be convex, the activation in

```text
g_theta(x) = sum_i u_i rho(a_i x + b_i) + affine(x),  u_i >= 0
```

must itself be convex. Positive output weights alone are not sufficient. Nonconvex activations such as `sin`, `tanh`, `sigmoid`, `GELU`, `SiLU`, `Mish`, Gaussian, and `x_absx` should be treated only as empirical baselines, not as semiconcavity-guaranteed models.

Ran a convex-safe PDPA_v1 activation sweep with the same autoresearch hyperparameters:

```text
alpha = 1e-5
gammas = [0, 1e-2, 1e-1, 1, 10]
power = 1
loss = h1
num_iterations = 10
num_insertion = 50
threshold = 1e-5
train_grid_size = 30
eval_grid_size = 61
```

First five-seed sweep used seeds `[42, 43, 44, 45, 46]` and tested:

```text
relu, leaky_relu, relu2,
leaky_relu2_a0_015_sphere,
leaky_relu2_a0_02_sphere,
leaky_relu2_a0_025_sphere,
leaky_relu2_a0_0375_sphere,
abs_act,
softplus_b0_25,
logcosh,
smoothy_relu_sphere,
softplus, softplus_b1, softplus_b2,
smooth_relu_1, smooth_relu_4,
bent_id, elu, celu, quartic
```

Combined five-seed aggregate is saved at:

```text
autoresearch/SemiconcaveFittingComparison/VDPReference/v1_convex_activation_sweep_aggregate.tsv
```

Top five-seed results by H1:

```text
leaky_relu                   h1=0.11109 +/- 0.00567, grad=0.13278, neurons=108.4, score=12.06, HJB_mean=0.01427
abs_act                      h1=0.11314 +/- 0.00757, grad=0.13518, neurons=118.8, score=13.46, HJB_mean=0.01532
relu                         h1=0.12220 +/- 0.01296, grad=0.14608, neurons=114.8, score=13.96, HJB_mean=0.01456
smoothy_relu_sphere          h1=0.21624 +/- 0.00322, grad=0.23494, neurons=114.6, score=24.76, HJB_mean=0.03517
leaky_relu2_a0_0375_sphere   h1=0.32638 +/- 0.01469, grad=0.38730, neurons=110.2, score=35.95, HJB_mean=0.07158
```

The smooth convex alternatives `softplus`, `logcosh`, `ELU/CELU`, `quartic`, and `smooth_relu` variants were not competitive under this fixed insertion/training setup; many selected zero active atoms and had H1 around `4`. This should not be interpreted as a universal statement about those activations, only that the current PDPA_v1 insertion/optimization setup did not use them effectively.

Extended the top two alternatives, `leaky_relu` and `abs_act`, to seeds `[42, ..., 56]` and compared them with the 15-seed ReLU PDPA_v1 and ReLU PDPA_v2 baselines. Aggregate saved at:

```text
autoresearch/SemiconcaveFittingComparison/VDPReference/v1_top_activation_vs_baselines_42_56.tsv
```

15-seed aggregate:

```text
v1_leaky_relu:
  mean rel_h1       = 0.11169 +/- 0.00607
  mean rel_grad     = 0.13350 +/- 0.00728
  mean neurons      = 111.00 +/- 9.50
  mean score        = 12.42  +/- 1.44
  mean HJB residual = 0.01436 +/- 0.00068

v1_abs_act:
  mean rel_h1       = 0.11379 +/- 0.00951
  mean rel_grad     = 0.13594 +/- 0.01133
  mean neurons      = 116.20 +/- 7.00
  mean score        = 13.23  +/- 1.37
  mean HJB residual = 0.01549 +/- 0.00062

v1_relu:
  mean rel_h1       = 0.11737 +/- 0.01198
  mean rel_grad     = 0.14030 +/- 0.01433
  mean neurons      = 112.07 +/- 6.84
  mean score        = 13.12  +/- 1.23
  mean HJB residual = 0.01437 +/- 0.00054

v2_relu:
  mean rel_h1       = 0.12665 +/- 0.03127
  mean rel_grad     = 0.15138 +/- 0.03726
  mean neurons      = 123.27 +/- 14.10
  mean score        = 15.31  +/- 2.83
  mean HJB residual = 0.01526 +/- 0.00237
```

Paired comparison:

```text
v1_leaky_relu vs v1_relu:
  H1 wins:       9 / 15
  score wins:   10 / 15
  mean H1 delta = -0.00567, 95% CI approx [-0.01223, 0.00088]

v1_leaky_relu vs v2_relu:
  H1 wins:       11 / 15
  score wins:    12 / 15
  neuron wins:   12 / 14 non-tied
  mean H1 delta  = -0.01496, 95% CI approx [-0.03333, 0.00341]
  mean score delta = -2.89659, 95% CI approx [-4.83542, -0.95776]
```

Updated activation conclusion:

```text
For the semiconcavity-preserving PDPA_v1 architecture, leaky ReLU is the best
tested convex-safe activation under the current autoresearch hyperparameters.
It improves mean H1 and sparsity-adjusted score relative to ReLU, while keeping
essentially the same mean HJB residual. abs_act is also competitive in H1, but
it has worse HJB residual and uses more neurons than leaky ReLU.
```

### Fair Same-Activation Model Comparison

The previous comparison `v1_leaky_relu` vs `v2_relu` was not a fully controlled model comparison because the activation differed. Updated `scripts/run_semiconcave_model_comparison.py` so activation is an explicit experiment axis and output files include the activation name. Then ran `PDPA_v2` with the same top convex-safe activations used for `PDPA_v1`:

```text
activations = [leaky_relu, abs_act]
seeds = [42, ..., 56]
gammas = [0, 1e-2, 1e-1, 1, 10]
```

Combined same-activation aggregate is saved at:

```text
autoresearch/SemiconcaveFittingComparison/VDPReference/same_activation_model_comparison_42_56.tsv
```

15-seed same-activation means:

```text
activation = leaky_relu
  PDPA_v1_semiconcave: h1=0.11169, grad=0.13350, neurons=111.00, score=12.42, HJB_mean=0.01436
  PDPA_v2_signed:      h1=0.12610, grad=0.15066, neurons=131.40, score=16.54, HJB_mean=0.01501

activation = abs_act
  PDPA_v1_semiconcave: h1=0.11379, grad=0.13594, neurons=116.20, score=13.23, HJB_mean=0.01549
  PDPA_v2_signed:      h1=0.11098, grad=0.13266, neurons=128.07, score=14.15, HJB_mean=0.01408

activation = relu
  PDPA_v1_semiconcave: h1=0.11737, grad=0.14030, neurons=112.07, score=13.12, HJB_mean=0.01437
  PDPA_v2_signed:      h1=0.12665, grad=0.15138, neurons=123.27, score=15.31, HJB_mean=0.01526
```

Paired same-activation conclusions:

```text
leaky_relu:
  H1 wins:      v1 12 / 15, v2 3 / 15
  score wins:   v1 13 / 15, v2 2 / 15
  neuron wins:  v1 14 / 14 non-tied
  mean H1 delta v1-v2 = -0.01440, 95% CI approx [-0.02768, -0.00112]
  mean score delta    = -4.12518, 95% CI approx [-6.00812, -2.24224]

abs_act:
  H1 wins:      v1 5 / 15, v2 10 / 15
  score wins:   v1 10 / 15, v2 5 / 15
  neuron wins:  v1 13 / 15
  HJB wins:     v1 2 / 15, v2 13 / 15

relu:
  H1 wins:      v1 7 / 15, v2 8 / 15
  score wins:   v1 12 / 15, v2 3 / 15
  neuron wins:  v1 13 / 15
```

Corrected interpretation:

```text
The fairest positive result for the semiconcavity-aware model is the same-activation
leaky-ReLU comparison: PDPA_v1_semiconcave is better than PDPA_v2_signed in mean H1,
mean gradient error, neuron count, and sparsity-adjusted score.

The result is activation-dependent. With abs_act, PDPA_v2_signed is slightly more
accurate and has better HJB residual, while PDPA_v1 remains more compact. With ReLU,
PDPA_v1 is more compact but not uniformly more accurate by H1.
```
