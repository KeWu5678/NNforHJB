# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sparse neural network training framework for solving Hamilton-Jacobi-Bellman (HJB) equations using primal-dual proximal algorithms (PDPA) with non-convex regularization and semismooth Newton (SSN) optimization. The Python implementation ports and extends a MATLAB reference at `/Users/ruizhechao/Documents/NonConvexSparseNN/`.

## Commands

```bash
# Install dependencies (pyproject.toml, no requirements.txt)
pip install -e .

# Run experiments via notebooks
jupyter notebook notebook/pdpa_vdp.ipynb

# Run a quick test from CLI
python -c "from src.PDPA_v2 import PDPA_v2; print('import ok')"
```

No dedicated test suite exists; testing is done through Jupyter notebooks in `notebook/`.

## Environment Notes

- **Project venv** (`.venv/`) uses Python 3.13.
- **Notebook kernel** may use a different Python (e.g., 3.9 via pyenv). When installing packages, ensure you target the correct interpreter — check which kernel the notebook is using.
- Dependencies are managed in `pyproject.toml` (there is no `requirements.txt`).

## Architecture

### Algorithm Flow (PDPA)

```
PDPA_v1/v2.retrain() loop (15-20 iterations):
  1. Insertion: sample S^d → L-BFGS maximize profile → merge duplicates → accept if |p(ω)| > α
  2. Warm-start: coordinate descent (v2) or Adam (v1) to give new neurons non-zero weights
  3. SSN: semismooth Newton on outer weights (frozen inner weights), 20 iterations
  4. Prune: merge near-duplicate neurons (cosine similarity), remove zeros from proximal operator
```

### Key Modules (`src/`)

- **`PDPA_v2.py`** — Current algorithm matching MATLAB `PDAPmultisemidiscrete.m`. Coordinate descent warm-start, iterative insertion with merge+re-optimize.
- **`PDPA_v1.py`** — Earlier version using Adam warm-start instead of coordinate descent.
- **`model.py`** — Training wrapper: data prep, network creation, loss computation, optimizer dispatch. Loss normalized by `Nx = N * d` (matching MATLAB `numel(xhat)`).
- **`net.py`** — `ShallowNetwork` PyTorch module: `input → hidden (ReLU^p) → output`. Provides `forward_network_matrix()` and `forward_gradient_kernel()` for SSN.
- **`ssn.py`** — Semismooth Newton optimizer. Accepts `power` param; stores `q = 2/(p+1)` for power-transformed penalty. Assumes NOC holds; requires non-zero initial weights. Uses data-only gradient (autograd gives full objective; SSN adds regularization separately).
- **`ssn_tr.py`** — Trust-region variant of SSN using MPCG inner solve. Passes `power` through to SSN base class.
- **`utils.py`** — Penalty `_phi(t, th, gamma)`, derivatives `_dphi`/`_ddphi`, proximal operators `_compute_prox(v, mu, q)`, `_compute_dprox(v, mu, q, prox_result)`, `_phi_prox(sigma, g, th, gamma, q)` (all accept optional `q` for power-transformed penalty), stereographic projection.
- **`metric.py`** — Experiment analysis utilities: `summarize_final_neuron_count_and_loss` (per-gamma neuron count / best loss table), `plot_loss_vs_neurons_by_gamma`, `plot_vdp_value_scatter3d`, `print_experiment_hyperparameters`. Requires `pandas` for the `table_df` output key.

### Data Flow

```
VDP ODE solver (data_generation_VDP.py) → dict {x, v, dv} → PDPA_v2 → pickle in models/
```

Experiments run in `notebook/pdpa_vdp.ipynb`. Results saved as pickle files in `models/experiment_N/`.

## MATLAB Reference

Reference implementation: `/Users/ruizhechao/Documents/NonConvexSparseNN/`

Key files and their Python counterparts:
| MATLAB | Python | Notes |
|--------|--------|-------|
| `PDAPmultisemidiscrete.m` | `PDPA_v2.py` | Main PDAP loop (NOT `PDAPsemidiscrete.m`) |
| `SSN.m` | `ssn.py` | Semismooth Newton optimizer |
| `setup_problem_NN_2d_from_xhat.m` | `model.py` + `net.py` | Kernel, loss, find_max |
| `run_vdp_2d_from_mat.m` | `notebook/pdpa_vdp.ipynb` | Experiment runner |

### Known Differences from MATLAB

- **Kernel**: MATLAB uses smoothed ReLU (`delta=0.001`), Python uses exact `torch.relu`
- **Parameterization**: MATLAB optimizes in stereographic R^d, Python on sphere S^d (when `use_sphere=True`)
- **Postprocess**: MATLAB merges by Euclidean distance in stereographic space; Python merges by cosine similarity on S^d
- **Gamma sweep**: MATLAB warm-starts each gamma from previous solution; Python runs independently
- **General power p**: Python supports `ReLU^p` with power-transformed penalty `phi(|u|^q)`, `q=2/(p+1)`. MATLAB only supports `p=1`.

## Critical Parameters (matching MATLAB)

| Parameter | Value | Source |
|-----------|-------|--------|
| `alpha` | `1e-5` | `run_vdp_2d_from_mat.m:105` |
| `th` | `0.5` | `setup_problem_NN_2d_from_xhat.m:183` |
| `delta` (MATLAB only) | `0.001` | `setup_problem_NN_2d_from_xhat.m:147` |
| Loss normalization | `Nx = d * N_points` | `setup_problem_NN_2d_from_xhat.m:198` |
| Max PDAP iterations | `15` | `run_vdp_2d_from_mat.m:12` |
| Max insert per step | `15` | `PDAPmultisemidiscrete.m:95` |
| SSN iterations | `20` | Converges in 3-4 steps |

## Critical Implementation Notes

1. **SSN requires non-zero initial weights**: `_initialize_q` assumes NOC → `G(q)=0` at `u=0` → `dq=0`. Must warm-start before SSN.
2. **SSN gradient must be data-only**: Autograd gives full objective gradient; SSN adds `alpha*dphi` separately. Double-counting breaks convergence.
3. **SSN line search uses full objective**: Data loss + regularization. Without regularization, sparsifying steps get rejected.
4. **Loss normalization**: Must use `Nx = N * d`, not `N`. Mismatch weakens regularization relative to data term.
5. **Insertion deduplication**: L-BFGS candidates converge to same maximum with exact ReLU. Must merge within insertion by cosine similarity and use iterative merge+re-optimize loop.
6. **Sphere parameterization (`use_sphere`)**: For positively homogeneous activations (ReLU), inner weights must be constrained to the unit sphere S^d (`use_sphere=True`). Without this, L-BFGS pushes weights to extreme norms (1e+15), causing kernel matrix overflow (`what=nan`) and coordinate descent failure.

## Power-Transformed Penalty (general activation power p)

When the activation is `ReLU^p` with `p != 1`, the regularization uses `phi(|u|^q)` instead of `phi(|u|)`, where `q = 2/(p+1)`. For `p=1`, `q=1` and everything reduces to the standard case.

**Files modified** (all changes are backward-compatible for p=1):
- `utils.py`: `_compute_prox(v, mu, q)`, `_compute_dprox(v, mu, q, prox_result)`, `_phi_prox(sigma, g, th, gamma, q)` — all accept optional `q` parameter. For `q != 1`, uses Newton's method instead of closed-form.
- `ssn.py`: Stores `self.q = 2/(power+1)`. `_initialize_q`, `_initilize_G`, `_DG` use chain-rule derivatives with `active` mask for `|u|^{q-1}` singularity at `u=0`. `step` passes `self.q` to proximal operators.
- `ssn_tr.py`: Passes `power` to `SSN.__init__`, passes `self.q` to proximal operators.
- `model.py`: `_compute_loss` applies `|u|^q` transform before `_phi`. `_setup_optimizer` passes `power` to SSN/SSN_TR.
- `PDPA_v2.py`: `_coordinate_descent_init` passes `q=2/(p+1)` to `_phi_prox`.

**Key math** (derivatives w.r.t. `t = |u|`, for `t > 0`):
- Full penalty gradient: `q * t^{q-1} * dphi(t^q)`
- Correction 2nd derivative: `q(q-1) * t^{q-2} * (dphi(t^q)-1) + q^2 * t^{2q-2} * ddphi(t^q)`
- Proximal of `mu * |.|^q` for `q < 1`: Newton solve of `t + mu*q*t^{q-1} = |v|` with threshold `t* = [mu*q*(1-q)]^{1/(2-q)}`

### SSN Proximal Dead Zone Bug (p != 1, OPEN)

**Status**: Diagnosed but NOT fixed. SSN currently broken for `power != 1` (q < 1).

**Stashed fix attempt**: `git stash@{0}` contains a partial fix (DPc clamping + active set preservation + dead zone zeroing). It was stashed because it also broke the regular `power=1` case. Apply with `git stash pop`, discard with `git stash drop`.

**Symptom**: SSN line search fails on every step when `power != 1` (e.g., `power=2.1`, `q=0.645`). Train loss decreases only from coordinate descent warm-start; SSN contributes nothing.

**Root cause 1 — Proximal jump discontinuity (two-branch problem)**:
The proximal of `mu * |.|^q` for `q < 1` has a jump discontinuity. The stationarity condition `t + mu*q*t^{q-1} = v` has **two roots** for `v > v_thresh`: `t_large` (local min, SOC > 0) and `t_small` (local max, SOC < 0). `_compute_prox` always returns `t_large`.

SSN's `_initialize_q` inverts the FOC by plugging `t = |u_i|` to get `q_var_i`. For small weights (`|u_i| < t*`), `|u_i|` is the `t_small` root. Then `_compute_prox(q_var)` returns `t_large != |u_i|`, so `prox(q) != params` — the fundamental SSN consistency assumption breaks.

For `q=1`, the equation `t + mu = v` has a unique root (no ambiguity).

**Root cause 2 — Indefinite Jacobian DG**:
For `q < 1`, the proximal Jacobian `DPc_i = 1/SOC_i` can exceed 1 when `SOC_i < 1` (weights near the dead-zone boundary). This makes the generalized Jacobian DG indefinite:
```
DG_{ii} = c + (H_data_{ii} - c) * DPc_i
```
When `DPc_i > 1` and `H_data_{ii} < c`, `DG_{ii} < 0`. The Newton direction becomes an ascent direction, so the line search can never succeed.

For `q=1`, `DPc_i` is always in `{0, 1}` (a projection), so `DG` is always PSD.

**Root cause 3 — Inactive weight activation via proximal jump**:
For inactive weights (`u_i = 0`), `_initialize_q` sets `q_i = -(1/c) * grad_flat_i`. If `|grad_flat_i| / c > v_thresh`, `prox(q_i)` jumps to a nonzero value, activating the neuron. For `q=1` this is smooth (soft thresholding continuous). For `q<1` it's a discontinuous jump from 0 to `~t*`.

**Key threshold**: `t* = [mu*q*(1-q)]^{1/(2-q)}` where `mu = alpha/c`, `c = 1 + alpha*gamma`. For `alpha=1e-5, gamma=0, q=0.645`: `t* ~ 7e-5`. Weights below this are in the dead zone.

**Stashed fix approach** (in `git stash@{0}`):
1. Dead-zone zeroing: zero weights where SOC <= 0 before SSN, force `q[dead_zone] = 0`
2. DPc clamping: clamp `_compute_dprox` diagonal to `[0, 1]` in `_DG` to keep Jacobian PSD
3. Active set preservation: force `unew[inactive] = 0` after prox to prevent jump activation
4. Inactive q clamping: force `q[inactive] = 0` for all zero-weight entries

**Why the stashed fix failed**: The combination of changes also broke the `power=1` (q=1) case. The changes to `_initialize_q`, `_initilize_G`, and `_DG` for general q altered the code paths used by q=1 as well. Need to ensure backward compatibility before re-applying.

**Visualization**: `scripts/visualize_proximal_deadzone.py` produces a 4-panel figure showing the FOC, SOC, proximal objective, and jump discontinuity.

## Directory Notes

- `src/` — Active code (use this, not `scr/` or `outdated/`)
- `scr/` — Stale copy, do not modify
- `outdated/` — Deprecated experiments
- `logs/` — Session documentation (see `2025-02-15.md` for debugging history)
