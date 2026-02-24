# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sparse neural network training framework for solving Hamilton-Jacobi-Bellman (HJB) equations using primal-dual proximal algorithms (PDPA) with non-convex regularization and semismooth Newton (SSN) optimization. The Python implementation ports and extends a MATLAB reference at `/Users/ruizhechao/Documents/NonConvexSparseNN/`.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run experiments via notebooks
jupyter notebook notebook/pdpa_vdp.ipynb

# Run a quick test from CLI
python -c "from src.PDPA_v2 import PDPA_v2; print('import ok')"
```

No dedicated test suite exists; testing is done through Jupyter notebooks in `notebook/`.

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
- **`ssn.py`** — Semismooth Newton optimizer. Assumes NOC holds; requires non-zero initial weights. Uses data-only gradient (autograd gives full objective; SSN adds regularization separately).
- **`utils.py`** — Penalty `_phi(t, th, gamma)`, derivatives `_dphi`/`_ddphi`, proximal operators `_compute_prox`, `_phi_prox` (non-convex), stereographic projection.

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
- **Parameterization**: MATLAB optimizes in stereographic R^d, Python on sphere S^d
- **Postprocess**: MATLAB merges by Euclidean distance in stereographic space; Python merges by cosine similarity on S^d
- **Gamma sweep**: MATLAB warm-starts each gamma from previous solution; Python runs independently

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

## Directory Notes

- `src/` — Active code (use this, not `scr/` or `outdated/`)
- `scr/` — Stale copy, do not modify
- `outdated/` — Deprecated experiments
- `logs/` — Session documentation (see `2025-02-15.md` for debugging history)
