# CLAUDE.md

Guidance for Claude Code working in this repository. This file is the top-level
map; deeper technical detail lives in `vault/` (linked per section).

## Project Overview
Sparse shallow-network framework for solving Hamilton-Jacobi-Bellman (HJB)
equations from value/gradient samples, using a primal-dual proximal algorithm
(PDAP) with non-convex regularization and a semismooth Newton (SSN) outer solve.
Ports and extends a MATLAB reference at `/Users/ruizhechao/Documents/NonConvexSparseNN/`.

## References
Thesis & papers: `/Users/chaoruiz/Documents/NotePaper/MasterThesis/`
- **Non-convex regularization** (main): `nonconvexity/noncovex regulerization.pdf`
- **Gradient-augmented regression** (main): `PDE/Gradient-Augmented Regression.pdf`
- Supporting: `nonconvexity/` (integral representation of convex functions on
  measures, l.s.c. of the regularizer, relaxation of non-convex functionals),
  `PDE/` (Barron, char_HJB).

## Research Directions
Each line: **objective / tried / result / reference**.

1. **Activation search — smooth VDP HJB.** Find the best accuracy↔sparsity
   activation on the smooth VDP value / 132 variants with PDPA_v2 (power=1, H1,
   gamma∈[0,.01,.1,1,10]) / softplus = best sparse family, GELU/Mish lower H1 at
   ~2× neurons, SmoothReLU best H1 but dense /
   `autoresearch/ActivationSearch/data:VDP/SUMMARY.md`.
2. **Activation search — discontinuous gradient.** Best activation at a ∇V jump
   (near_grad) / activation families on analytic `x1 + x2|x2|/2 = 0`, near/far
   masks / spherical leaky squared-ReLU (`leaky_relu2`, sphere, α≈0.02) wins /
   `autoresearch/ActivationSearch/data:analytical/SUMMARY.md`.
3. **Non-convexity (gamma) effect.** Quantify how the non-convex penalty gamma
   shifts accuracy/sparsity / gamma sweep across activations on both datasets /
   effect is activation-specific (largest for weak sharp/smooth-gated acts); the
   "more neurons ⇒ bigger gamma effect" rule holds only locally; squared-ReLU is
   gamma-stable / `autoresearch/gamma_effect_summary.md`,
   `autoresearch/gamma_pattern_check.md`.
4. **Semiconcave reference (analytic).** Validate a semiconcavity-aware model on a
   known semiconcave target (min of Gaussians, diagonal switching) / semiconcave
   vs signed PDPA_v2 over gammas/seeds + convex activation sweep / model trains
   cleanly, leaky_relu top activation; baseline that motivated the pendulum study
   / `autoresearch/SemiconcaveFittingComparison/VDPReference/`.
5. **Pendulum swing-up — semiconcave vs pure NN at a real discontinuity.** Test
   whether enforcing semiconcavity helps a paper-backed Lipschitz-V /
   discontinuous-∇V value (Han–Yang) / PDAP semiconcave model vs signed model
   under the *same* SSN pipeline on PMP (∞-horizon) and transient (T=3) data,
   gammas, seeds 42–44 / semiconcavity does **not** help at the discontinuity
   (near-grad 2.98 vs 2.22 on PMP); single global ‖x‖² envelope mismatched to the
   periodic multi-well value; tie on smooth transient /
   `autoresearch/SemiconcaveFittingComparison/PendulumSwingUp/extended_semiconcave_runs/`, `vault/semiconcave_model.md`,
   `vault/pendulum_bb_tpbvp.md`, `CONTEXT.md`.

## Code — core algorithm
PDAP outer loop (`PDAP.fit`, 15–20 iters):
1. **Insertion** — sample S^d, L-BFGS-maximize the dual profile, merge duplicates,
   accept atoms (signed model: `|p(ω)|>α` two-sided; semiconcave: `p(ω)>α`
   one-sided/convex; `finite_step`: accept where `ΔJ(c*;ω)<0`).
2. **Warm-start** — coordinate descent gives new neurons non-zero outer weights
   (`finite_step` skips it: `c*` comes from the insertion criterion).
3. **SSN** — semismooth Newton on outer weights, inner weights frozen (~20 iters).
4. **Prune** — merge near-duplicates (cosine sim on S^d), drop proximal zeros.

**One `PDAP` class, two axes** (`PDAP(model=, insertion=)`; `from_alias("v1"/"v2"/"v3")`):
- `model="signed"` — pure signed network `V = Σ c_i σ(w_i·x+b_i)^p`, all `c_i` penalized.
- `model="semiconcave"` — `V = 0.5·C·‖x‖² − g(x)`, convex `g` (`c_i ≥ 0`) + affine;
  `C`/affine trainable but **unpenalized**.
- `insertion="profile"` (dual-threshold) | `"finite_step"` (ΔJ<0, for q<1).
- Aliases: `v2`=signed+profile, `v1`=semiconcave+profile, `v3`=signed+finite_step.

Modules (`src/`): `net.py`, `utils.py`, `metric.py`, the `src/SSN/` optimizer
package, the `src/models/` model package, the `src/PDAP/` outer-loop package, plus
the open-loop data subsystem in `src/OpenLoop/` (solvers, the pendulum PMP sampler,
and the VDP / pendulum dataset generators).

The **`src/SSN/` package** is one configurable semismooth-Newton optimizer
(layout mirrors `torch.optim`): `optimizer.py` (`SSN`), `strategies.py`
(`levenberg_marquardt` damped-Newton / `steihaug_cg` trust-region globalizations,
selected by `method=`), `prox.py` + `penalty.py` (kernels, re-exported by
`utils.py`), `mpcg.py`. The signed network, the semiconcave model (via
`penalized_mask`/`nonneg_mask`), and the old trust-region variant are all
configurations of this single class — the former `ssn.py`/`ssn_tr.py`/
`ssn_semiconcave.py` no longer exist.

The **`src/models/` package** holds the parametric value-function models behind a
shared `Model` protocol (`base.py`): `signed.py` (`SignedModel`) and
`semiconcave.py` (`SemiconcaveModel`). Both expose the uniform interface the loop
drives — `set_atoms`/`get_atoms`/`warm_start`/`fit_outer_weights`/`predict_tensors`.
The former `model.py`/`semiconcave_model.py` no longer exist.

The **`src/PDAP/` package** is the unified outer loop: `pdap.py` (`PDAP` + `fit`),
`insertion.py` (`profile_threshold` / `finite_step` strategies + `solve_insertion_weight`),
`registry.py` (aliases). The former `PDPA.py`/`PDPA_v1.py`/`PDPA_v2.py`/`PDPA_v3.py`
no longer exist.

**Technical detail → `vault/`:**
- Full pipeline, MATLAB mapping, known differences, critical parameters, gotchas
  (loss `Nx=N·d`, data-only SSN gradient, sphere parameterization, data scaling),
  and a script index: `vault/algorithm.md`.
- Power-q penalty `φ(|u|^q)`, `q=2/(p+1)`, and the proximal dead-zone bug (p≠1,
  OPEN): `vault/power_q_penalty.md`.
- Semiconcave model, the masked-nonneg-prox / unpenalized-coords configuration of
  `SSN`, augmented Hessian, envelope-mismatch finding: `vault/semiconcave_model.md`.
- Pendulum open-loop TPBVP/PMP derivation: `vault/pendulum_bb_tpbvp.md`.
- Session/debugging history: `vault/LOGS.md`.

## Directory Notes
- `src/` — active code; `scripts/` — experiment runners & dataset generators (index in `vault/algorithm.md`).
- `autoresearch/` — research outputs (per-direction subfolders + summaries).
- `vault/` — layered technical docs (this file links them).
- `rawdata/` — generated datasets/plots; `outdated/` — deprecated experiments.
- `CONTEXT.md` — domain language / example-selection notes for the open-loop study.
