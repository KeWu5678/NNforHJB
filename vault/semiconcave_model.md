# Semiconcave Model (PDAP model="semiconcave")

Sub-level disclosure for `../CLAUDE.md`. The semiconcave parametric model, its
SSN optimiser, the augmented Hessian, and the main empirical finding.

## Model

```
V(x) = 0.5 * C * ||x||^2 - g(x),   g(x) = sum_i c_i * sigma(w_i.x + b_i)^p + a.x + b0
```
- `g` is a **convex** ReLU^p network (`c_i >= 0`, requires `power >= 1`) plus an
  affine term, so `V - 0.5*C*||x||^2 = -g` is concave => `V` is semiconcave.
- `C >= 0` (the quadratic envelope) and the affine `(a, b0)` are trainable but
  **unpenalised**; only the `c_i` carry the `alpha*phi` sparsity penalty.
- Motivation: value functions of optimal-control problems are semiconcave, so
  this is a value-function-aware prior.

`PDAP(model="semiconcave")` runs the *same* `fit()` loop as the signed model
(insert -> warm-start -> SSN -> prune); the model-forced pieces follow from the
semiconcave model's interface:
- residuals/insertion profiles computed from the semiconcave `V` (`predict_tensors`);
- one-sided insertion acceptance (`profile > alpha`, convex atoms carry `c >= 0`,
  i.e. `profile_threshold(two_sided=False)`);
- nonneg coordinate-descent warm-start (`SemiconcaveModel.warm_start`, `c >= 0`).

## SSN for the augmented parameter vector

Outer solve variable: `theta = [c (n) | C (1) | a (d) | b0 (1)]`, inner weights
frozen. Both value and gradient predictions are **linear** in `theta`, so the
data loss is a quadratic with a closed-form Hessian.

`src/models/semiconcave.py:SemiconcaveModel`
- `_build_features(x)` -> `Phi_v (N, P)`, `Phi_g (N*d, P)` with the model signs
  (`-phi_i`, `-grad phi_i` for atoms; `0.5||x||^2`, `x` for C; `-x`, `-I_d` for a;
  `-1`, `0` for b0).
- Augmented data Hessian `H = (w1/Nx) Phi_v^T Phi_v + (w2/Nx) Phi_g^T Phi_g`
  (verified equal to the autograd Hessian of the data loss to ~1e-10).
- `train_ssn` assembles `theta`, sets `optimizer.data_hessian = H`, and runs
  `SSN(..., penalized_mask=, nonneg_mask=)`; the closure returns the **full**
  objective (data + penalty on the `c` block) because `SSN` subtracts the penalty
  gradient to recover the data-only gradient.

The semiconcave model is now a **configuration of the unified `src/SSN`**, not a
subclass.  (Historically it was `SSN_semiconcave(SSN)`, kept separate because
editing the shared SSN once broke the q=1 path — see `power_q_penalty.md`; the
merge was gated by a bit-identical golden test, `tests/test_ssn_equivalence.py`.)
Its three differences from a signed SSN, all driven by the masks:
- **per-coordinate `alpha`** (`alpha_vec`: `alpha` on the `c` block, `0` elsewhere)
  so only `c` is penalised; the prox/penalty kernels broadcast a tensor `alpha`/`mu`.
- **nonnegative proximal** on `nonneg_mask` (the `c` block and `C`): the masked
  `_prox` reuses the signed `_compute_prox` then zeros entries whose proximal
  preimage is `<= 0`.
- **unpenalised-coordinate fix**: for coords with no penalty the proximal is the
  identity, so the SSN preimage must be `q_var = params` (not
  `params - (1/c) grad_data`). Otherwise `G` collapses to `0` on those coords and
  the Newton step never moves `C`/affine. This is the key correctness fix.

## Main empirical finding (pendulum)

Under the *same* SSN pipeline (so optimizer is controlled), on the Han-Yang
pendulum swing-up value (Lipschitz `V`, discontinuous `dV`):

| dataset | model | rel_h1 | grad err NEAR switch | far | near/far |
|---|---|---|---|---|---|
| PMP (inf-horizon) | pure NN (v2) | 0.501 | 2.218 | 0.973 | 2.29 |
| PMP | semiconcave (v1) | 0.664 | 2.976 | 1.580 | 1.89 |
| transient (T=3) | pure NN (v2) | 0.325 | 0.307 | 0.185 | 1.67 |
| transient | semiconcave (v1) | 0.318 | 0.310 | 0.177 | 1.77 |

(means over seeds 42-44, best gamma; `autoresearch/SemiconcaveFittingComparison/data: pendulum/extended_semiconcave_runs/`.)

**Enforcing semiconcavity does not help at the discontinuity** — its near-switch
gradient error is actually higher (2.98 vs 2.22 on PMP) and it is worse
everywhere; on the smooth transient set the two tie.

**Why**: (1) the `dV` jump is approximated by smooth atoms in *both* models, so
the convexity prior adds no resolution at the kink; (2) a **single global,
isotropic `0.5*C*||x||^2` envelope** is mismatched to the pendulum's periodic,
multi-well value (wells at `theta = 0, +/-2pi`). To stay semiconcave the
optimizer drives `C ~ 30-38` in normalized units while `V_norm <= 1`, so an O(1)
value is reconstructed as the near-cancellation of two ~35-magnitude terms
(huge quadratic minus huge convex `g`), wasting precision and inflating error.

**Proposed fix (untried)**: replace the global `||x||^2` envelope with a
**localized / periodic curvature** term (e.g. `C*(1 - cos theta) + 0.5*C_omega*omega^2`)
matched to the geometry, so `C` stays O(1) and there is no cancellation. This is
an envelope-only change; the convex network, insertion, SSN, and nonneg
constraint are unchanged. It still will not sharpen the jump itself, but removes
the structural penalty so the comparison is fair.

## Tests
`tests/test_semiconcave_model.py`: SSN nonneg/sparse recovery with a free
unpenalised coord; `predict == linear features`; augmented Hessian == autograd
Hessian; recovery of a synthetic semiconcave target.
