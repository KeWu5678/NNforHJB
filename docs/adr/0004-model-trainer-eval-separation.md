---
status: accepted
---

# Models are parametrizations; PDAP is the trainer; evaluation is pure functions

## Context

`src/models/signed.py` and `src/models/semiconcave.py` had each grown to carry
five concerns at once: the network parametrization (forward, weights), the SSN
optimization (both models `from ..SSN import SSN`), the warm-start coordinate
descent, the loss + nonconvex regularizer, and the relative-error evaluation.
The two models duplicated the warm-start (~75 vs ~54 lines, differing only in a
sign convention and a nonnegativity clamp) and the relative-error computation
(identical arithmetic), and they each embedded an SSN solve.

Reading the two SSN solves side by side shows the models are the *same kind of
object*. Both are **linear in their outer parameters θ** for the outer solve:
the data Hessian is `(w1/Nx)·Φvᵀ Φv + (w2/Nx)·Φgᵀ Φg` in both cases — the signed
model's `Φv` is its network matrix and `Φg` its gradient kernel; the semiconcave
model appends structural columns for the curvature `C` and affine `a, b0`. The
differences reduce to (1) which feature columns exist, (2) the penalized/nonneg
masks, (3) sign convention — and (3) is just the nonneg mask. The duplication
existed because nobody factored out "the linear-in-θ SSN problem."

`V = ½C‖x‖² − g(x)` for the semiconcave model is a composition of submodules
(quadratic head + shallow network + affine), each linear in its own parameters,
so it needs no bespoke analytic feature code: predictions and the constant
Jacobians `(Φv, Φg)` can come from autodiffing `forward`. SSN is retained for
*both* models — it is the prox-Newton solver for the nonconvex sparse
regularizer that defines this project; a smooth optimizer (Adam/SGD) cannot
deliver the sparsity or handle the nonsmooth penalty, and the signed model needs
SSN for the identical reason.

## Decision

Separate the three concerns along these layers:

- **`src/models/`** — parametrizations only: `forward` / predictions, named
  parameters, and a tag for which parameters are penalized / nonnegative. No
  import of `..SSN`. The semiconcave model becomes a composed network; its
  Jacobians come from autodiff (constant per support, recomputed only when the
  support changes at insertion), retiring the hand-built feature maps.
- **`src/PDAP/`** — the trainer and the loop-in-loop
  (`insert → warm-start → SSN → prune → insert`). One generic SSN outer solve and
  one generic warm-start, driven by the model's Jacobians and masks.
- **`src/eval.py`** — performance evaluation as pure functions of
  `(prediction, target)` tensors: relative L2/gradient/H1 errors and the
  value/gradient data-loss split. No model state. The nonconvex penalty is *not*
  here — it depends on which parameters a model penalizes, so it stays the
  model's responsibility and is added to the data loss to form the objective.
- **`src/data.py`** — value-sample loading, normalization, and the train/valid
  split (`split_value_samples`), surfaced as `DataConfig.train_fraction`.

This revises ADR-0003's note that "model-level relative errors may remain in the
model/PDAP training path until there is a concrete extraction need": that need
has arrived (verbatim duplication across two models). Evaluation moves to a
dedicated `src/eval.py` rather than into `src/metric.py`, which ADR-0003 reserves
as a reporting/table utility — `metric.py` remains a presentation layer and does
not own performance evaluation.

## Consequences

The change lands as small, golden-backed steps (`tests/test_pdap_equivalence.py`
pins exact summaries), regenerating the golden only when a step deliberately
changes numerics:

1. Extract evaluation to `src/eval.py`; models and PDAP call it; delete the
   duplicated `_compute_relative_errors`. Behavior-preserving (golden unchanged).
2. Delete the dead non-SSN path in `signed.py` (`train`, the generic-optimizer
   branch, SGD fallback, `optimizer_type`); PDAP always uses SSN.
3. Move warm-start into `src/PDAP/` as one generic function.
4. Move the SSN outer solve into `src/PDAP/`, driven by model Jacobians.
5. Rewrite the semiconcave model as a composed network with autodiff Jacobians.
6. Define the thin model contract; `SignedModel` becomes an adapter over
   `ShallowNetwork`.

Step 1 is implemented. The trade-off is a multi-step migration with several
golden regenerations, against the benefit of removing the duplication, fixing the
layering inversion (networks no longer depend on their optimizer), and giving
evaluation a single home.
