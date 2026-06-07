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
6. Define the thin model contract (`src/models/base.py:PDAPModel`); `SignedModel`
   becomes an adapter over `ShallowNetwork`.

All six steps are implemented. In the event, only step 1's split changed
numerics (a real train/validation split, golden regenerated once); steps 2-6 all
landed bit-exact, including the semiconcave autodiff Jacobians (autograd runs the
same operations the analytic feature maps did). The result: the duplication is
gone, the layering inversion is fixed (neither model imports the SSN optimizer),
evaluation has a single home, and the trainer depends on one explicit model
contract.

## Follow-up: models as nn.Modules; objective/solver config out of the model

Pushing the same principle (model = parametrization, use the framework) further,
three more steps, all bit-exact:

7. **Both models are `nn.Module`s.** `SignedModel` *subclasses* `ShallowNetwork`
   (it is the shallow net, not a wrapper around `self.net`); `SemiconcaveModel`
   is an `nn.Module` with `c, C, a, b0` as parameters registered in θ-order and
   `W, b` as buffers. `theta` (the SSN working vector) is just the trainable
   parameters, read/written with torch's `parameters_to_vector` /
   `vector_to_parameters` — so `get_theta` / `set_theta` leave the contract.
8. **The objective leaves the model.** `compute_loss` is removed; the trainer
   computes the objective (`PDAP._record_loss` for the data term via
   `src.eval.data_loss_terms`, plus `src.PDAP.ssn_solve.nonconvex_penalty` for the
   regularizer — the same penalty the SSN closure uses, so it exists once).
9. **Objective and solver hyperparameters leave the model.** `alpha, gamma, th,
   loss_weights` and `lr, method, max_ls_iter, tolerance_*, sigmamax` move into
   trainer dataclasses (`Objective`, `SolverConfig` in `ssn_solve`), built by PDAP
   from config and threaded to the solve / warm-start / recording. Model
   constructors take only forward-defining parameters (`activation`, `power`,
   plus `c_init`/`dtype` for semiconcave); `power` stays because it defines
   `σ^p` and induces `q = 2/(power+1)`.

The `PDAPModel` contract is now minimal: `power, q, input_dim, last_fit_summary`,
`parameters()`, and the methods `set_atoms / get_atoms / predict / predict_tensors
/ jacobians / penalty_masks`. `jacobians` and `penalty_masks` are the only
model-specific surface the trainer can't obtain from `nn.Module` built-ins —
`jacobians` is the feature-map computation, `penalty_masks` the irreducible
penalty/nonneg structure (the nonneg part *is* semiconcavity, so it cannot be
flattened away). Bit-exactness held because building `nn.Parameter`s consumes no
RNG (unlike `nn.Linear`, whose draw `set_atoms` reuses via `ShallowNetwork`'s
constructor) and `parameters()` yields θ in the same `[c | C | a | b0]` order the
old hand-packing used.

## Penalty selection stays config-driven

The penalty is `alpha * Σ phi(|c|^q)` with two independent knobs — the power-q
exponent (`q = 2/(power+1)`) and the log-nonconvex `phi` (`gamma`, `th`). We
considered a named `penalty: l1 | power_q | log | composed` selector (which would
have decoupled the penalty exponent from the activation power so `log`/`l1` force
`q = 1`), but rejected it: it adds an enum plus validation and a second notion of
`q` for a choice the existing parameters already express. Instead the two
penalties this project uses are selected by parameter values, documented inline in
`ModelConfig`:

- **power penalty** `alpha * Σ |c|^q` — `gamma = 0` (phi is the identity),
  `power > 1` (so `q < 1` is non-convex).
- **log penalty** `alpha * Σ phi(|c|)` — `power = 1` (so `q = 1`), `gamma > 0`.

So the penalty is configurable but not a named axis; `power` and `gamma` remain
its controls, and the penalty exponent stays coupled to the activation power.
