"""The parametric-model interface the PDAP outer loop depends on.

A ``Model`` represents the value function V(x) and exposes the hooks the PDAP
loop calls between insertion / SSN / prune steps.  This is a ``typing.Protocol``
(structural, not a base class): ``SignedModel`` and ``SemiconcaveModel`` already
satisfy the shared surface below without inheriting from it, so nothing is forced
to change — the protocol documents and type-checks the contract.

Note (Phase 3 of the consolidation): the *training* entry point is the one place
the two models still diverge by name — ``SignedModel.train(data_train, ...)`` vs
``SemiconcaveModel.train_ssn(x, V, dV, ...)``.  Unifying it to a single
``fit_outer_weights`` is deferred to when the PDAP ``fit`` loop is unified (the
signature should be designed together with that sole consumer, not in isolation).
That divergence is why it is intentionally absent from this protocol for now.
"""

from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable

import torch

TensorTriple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@runtime_checkable
class Model(Protocol):
    """Structural interface shared by all PDAP parametric models."""

    # --- configuration the PDAP loop reads ---
    input_dim: int | None
    power: float
    loss_weights: Tuple[float, float]
    th: float
    gamma: float

    def _prepare_data(self, data: dict) -> Tuple[TensorTriple, TensorTriple]:
        """Split a data dict into (train, valid) tensor triples (x, V, dV)."""
        ...

    def _compute_loss(self, x, V, dV):
        """Return (total_loss, value_loss, grad_loss) at the current parameters."""
        ...

    def _compute_relative_errors(self, x, V, dV):
        """Return (rel_l2, rel_grad, rel_h1) errors at the current parameters."""
        ...
