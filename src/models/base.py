"""The model contract the PDAP trainer depends on.

A model is a *parametrization*, nothing more: it owns the value/gradient
prediction, the atom support, and -- because the outer solve is linear in the
outer parameters ``theta`` -- the feature maps (Jacobians), the ``theta``
packing, and the penalty masks. It owns no training algorithm: the warm start
(:mod:`src.PDAP.warmstart`), the SSN solve (:mod:`src.PDAP.ssn_solve`), and the
evaluation (:mod:`src.eval`) live in the trainer.

``SignedModel`` and ``SemiconcaveModel`` both satisfy this structurally, so it is
a :class:`typing.Protocol` (no inheritance required). ``@runtime_checkable`` lets
tests assert conformance with ``isinstance``.
"""

from __future__ import annotations

from typing import Iterator, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class PDAPModel(Protocol):
    # --- forward-defining scalars ---
    # power defines the activation sigma^p; q = 2/(power+1) is the induced penalty
    # exponent (the prox closed-forms depend on it).  The objective and SSN-solver
    # hyperparameters are the trainer's, not the model's.
    power: float
    q: float
    input_dim: Optional[int]

    # --- the model is an nn.Module: theta is its trainable parameters ---
    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]: ...

    # --- atom support ---
    @property
    def n_neurons(self) -> int: ...
    def set_atoms(self, W: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> None: ...
    def get_atoms(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    # --- prediction ---
    def predict_tensors(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: ...
    def predict(self, x) -> Tuple[np.ndarray, np.ndarray]: ...

    # --- linear-in-theta interface for the SSN solve ---
    # theta (the SSN working vector) is the model's trainable parameters, read and
    # written by the trainer with torch's parameters_to_vector / vector_to_parameters
    # built-ins -- so the contract carries only what those can't express: the
    # feature maps and the penalty/nonneg structure.
    def jacobians(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: ...
    def penalty_masks(self) -> Tuple[torch.Tensor, torch.Tensor]: ...
