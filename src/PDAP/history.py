"""The training record produced by ``PDAP.fit``.

``PDAP`` is a pure trainer: the per-iteration metrics and weight snapshots are
not its state, they are returned to the caller in a :class:`History` (the Keras
``fit``-returns-``History`` convention).  The metric *computation* is pure
evaluation (``src.eval``) plus the regularizer; ``History.record`` orchestrates
it once per iteration so the training loop stays free of evaluation code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch
from torch.nn.utils import parameters_to_vector

from ..eval import data_loss_terms, relative_errors
from .ssn_solve import Objective, nonconvex_penalty


def objective_value(model, objective: Objective, data) -> float:
    """The training objective (data fidelity + nonconvex penalty) at ``model``."""
    X, V, dV = data
    Vp, dVp = model.predict_tensors(X)
    data_loss = data_loss_terms(Vp, dVp, V, dV, objective.loss_weights)[0]
    theta = parameters_to_vector([p for p in model.parameters() if p.requires_grad]).detach()
    penalized, nonneg = model.penalty_masks()
    penalty = nonconvex_penalty(
        theta, penalized, nonneg,
        alpha=objective.alpha, th=objective.th, gamma=objective.gamma, q=model.q,
    )
    return float((data_loss + penalty).detach())


@dataclass
class History:
    """Per-iteration losses, relative errors, and support snapshots."""

    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    err_l2_train: List[float] = field(default_factory=list)
    err_l2_val: List[float] = field(default_factory=list)
    err_grad_train: List[float] = field(default_factory=list)
    err_grad_val: List[float] = field(default_factory=list)
    err_h1_train: List[float] = field(default_factory=list)
    err_h1_val: List[float] = field(default_factory=list)
    inner_weights: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    outer_weights: List[torch.Tensor] = field(default_factory=list)
    model_states: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    best_iteration: int = 0
    best_train_loss: float = float("inf")
    final_neurons: int = 0

    def record(self, model, objective: Objective, data_train, data_valid) -> None:
        """Evaluate the current model and append one iteration's record."""
        tl = objective_value(model, objective, data_train)
        vl = objective_value(model, objective, data_valid)
        self.train_loss.append(tl)
        self.val_loss.append(vl)
        l2t, gt, h1t = relative_errors(*model.predict_tensors(data_train[0]), *data_train[1:])
        l2v, gv, h1v = relative_errors(*model.predict_tensors(data_valid[0]), *data_valid[1:])
        self.err_l2_train.append(l2t); self.err_l2_val.append(l2v)
        self.err_grad_train.append(gt); self.err_grad_val.append(gv)
        self.err_h1_train.append(h1t); self.err_h1_val.append(h1v)
        W, b, c = model.get_atoms()
        self.inner_weights.append({"weight": W, "bias": b})
        self.outer_weights.append(c.reshape(1, -1))
        self.model_states.append(
            {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        )
        i = len(self.train_loss) - 1
        if tl < self.best_train_loss:
            self.best_train_loss = tl
            self.best_iteration = i

    @property
    def best_neurons(self) -> int:
        return int(self.inner_weights[self.best_iteration]["weight"].shape[0])

    @property
    def best_err_l2_train(self) -> float:
        return self.err_l2_train[self.best_iteration]

    @property
    def best_err_h1_train(self) -> float:
        return self.err_h1_train[self.best_iteration]

    def summary_metrics(self) -> dict[str, float | int]:
        """Scalar comparison metrics at the selected best iteration."""
        i = int(self.best_iteration)
        return {
            "rel_l2_train": float(self.err_l2_train[i]),
            "rel_l2_val": float(self.err_l2_val[i]),
            "rel_grad_train": float(self.err_grad_train[i]),
            "rel_grad_val": float(self.err_grad_val[i]),
            "rel_h1_train": float(self.err_h1_train[i]),
            "rel_h1_val": float(self.err_h1_val[i]),
            "best_iteration": i,
            "best_neurons": int(self.best_neurons),
            "final_neurons": int(self.final_neurons),
        }
