"""Build a PDAP model from a config section.

The model construction (signed vs semiconcave, activation resolution, input
dimension) belongs with the script that owns the run, not the trainer.  This is
the single place that maps a ``cfg.model`` section to a constructed model.
"""

from __future__ import annotations

from ..config.activations import get_activation
from .semiconcave import SemiconcaveModel
from .signed import SignedModel


def build_model(cfg, input_dim: int):
    """Construct the model named by ``cfg.model`` with its input dimension set."""
    m = cfg.model
    activation = get_activation(m.activation)
    verbose = cfg.env.verbose
    if m.kind == "signed":
        model = SignedModel(activation=activation, power=m.power, verbose=verbose)
    elif m.kind == "semiconcave":
        model = SemiconcaveModel(
            power=m.power, activation=activation, c_init=m.c_init, verbose=verbose,
        )
    else:
        raise ValueError(f"model.kind must be 'signed' or 'semiconcave', got {m.kind!r}")
    model.input_dim = input_dim
    if isinstance(model, SemiconcaveModel):
        model._ensure_affine(input_dim)
    return model
