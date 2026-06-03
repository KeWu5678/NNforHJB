"""Descriptive configuration aliases for the unified PDAP outer loop."""

from __future__ import annotations

from .pdap import PDAP

ALIASES = {
    "signed": {"model": "signed", "insertion": "profile"},
    "semiconcave": {"model": "semiconcave", "insertion": "profile"},
    "finite_step": {"model": "signed", "insertion": "finite_step"},
}


def from_alias(alias: str, data: dict, alpha: float, gamma: float, power: float, **kwargs) -> PDAP:
    """Construct a :class:`PDAP` from a descriptive configuration alias."""
    if alias not in ALIASES:
        raise ValueError(f"unknown PDAP alias {alias!r}; choices: {sorted(ALIASES)}")
    cfg = ALIASES[alias]
    return PDAP(data, alpha=alpha, gamma=gamma, power=power, **cfg, **kwargs)
