"""Convenience aliases for the historical PDPA variants.

Each alias expands to a ``(model, insertion)`` configuration of :class:`PDAP`:

    v2 -> signed network,     profile-threshold insertion
    v1 -> semiconcave model,  profile-threshold insertion
    v3 -> signed network,     finite-step insertion (q < 1)

Descriptive names map to the same bundles.
"""

from __future__ import annotations

from .pdap import PDAP

ALIASES = {
    "v2": {"model": "signed", "insertion": "profile"},
    "v1": {"model": "semiconcave", "insertion": "profile"},
    "v3": {"model": "signed", "insertion": "finite_step"},
    "signed": {"model": "signed", "insertion": "profile"},
    "semiconcave": {"model": "semiconcave", "insertion": "profile"},
    "finite_step": {"model": "signed", "insertion": "finite_step"},
}


def from_alias(alias: str, data: dict, alpha: float, gamma: float, power: float, **kwargs) -> PDAP:
    """Construct a PDAP from a named bundle (e.g. ``"v1"``/``"v2"``/``"v3"``)."""
    if alias not in ALIASES:
        raise ValueError(f"unknown PDAP alias {alias!r}; choices: {sorted(ALIASES)}")
    cfg = ALIASES[alias]
    return PDAP(data, alpha=alpha, gamma=gamma, power=power, **cfg, **kwargs)
