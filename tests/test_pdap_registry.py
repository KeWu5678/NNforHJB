from __future__ import annotations

import numpy as np
import pytest

from src.PDAP import ALIASES, from_alias
from src.models.semiconcave import SemiconcaveModel
from src.models.signed import SignedModel


def _data() -> dict:
    x = np.array([[0.0, 0.0], [1.0, -1.0]], dtype=np.float64)
    return {
        "x": x,
        "v": np.zeros(2, dtype=np.float64),
        "dv": np.zeros((2, 2), dtype=np.float64),
    }


def test_descriptive_aliases_are_registered_without_legacy_variant_names() -> None:
    assert ALIASES == {
        "signed": {"model": "signed", "insertion": "profile"},
        "semiconcave": {"model": "semiconcave", "insertion": "profile"},
        "finite_step": {"model": "signed", "insertion": "finite_step"},
    }
    assert "v1" not in ALIASES
    assert "v2" not in ALIASES
    assert "v3" not in ALIASES


def test_from_alias_constructs_registered_models() -> None:
    semiconcave = from_alias("semiconcave", _data(), alpha=1e-4, gamma=0.0, power=1.0, verbose=False)
    signed = from_alias("signed", _data(), alpha=1e-4, gamma=0.0, power=1.0, verbose=False)
    finite_step = from_alias("finite_step", _data(), alpha=1e-4, gamma=0.0, power=1.0, verbose=False)

    assert isinstance(semiconcave.model, SemiconcaveModel)
    assert semiconcave.insertion_kind == "profile"
    assert isinstance(signed.model, SignedModel)
    assert signed.insertion_kind == "profile"
    assert isinstance(finite_step.model, SignedModel)
    assert finite_step.insertion_kind == "finite_step"


def test_legacy_aliases_stay_removed() -> None:
    with pytest.raises(ValueError, match="unknown PDAP alias 'v1'"):
        from_alias("v1", _data(), alpha=1e-4, gamma=0.0, power=1.0, verbose=False)
