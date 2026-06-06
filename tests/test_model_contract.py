"""Both models satisfy the PDAPModel contract the trainer depends on."""

import torch

from src.models.base import PDAPModel
from src.models.signed import SignedModel
from src.models.semiconcave import SemiconcaveModel


def _atoms(n=4, d=2):
    W = torch.randn(n, d, dtype=torch.float64)
    W = W / W.norm(dim=1, keepdim=True)
    b = torch.randn(n, dtype=torch.float64)
    c = torch.rand(n, dtype=torch.float64)
    return W, b, c


def test_signed_model_is_pdap_model():
    m = SignedModel(alpha=1e-4, gamma=0.1, power=1.0, verbose=False)
    m.input_dim = 2
    m.set_atoms(*_atoms())
    assert isinstance(m, PDAPModel)


def test_semiconcave_model_is_pdap_model():
    m = SemiconcaveModel(alpha=1e-4, gamma=0.1, power=1.0, verbose=False)
    m.input_dim = 2
    m.set_atoms(*_atoms())
    assert isinstance(m, PDAPModel)


def test_both_models_predict_to_numpy():
    """predict() returns numpy (V, dV) for both — the gap SignedModel had filled."""
    x = torch.randn(6, 2, dtype=torch.float64).numpy()
    for cls in (SignedModel, SemiconcaveModel):
        m = cls(alpha=1e-4, gamma=0.1, power=1.0, verbose=False)
        m.input_dim = 2
        m.set_atoms(*_atoms())
        V, dV = m.predict(x)
        assert V.shape == (6, 1) and dV.shape == (6, 2)
