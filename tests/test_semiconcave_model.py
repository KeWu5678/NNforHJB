"""Tests for the semiconcave model and its SSN optimiser."""

import numpy as np
import torch

from src.SSN import SSN
from src.eval import relative_errors
from src.models.semiconcave import SemiconcaveModel


def _atoms(n, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(n, d, generator=g, dtype=torch.float64)
    W = W / W.norm(dim=1, keepdim=True)
    b = torch.randn(n, generator=g, dtype=torch.float64)
    return W, b


def test_ssn_semiconcave_recovers_sparse_nonneg_with_free_coord():
    """Penalised c_i go nonneg/sparse; the unpenalised free coord is not shrunk."""
    A = torch.tensor(
        [[1.0, 0.0, 0.3], [0.2, 1.0, 0.0], [0.0, 0.4, 1.0], [0.5, 0.5, 0.5], [1.0, 1.0, 0.2]],
        dtype=torch.float64,
    )
    theta_true = torch.tensor([2.0, 0.0, -1.5], dtype=torch.float64)
    y = A @ theta_true
    H = A.T @ A
    theta = torch.nn.Parameter(torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64).reshape(1, -1))
    pen = torch.tensor([True, True, False])
    # nonneg_mask defaults to all-False on the base SSN, so pass it explicitly to
    # exercise the nonnegative-prox path the semiconcave model relies on.
    opt = SSN([theta], alpha=1e-2, gamma=1.0, penalized_mask=pen, nonneg_mask=pen, th=0.5, power=1.0)
    opt.data_hessian = H

    def closure():
        opt.zero_grad()
        r = A @ theta.reshape(-1) - y
        return 0.5 * (r @ r)

    for _ in range(15):
        opt.step(closure)
    th = theta.detach().reshape(-1)
    assert bool((th[:2] >= -1e-9).all())          # nonneg c
    assert abs(float(th[1])) < 1e-6               # sparsified
    assert float(th[2]) < -1.0                    # free coord stays negative, not shrunk to 0


def test_predict_matches_linear_features():
    d, N, n = 2, 40, 4
    x = torch.randn(N, d, dtype=torch.float64)
    W, b = _atoms(n, d)
    m = SemiconcaveModel(alpha=1e-3, gamma=1.0, power=1.0, verbose=False)
    m.set_atoms(W, b, torch.tensor([1.0, 0.5, 0.0, 2.0], dtype=torch.float64))
    m.C = 1.3
    m.affine_w = torch.tensor([0.2, -0.4], dtype=torch.float64)
    m.affine_b = 0.7
    Phi_v, Phi_g, _ = m._build_features(x)
    theta = m._theta_vector(n)
    Vp, dVp = m.predict_tensors(x)
    assert torch.allclose(Phi_v @ theta, Vp.reshape(-1), atol=1e-10)
    assert torch.allclose(Phi_g @ theta, dVp.reshape(-1), atol=1e-10)


def test_augmented_hessian_matches_autograd():
    d, N, n = 2, 40, 4
    x = torch.randn(N, d, dtype=torch.float64)
    W, b = _atoms(n, d, seed=3)
    m = SemiconcaveModel(alpha=1e-3, gamma=1.0, power=1.0, verbose=False)
    m.set_atoms(W, b, torch.zeros(n))
    Phi_v, Phi_g, _ = m._build_features(x)
    Vt = torch.randn(N, dtype=torch.float64)
    dVt = torch.randn(N * d, dtype=torch.float64)
    Nx = N * d
    H = (1.0 / Nx) * (Phi_v.T @ Phi_v) + (1.0 / Nx) * (Phi_g.T @ Phi_g)

    def dloss(th):
        rv = Phi_v @ th - Vt
        rg = Phi_g @ th - dVt
        return (1.0 / (2 * Nx)) * (rv @ rv) + (1.0 / (2 * Nx)) * (rg @ rg)

    Hau = torch.autograd.functional.hessian(dloss, m._theta_vector(n))
    assert torch.allclose(H, Hau, atol=1e-10)


def test_train_ssn_recovers_synthetic_semiconcave_target():
    d, N, n = 2, 80, 4
    x = torch.randn(N, d, dtype=torch.float64)
    W, b = _atoms(n, d, seed=7)
    truth = SemiconcaveModel(alpha=1e-4, gamma=1.0, power=1.0, verbose=False)
    truth.set_atoms(W, b, torch.tensor([1.5, 0.0, 0.8, 0.0], dtype=torch.float64))
    truth.C = 2.0
    truth.affine_w = torch.tensor([0.3, 0.1], dtype=torch.float64)
    truth.affine_b = -0.5
    V, dV = truth.predict_tensors(x)

    fit = SemiconcaveModel(alpha=1e-4, gamma=1.0, power=1.0, verbose=False)
    fit.set_atoms(W, b, torch.full((n,), 0.1, dtype=torch.float64))
    fit.C = 0.5
    fit.affine_w = torch.zeros(d, dtype=torch.float64)
    fit.affine_b = 0.0
    fit.train_ssn(x, V, dV, iterations=25)

    _, _, h1 = relative_errors(*fit.predict_tensors(x), V, dV)
    assert h1 < 1e-2
    assert abs(fit.C - 2.0) < 0.05
    assert bool((fit.c >= -1e-9).all())
