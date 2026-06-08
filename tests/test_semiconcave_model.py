"""Tests for the semiconcave model and its SSN optimiser."""

import numpy as np
import torch

from torch.nn.utils import parameters_to_vector

from src.SSN import SSN
from src.PDAP.ssn_solve import ssn_solve, Objective, SolverConfig
from src.eval import relative_errors
from src.models.semiconcave import SemiconcaveModel


def _theta(m):
    """The model's trainable parameters as a flat vector (the SSN working var)."""
    return parameters_to_vector([p for p in m.parameters() if p.requires_grad])


def _set_structural(m, C=None, a=None, b0=None):
    """Set the (now nn.Parameter) structural fields in place."""
    with torch.no_grad():
        if C is not None:
            m.C.fill_(C)
        if a is not None:
            m.affine_w.copy_(torch.as_tensor(a, dtype=torch.float64))
        if b0 is not None:
            m.affine_b.fill_(b0)


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
    m = SemiconcaveModel(power=1.0, verbose=False)
    m.set_atoms(W, b, torch.tensor([1.0, 0.5, 0.0, 2.0], dtype=torch.float64))
    _set_structural(m, C=1.3, a=[0.2, -0.4], b0=0.7)
    Phi_v, Phi_g = m.jacobians(x)
    theta = _theta(m)
    Vp, dVp = m.predict_tensors(x)
    assert torch.allclose(Phi_v @ theta, Vp.reshape(-1), atol=1e-10)
    assert torch.allclose(Phi_g @ theta, dVp.reshape(-1), atol=1e-10)


def test_jacobians_match_finite_differences_without_functional_jacobian(monkeypatch):
    d, N, n = 2, 10, 3
    x = torch.randn(N, d, dtype=torch.float64)
    W, b = _atoms(n, d, seed=11)
    m = SemiconcaveModel(power=1.0, activation=torch.tanh, verbose=False)
    m.set_atoms(W, b, torch.rand(n, dtype=torch.float64))
    _set_structural(m, C=1.2, a=[0.15, -0.35], b0=0.4)

    def fail(*args, **kwargs):
        raise AssertionError("functional.jacobian materializes the large Phi_g tape")

    def value_and_grad(theta):
        x_req = x.detach().clone().requires_grad_(True)
        V = m._value(theta, x_req)
        dV = torch.autograd.grad(V.sum(), x_req, create_graph=False)[0]
        return V.reshape(-1).detach(), dV.reshape(-1).detach()

    monkeypatch.setattr(torch.autograd.functional, "jacobian", fail)
    Phi_v, Phi_g = m.jacobians(x)

    theta = _theta(m)
    eps = 1e-6
    fd_v = torch.zeros_like(Phi_v)
    fd_g = torch.zeros_like(Phi_g)
    for j in range(theta.numel()):
        delta = torch.zeros_like(theta)
        delta[j] = eps
        V_plus, dV_plus = value_and_grad(theta + delta)
        V_minus, dV_minus = value_and_grad(theta - delta)
        fd_v[:, j] = (V_plus - V_minus) / (2 * eps)
        fd_g[:, j] = (dV_plus - dV_minus) / (2 * eps)

    assert torch.allclose(Phi_v, fd_v, atol=1e-6, rtol=1e-6)
    assert torch.allclose(Phi_g, fd_g, atol=1e-6, rtol=1e-6)


def test_augmented_hessian_matches_autograd():
    d, N, n = 2, 40, 4
    x = torch.randn(N, d, dtype=torch.float64)
    W, b = _atoms(n, d, seed=3)
    m = SemiconcaveModel(power=1.0, verbose=False)
    m.set_atoms(W, b, torch.zeros(n))
    Phi_v, Phi_g = m.jacobians(x)
    Vt = torch.randn(N, dtype=torch.float64)
    dVt = torch.randn(N * d, dtype=torch.float64)
    Nx = N * d
    H = (1.0 / Nx) * (Phi_v.T @ Phi_v) + (1.0 / Nx) * (Phi_g.T @ Phi_g)

    def dloss(th):
        rv = Phi_v @ th - Vt
        rg = Phi_g @ th - dVt
        return (1.0 / (2 * Nx)) * (rv @ rv) + (1.0 / (2 * Nx)) * (rg @ rg)

    Hau = torch.autograd.functional.hessian(dloss, _theta(m))
    assert torch.allclose(H, Hau, atol=1e-10)


def test_train_ssn_recovers_synthetic_semiconcave_target():
    d, N, n = 2, 80, 4
    x = torch.randn(N, d, dtype=torch.float64)
    W, b = _atoms(n, d, seed=7)
    truth = SemiconcaveModel(power=1.0, verbose=False)
    truth.set_atoms(W, b, torch.tensor([1.5, 0.0, 0.8, 0.0], dtype=torch.float64))
    _set_structural(truth, C=2.0, a=[0.3, 0.1], b0=-0.5)
    V, dV = truth.predict_tensors(x)

    fit = SemiconcaveModel(power=1.0, verbose=False)
    fit.set_atoms(W, b, torch.full((n,), 0.1, dtype=torch.float64))
    _set_structural(fit, C=0.5, a=[0.0, 0.0], b0=0.0)
    ssn_solve(fit, (x, V, dV), Objective(alpha=1e-4, gamma=1.0, th=0.5), SolverConfig(), iterations=25)

    _, _, h1 = relative_errors(*fit.predict_tensors(x), V, dV)
    assert h1 < 1e-2
    assert abs(float(fit.C.detach()) - 2.0) < 0.05
    assert bool((fit.c >= -1e-9).all())
