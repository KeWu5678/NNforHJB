"""Non-convex penalty phi and its derivatives / SSN correction terms.

phi interpolates between the L1 penalty (th=1 or gamma=0) and the full
non-convex penalty (th=0).  These are the regularisation primitives of the SSN
splitting; the optimizer and the model closures both build on them.
"""

import torch


def _phi(t, th, gamma):
    """
    Non-convex penalty function phi(t).
    th = 0: the full nonconvex penalty
    th = 1: the L1 penalty
    """
    # Degenerate case: gamma = 0 corresponds to L1.
    # This avoids the 0/0 form in log(1+gam*t)/gam when gam=gamma/(1-th)=0.
    if gamma == 0:
        return t
    if th == 1:
        return t
    else:
        # Safeguard against th very close to 1
        if abs(1 - th) < 1e-8:
            return t
        gam = gamma / (1 - th)  # = 2*gamma
        return th * t + (1 - th) * torch.log(1 + gam * t) / gam

def _dphi(t, th, gamma):
    """Derivative of penalty function."""
    # Degenerate case: gamma = 0 corresponds to L1 => d/dt |t| = 1 for t>=0 here.
    if gamma == 0:
        return torch.ones_like(t)
    if th == 1:
        return torch.ones_like(t)
    else:
        gam = gamma / (1 - th)
        return th + (1 - th) / (1 + gam * t)

def _ddphi(t, th, gamma):
    """Second derivative of penalty function."""
    # Degenerate case: gamma = 0 corresponds to L1 => second derivative is zero a.e.
    if gamma == 0:
        return torch.zeros_like(t)
    if th == 1:
        return torch.zeros_like(t)
    else:
        # Safeguard against th very close to 1
        if abs(1 - th) < 1e-8:
            return torch.zeros_like(t)
        gam = gamma / (1 - th)
        return -(1 - th) * gam / ((1 + gam * t) ** 2)

def _penalty_grad(abs_u, sign_u, alpha, th, gamma, q=1.0):
    """Gradient of alpha * phi(|u|^q) w.r.t. u.

    For q=1: alpha * dphi(|u|) * sign(u).
    For q!=1: alpha * q * |u|^{q-1} * sign(u) * dphi(|u|^q),
    with singularity guard at u=0.
    """
    if q == 1.0:
        return alpha * _dphi(abs_u, th, gamma) * sign_u
    active = abs_u > 0
    s = abs_u.clamp(min=1e-30)
    return torch.where(
        active,
        alpha * q * s ** (q - 1) * sign_u * _dphi(s ** q, th, gamma),
        torch.zeros_like(abs_u),
    )


def _nonconvex_correction(abs_u, sign_u, th, gamma, q=1.0):
    """Gradient of (phi(|u|^q) - |u|^q) w.r.t. u (the non-convex part).

    For q=1: sign(u) * (dphi(|u|) - 1).
    For q!=1: q * |u|^{q-1} * sign(u) * (dphi(|u|^q) - 1).
    """
    if q == 1.0:
        return sign_u * (_dphi(abs_u, th, gamma) - 1.0)
    active = abs_u > 0
    s = abs_u.clamp(min=1e-30)
    return torch.where(
        active,
        q * s ** (q - 1) * sign_u * (_dphi(s ** q, th, gamma) - 1.0),
        torch.zeros_like(abs_u),
    )


def _nonconvex_correction_dd(abs_u, th, gamma, q=1.0):
    """Second derivative of (phi(|u|^q) - |u|^q) w.r.t. u (diagonal entries).

    For q=1: ddphi(|u|).
    For q!=1: q(q-1)|u|^{q-2}(dphi(|u|^q)-1) + q^2|u|^{2q-2}*ddphi(|u|^q).
    """
    if q == 1.0:
        return _ddphi(abs_u, th, gamma)
    active = abs_u > 0
    s = abs_u.clamp(min=1e-30)
    uq = s ** q
    term1 = q * (q - 1) * s ** (q - 2) * (_dphi(uq, th, gamma) - 1.0)
    term2 = q ** 2 * s ** (2 * q - 2) * _ddphi(uq, th, gamma)
    return torch.where(active, term1 + term2, torch.zeros_like(abs_u))
