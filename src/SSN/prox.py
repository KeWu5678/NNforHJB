"""Proximal operators for the SSN splitting: prox of mu * |.|^q and its Jacobian.

Closed-form solvers for q in {1, 1/2, 2/3} and a Newton fallback for other q<1.
``_phi_prox`` is the scalar proximal of sigma * phi(t^q) used by the PDAP
insertion step.
"""

import math

import torch

from .penalty import _dphi, _ddphi


def _compute_prox_q_half(v, mu):
    """Closed-form proximal for mu * |.|^{1/2}.

    FOC: t + (mu/2)*t^{-1/2} = |v|.  Substitute s = sqrt(t):
      s^3 - |v|*s + mu/2 = 0  (depressed cubic)
    Solved via trigonometric method (three real roots when |v| > v_thresh).
    """
    abs_v = torch.abs(v)
    q = 0.5
    t_star = (mu * q * (1.0 - q)) ** (1.0 / (2.0 - q))  # = (mu/4)^{2/3}
    v_thresh = t_star + mu * q * t_star ** (q - 1)
    active = abs_v > v_thresh

    av = abs_v.clamp(min=1e-30)
    # Depressed cubic s^3 - A*s + B = 0 with A = |v|, B = mu/2
    # Trigonometric solution: s = 2*sqrt(A/3)*cos(theta/3)
    # where theta = arccos(-3*B*sqrt(3) / (2*A^{3/2}))
    cos_arg = (-3.0 * mu * math.sqrt(3.0) / (4.0 * av ** 1.5)).clamp(-1.0, 1.0)
    theta = torch.acos(cos_arg) / 3.0
    s = 2.0 * torch.sqrt(av / 3.0) * torch.cos(theta)
    t = s ** 2

    t = torch.where(active, t, torch.zeros_like(t))
    return torch.sign(v) * t


def _compute_prox_q_twothirds(v, mu):
    """Closed-form proximal for mu * |.|^{2/3}.

    FOC: t + (2mu/3)*t^{-1/3} = |v|.  Substitute s = t^{1/3}:
      s^4 - |v|*s + 2mu/3 = 0  (depressed quartic)
    Solved via Ferrari's method:
      1. Resolvent cubic: y^3 - (2mu/3)*y - |v|^2/8 = 0
      2. Factor quartic into two quadratics using y
      3. Take largest positive root from the quadratic with real roots
    """
    abs_v = torch.abs(v)
    q = 2.0 / 3.0
    t_star = (mu * q * (1.0 - q)) ** (1.0 / (2.0 - q))  # = (2mu/9)^{3/4}
    v_thresh = t_star + mu * q * t_star ** (q - 1)
    active = abs_v > v_thresh

    av = abs_v.clamp(min=1e-30)

    # Resolvent cubic: y^3 + p_rc*y + q_rc = 0
    p_rc = -2.0 * mu / 3.0
    q_rc = -(av ** 2) / 8.0
    # Cardano discriminant: delta_c = (q_rc/2)^2 + (p_rc/3)^3
    delta_c = (q_rc / 2.0) ** 2 + (p_rc / 3.0) ** 3
    use_cardano = delta_c >= 0

    # Branch 1: Cardano (delta_c >= 0, one real root — typical case)
    sqrt_delta = torch.sqrt(delta_c.clamp(min=0))
    u_plus = -q_rc / 2.0 + sqrt_delta
    u_minus = -q_rc / 2.0 - sqrt_delta
    y_cardano = (torch.sign(u_plus) * torch.abs(u_plus).clamp(min=1e-300) ** (1.0 / 3.0)
               + torch.sign(u_minus) * torch.abs(u_minus).clamp(min=1e-300) ** (1.0 / 3.0))

    # Branch 2: Trigonometric (delta_c < 0, three real roots — rare)
    sqrt_neg_p3 = torch.sqrt((-p_rc / 3.0) * torch.ones_like(av))  # scalar p_rc
    cos_arg_rc_denom = (2.0 * (-p_rc) * sqrt_neg_p3).clamp(min=1e-30)
    cos_arg_rc = (3.0 * q_rc / cos_arg_rc_denom).clamp(-1.0, 1.0)
    theta_rc = torch.acos(cos_arg_rc) / 3.0
    y_trig = 2.0 * sqrt_neg_p3 * torch.cos(theta_rc)

    y = torch.where(use_cardano, y_cardano, y_trig)
    y = y.clamp(min=1e-30)

    # Factor quartic: x^2 - r*x + (y - |v|/(2r)) = 0
    r = torch.sqrt(2.0 * y)
    disc = (-2.0 * y + 2.0 * av / r.clamp(min=1e-30)).clamp(min=0)
    x = (r + torch.sqrt(disc)) / 2.0
    t = x ** 3  # s = t^{1/3}, so t = s^3

    t = torch.where(active, t, torch.zeros_like(t))
    return torch.sign(v) * t


def _compute_prox(v, mu, q=1.0):
    """Proximal operator for mu * |·|^q (the simple part of the SSN splitting).

    Solves: argmin_t { mu * t^q + (1/2) * (t - |v|)^2 } for t >= 0,
    then returns sign(v) * t_opt.

    For q=1: soft thresholding, prox(v) = sign(v) * max(|v| - mu, 0).
    For q=1/2: closed-form via depressed cubic (trigonometric method).
    For q=2/3: closed-form via depressed quartic (Ferrari's method).
    For other q<1: Newton's method on the optimality condition t + mu*q*t^{q-1} = |v|.

    Args:
        v: input tensor
        mu: proximal parameter (typically alpha / c in SSN)
        q: power exponent, q = 2/(p+1) where p is the activation power
    Returns:
        vprox: proximal operator result
    """
    if q == 1.0:
        normsv = torch.abs(v)
        eps = torch.finfo(v.dtype).eps
        normsv_safe = torch.clamp(normsv, min=(mu + eps) * eps)
        shrinkage_factor = torch.clamp(1 - mu / normsv_safe, min=0)
        return shrinkage_factor * v

    if abs(q - 0.5) < 1e-12:
        return _compute_prox_q_half(v, mu)

    if abs(q - 2.0 / 3.0) < 1e-12:
        return _compute_prox_q_twothirds(v, mu)

    # General q != 1: Newton's method fallback
    abs_v = torch.abs(v)
    t_star = (mu * q * (1.0 - q)) ** (1.0 / (2.0 - q))
    v_thresh = t_star + mu * q * t_star ** (q - 1)
    active = abs_v > v_thresh

    t = abs_v.clone()
    for _ in range(20):
        t_safe = t.clamp(min=1e-30)
        h = t_safe + mu * q * t_safe ** (q - 1) - abs_v
        hp = 1.0 + mu * q * (q - 1) * t_safe ** (q - 2)
        t = torch.where(active, t - h / hp, torch.zeros_like(t))
        t = t.clamp(min=0.0)

    t = torch.where(active, t, torch.zeros_like(t))
    return torch.sign(v) * t


def _compute_dprox(v, mu, q=1.0, prox_result=None):
    """Jacobian of the proximal operator prox_{mu*|·|^q}(v).

    Ported from MATLAB computeDProx.m. With N=1 (scalar outer weights),
    each neuron's Jacobian block reduces to a scalar diagonal entry.

    For q=1 (soft thresholding prox(v) = sign(v)*max(|v|-mu, 0)):
        d prox/dv = 1 for active (|v| > mu), 0 for inactive.
        (Computed via MATLAB's general N-dim formula specialized to N=1.)

    For q<1 (prox from Newton solve of t + mu*q*t^{q-1} = |v|):
        d prox/dv = 1 / (1 + mu*q*(q-1)*|prox|^{q-2}) for active, 0 for inactive.
        Requires prox_result to avoid recomputing the Newton loop.

    Used in SSN's _DG to form the generalized Jacobian:
        DG = c*(I - DPc) + alpha*diag(correction_dd)*DPc + H_data*DPc

    Args:
        v: proximal preimage tensor (the SSN variable q_var)
        mu: proximal parameter (typically alpha / c)
        q: power exponent, q = 2/(p+1)
        prox_result: precomputed prox(v) — required for q != 1

    Returns:
        DP: Jacobian matrix (diagonal), shape (n, n)
    """
    assert torch.is_floating_point(v), "Input must be real-valued"

    if q == 1.0:
        normsv = torch.abs(v)
        eps = torch.finfo(v.dtype).eps
        normsv_safe = torch.clamp(normsv, min=(mu + eps) * eps)
        # MATLAB computeDProx.m with N=1:
        #   max(0, 1 - mu/|v|) + (|v|>=mu) * mu/|v|^3 * v^2
        # For N=1 these sum to 1 (active) or 0 (inactive).
        # FOLLOW-UP (numerical cleanup): for q=1 this diagonal is analytically the
        # 0/1 active-set indicator, but reconstructing it via a large 1/|v|^3
        # intermediate makes it sensitive to whether ``mu`` is a Python float or a
        # broadcast tensor (~5e-10 wobble that only perturbs the Newton *direction*,
        # not the fixed point).  Replacing this with ``(normsv > mu).to(v.dtype)``
        # would be more robust, but changes the original behaviour by ~5e-10, so it
        # belongs in its own commit with the golden fixture re-captured.
        diagonal_term = torch.clamp(1 - mu / normsv_safe, min=0)
        mask = normsv >= mu
        outer_product_term = mask.float() * mu / (normsv_safe ** 3) * (v ** 2)
        return torch.diag(diagonal_term + outer_product_term)

    # General q != 1: implicit differentiation of t + mu*q*t^{q-1} = |v|
    abs_v = torch.abs(v)
    t_star = (mu * q * (1.0 - q)) ** (1.0 / (2.0 - q))
    v_thresh = t_star + mu * q * t_star ** (q - 1)
    active = abs_v > v_thresh

    prox_abs = torch.abs(prox_result)
    prox_safe = prox_abs.clamp(min=1e-30)

    denom = 1.0 + mu * q * (q - 1) * prox_safe ** (q - 2)
    diag = torch.where(active, 1.0 / denom, torch.zeros_like(v))

    return torch.diag(diag.reshape(-1))


def _phi_prox(sigma: float, g: float, th: float, gamma: float, q: float = 1.0) -> float:
    """
    Proximal operator for sigma * phi(t^q), solving:
        argmin_{t >= 0} { sigma * phi(t^q) + (1/2) * (t - g)^2 }

    For q=1, matches MATLAB setup_problem_NN_2d.m lines 176-177 (closed-form).
    For q!=1, uses Newton's method on the optimality condition:
        F(tau) = tau - g + sigma * q * tau^{q-1} * dphi(tau^q) = 0

    Args:
        sigma: proximal parameter (typically alpha / what)
        g: proximal center point (typically -phat / what)
        th: interpolation parameter between L1 (th=0) and non-convex (th=1)
        gamma: nonconvex penalty parameter
        q: power exponent, q = 2/(p+1) where p is the activation power

    Returns:
        The proximal point (float >= 0).
    """
    # q=1 path: original closed-form
    if q == 1.0:
        if gamma == 0 or th >= 1.0:
            return max(g - sigma, 0.0)
        gam = gamma / (1.0 - th)
        a = g - sigma * th - 1.0 / gam
        disc = a * a + 4.0 * (g - sigma) / gam
        if disc < 0:
            return 0.0
        return max(0.5 * (a + math.sqrt(disc)), 0.0)

    # General q != 1
    if g <= 0:
        return 0.0

    # For gamma=0 or th=1: phi(t) = t, so phi(t^q) = t^q.
    # Reduces to the simple proximal of sigma * |.|^q.
    if gamma == 0 or th >= 1.0:
        v_tensor = torch.tensor([g], dtype=torch.float64)
        result = _compute_prox(v_tensor, sigma, q=q)
        return max(float(result[0].item()), 0.0)

    # Newton's method for general phi(t^q) with gamma > 0
    tau = g
    for _ in range(30):
        if tau <= 0:
            return 0.0
        tq = tau ** q
        dp = _dphi(torch.tensor(tq), th, gamma).item()
        ddp = _ddphi(torch.tensor(tq), th, gamma).item()
        F_val = tau - g + sigma * q * tau ** (q - 1) * dp
        F_deriv = 1.0 + sigma * (
            q * (q - 1) * tau ** (q - 2) * dp
            + q ** 2 * tau ** (2 * q - 2) * ddp
        )
        if abs(F_deriv) < 1e-30:
            break
        tau_new = tau - F_val / F_deriv
        tau_new = max(tau_new, 0.0)
        if abs(tau_new - tau) < 1e-14 * max(abs(tau), 1.0):
            tau = tau_new
            break
        tau = tau_new

    return max(tau, 0.0)
