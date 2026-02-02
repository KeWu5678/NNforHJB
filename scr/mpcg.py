import torch
from typing import Optional


def _to_boundary(
    x: torch.Tensor, 
    d: torch.Tensor, 
    s: float
    ) -> tuple[torch.Tensor, float]:
    """
    Project x + tau d to the boundary |x| = s by solving for tau.
    In the algorithm, we should have
    |x| <= s, |x + tau d| <= s
    Returns the updated x and the step length tau used.
    """
    # Scalars used in quadratic for tau
    dd = torch.dot(d, d)
    xd = torch.dot(x, d)
    xx = torch.dot(x, x)

    # det = xd^2 + dd * (s^2 - xx)
    det = xd * xd + dd * (s * s - xx)
    # Numerical safety
    det = torch.clamp(det, min=0.0)
    tau = (s * s - xx) / (xd + torch.sqrt(det + 0.0))
    x_new = x + tau * d
    return x_new, float(tau)


def mpcg(
    H: torch.Tensor,
    b: torch.Tensor,
    rtol: float = 1e-3,
    maxit: int = 100,
    sigma: float = 0.0,
    DP: Optional[torch.Tensor] = None,
):
    """Projected Conjugate Gradient with optional trust-region.
    Args:
        H: (approximated) Hessian.
        b: ( - gradient f) Right-hand side vector.
        rtol: Relative tolerance.
        maxit: Maximum inner iterations.
        sigma: Trust-region radius (<= 0 disables TR handling).
        DP: Projection/preconditioner used only in inner products (can be None for identity).

    Returns:
        x:      Solution vector
        flag:   'convgd' | 'maxitr' | 'negdef' | 'radius'
        pred:   Predicted overall decrease in the model (scalar float)
        relres: Relative residual
        iters:  Number of iterations performed (int)
    """
    device = b.device
    dtype = b.dtype

    x = torch.zeros_like(b)

    # r(x) = b - H x, start with x = 0
    r = b.clone() 
    z = r.clone()
    DPz = z if DP is None else (DP @ z)
    resres = torch.dot(r, DPz)  # r' * DP * z

    # clone of the iterates and preconditioned iterates
    d = z.clone()
    DPd = DPz.clone()

    # Residuals
    res0 = torch.sqrt(torch.clamp(resres, min=0.0)) # projected residual
    fres0 = torch.sqrt(torch.dot(r, r)) # full residual
    epseps = -0.0 # current step decrease in the functional j(x) = (1/2) * DP(x)' * H(x) - DP(x)' * b
    pred = -0.0 # predicted overall decrease in the model

    iters = 0
    flag = 'convgd'

    # Main loop; stopping like in MATLAB
    # while sqrt(-epseps) >= sqrt(-pred)*rtol and sqrt(resres) >= res0*(1e-2*rtol)
    def _continue() -> bool:
        left = torch.sqrt(torch.tensor(max(0.0, -epseps), device=device, dtype=dtype))
        right = torch.sqrt(torch.tensor(max(0.0, -pred), device=device, dtype=dtype)) * rtol
        res_ok = torch.sqrt(torch.clamp(resres, min=0.0)) >= res0 * (1e-2 * rtol)
        return (left >= right) and bool(res_ok)

    while _continue():
        iters += 1
        if iters > maxit:
            flag = 'maxitr'
            iters = maxit
            break

        Hd = H @ d
        gamma = torch.dot(Hd, DPd)

        # Negative curvature
        if gamma <= 0:
            flag = 'negdef'
            if sigma > 0:
                x, tau = _to_boundary(x, d, sigma)
                pred = pred - tau * resres.item() + 0.5 * tau * tau * gamma.item()
                r = r - tau * Hd
                resres = torch.dot(r, r) if DP is None else torch.dot(r, DP @ r)
            relres = float(torch.sqrt(torch.clamp(resres, min=0.0)) / (res0 + 1e-30))
            return x, flag, float(pred), relres, iters

        alpha = resres / (gamma + 1e-30)
        xnew = x + alpha * d

        # Trust region redius reached 
        normx = torch.norm(xnew)
        if sigma > 0 and normx > sigma:
            flag = 'radius'
            x, tau = _to_boundary(x, d, sigma)
            pred = pred - tau * resres.item() + 0.5 * tau * tau * gamma.item()
            r = r - tau * Hd
            resres = torch.dot(r, r) if DP is None else torch.dot(r, DP @ r)
            relres = float(torch.sqrt(torch.clamp(resres, min=0.0)) / (res0 + 1e-30))
            return x, flag, float(pred), relres, iters
        else:
            x = xnew

        r = r - alpha * Hd
        epseps = -0.5 * (alpha * resres).item()
        pred = pred + epseps

        z = r.clone()
        DPz = z if DP is None else (DP @ z)
        resresold = resres
        resres = torch.dot(r, DPz)

        beta = - (alpha * torch.dot(Hd, DPz)) / (resresold + 1e-30)
        d = z + beta * d
        DPd = DPz + beta * DPd

    # determine theta to minimize ||H(x + theta z) - b||^2
    z = r.clone()
    Hz = H @ z
    numerator = torch.dot(r, Hz)
    denom = torch.dot(Hz, Hz) + 1e-30
    theta = numerator / denom

    xnew = x + theta * z
    normx = torch.norm(xnew)
    if sigma > 0 and normx > sigma:
        flag = 'radius'
        x, theta_val = _to_boundary(x, z, sigma)
        theta = torch.tensor(theta_val, device=device, dtype=dtype)
    else:
        x = xnew

    DPz = z if DP is None else (DP @ z)
    pred = pred - theta.item() * resres.item() + 0.5 * (theta * theta * torch.dot(Hz, DPz)).item()
    r = r - theta * Hz
    # MATLAB: z = r; relres = sqrt(r' * z) / fres0 = ||r|| / fres0
    z = r
    relres = float(torch.sqrt(torch.dot(r, z)) / (fres0 + 1e-30))

    return x, flag, float(pred), relres, iters


__all__ = ["mpcg"]


