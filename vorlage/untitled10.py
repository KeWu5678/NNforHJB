#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 22:50:43 2024

@author: chaoruiz
"""

import torch
import time

def PDAPmultisemidiscrete(p, y_ref, alpha, phi, alg_opts=None):
    """
    Solves the optimization problem:
        min (1/2) * ||Sred*u - ref||^2 + alpha * phi(||u||_1,2)
    
    Parameters:
        p - A dictionary or class containing auxiliary functions and data types.
        y_ref - Reference target.
        alpha - Regularization hyperparameter.
        phi - Penalizing function (with methods phi, inv, prox).
        alg_opts - Dictionary of optional arguments:
            - max_step: Maximum number of iterations.
            - plot_every: Frequency of plotting iterations.
            - sparsification: Boolean to sparsify nodes.
            - TOL: Tolerance for convergence.
            - optimize_x: Boolean to optimize input positions.
            - update_M0: Boolean to dynamically update M0.
            - u0: Initial guess for u.
    
    Returns:
        u_opt - Optimized solution.
        alg_out - Diagnostics and metadata of the optimization process.
    """

    if alg_opts is None:
        alg_opts = {}

    # Algorithm options with default values
    max_step = alg_opts.get("max_step", 1000)
    TOL = alg_opts.get("TOL", 1e-5)
    update_M0 = alg_opts.get("update_M0", True)
    sparsification = alg_opts.get("sparsification", False)
    optimize_x = alg_opts.get("optimize_x", False)
    plot_final = alg_opts.get("plot_final", True)
    plot_every = alg_opts.get("plot_every", 0)

    # Initial guess
    u0 = alg_opts.get("u0", p["u_zero"])
    uk = u0

    # Initial values
    Ku = p["K"](p, p["xhat"], uk)
    norms_u = compute_norm(uk["u"], p["N"])
    obj = p["obj"]
    j = obj["F"](Ku - y_ref) + alpha * torch.sum(phi.phi(norms_u))
    suppsize = torch.count_nonzero(norms_u)

    M0 = min(phi.inv(obj["F"](-y_ref) / alpha), 1e8)

    # Diagnostics
    alg_out = {
        "us": [uk],
        "js": [j],
        "supps": [suppsize],
        "tics": [0],
        "Psis": [0]
    }

    start_time = time.time()

    print(f"PDAP: {0:3d}, desc: ({0:.1e},{0:.1e}), supp: {suppsize}, j: ({j:.2e},{float('inf'):.1e}), M0: {M0:.2e}")

    iter = 1
    while True:
        # Gradient computation
        yk = obj["dF"](Ku - y_ref)
        xinit = uk["x"] if not optimize_x else p["u_zero"]["x"]
        xmax = p["find_max"](p, yk, xinit)
        grad = p["Ks"](p, xmax, p["xhat"], yk).reshape(p["N"], xmax.shape[1])

        # Dynamic update of M0
        if update_M0:
            M0 = min(phi.inv(j / alpha), 1e6)

        norms_grad = compute_norm(grad, p["N"])
        _, perm = torch.sort(norms_grad, descending=True)
        xmax = xmax[:, perm]
        grad = grad[:, perm]
        norms_grad = norms_grad[perm]

        # Insert multiple points
        buffer = 0.5 * (norms_grad.max() - alpha)
        cutoff = alpha + buffer
        locs = torch.where(norms_grad >= min(alpha + buffer, norms_grad.min()))[0][:15]

        xmax = xmax[:, locs]
        grad = grad[:, locs]
        norms_grad = norms_grad[locs]

        # Descent direction
        coeff = -torch.sqrt(torch.finfo(grad.dtype).eps) * grad / norms_grad
        coeff[:, 0] = -grad[:, 0] / norms_grad[0]
        newsupp = torch.cat((uk["x"], xmax), dim=1)
        vhat = {"x": newsupp, "u": torch.cat((torch.zeros_like(uk["u"]), coeff), dim=1)}
        uk_new = {"x": newsupp, "u": torch.cat((uk["u"], torch.zeros_like(coeff)), dim=1)}

        # Slope and curvature
        Kvhat = p["K"](p, p["xhat"], vhat)
        phat = (Kvhat.T @ yk).real
        what = (Kvhat.T @ obj["ddF"](Ku - y_ref) @ Kvhat).real

        phi_u = alpha * torch.sum(phi.phi(norms_u))
        phi_vhat = alpha * phi.phi(torch.sum(M0 * compute_norm(vhat["u"], p["N"])))
        upperb = phi_u + (Ku.T @ yk).real - min(phi_vhat + M0 * phat, 0)

        # Termination
        if (upperb <= TOL * j) or (iter > max_step):
            print(f"PDAP: {iter:3d}, desc: ({0:.1e},{0:.1e}), supp: {suppsize}, j: ({j:.2e},{upperb:.1e}), M0: {M0:.2e}")
            alg_out["Psis"].append(upperb)
            break

        # Coordinate descent step
        tau = phi.prox(alpha / what, -phat / what) if phat <= -alpha else 0
        uk = uk_new
        uk["u"] += tau * vhat["u"]

        # Update values
        Ku = p["K"](p, p["xhat"], uk)
        norms_u = compute_norm(uk["u"], p["N"])
        j = obj["F"](Ku - y_ref) + alpha * torch.sum(phi.phi(norms_u))
        suppsize = torch.count_nonzero(norms_u)

        # Diagnostics
        iter += 1
        alg_out["us"].append(uk)
        alg_out["js"].append(j.item())
        alg_out["supps"].append(suppsize)
        alg_out["tics"].append(time.time() - start_time)
        alg_out["Psis"].append(upperb)

        # Logging
        print(f"PDAP: {iter:3d}, desc: ({0:.1e},{0:.1e}), supp: {suppsize}, j: ({j:.2e},{upperb:.1e}), M0: {M0:.2e}")

    # Final plotting
    if plot_final:
        yk = obj["dF"](Ku - y_ref)
        p["plot_forward"](p, uk, y_ref)
        p["plot_adjoint"](p, uk, yk, alpha)

    return uk, alg_out
