"""Canonical activation registry + resolver.

The ``ACTIVATIONS`` table maps a name to a ``(callable, use_sphere)`` tuple. The
config stores the activation as a **string** (OmegaConf holds only primitives);
:func:`get_activation` resolves it to the callable and :func:`get_use_sphere`
resolves it to the geometry flag at build time.

``use_sphere`` records whether candidate directions are sampled on S^d — valid
only for positively-homogeneous activations (relu, abs, relu2, ...). It lives
*with* the activation here so each entry self-documents its geometry; there is no
separate ``model.use_sphere`` config field.

This module is the single home of the registry. Hydra model configs resolve
activation names here instead of carrying local dictionaries.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------- #
# Helper builders
# ---------------------------------------------------------------------------- #
def matern52(z: torch.Tensor) -> torch.Tensor:
    """Matern 5/2 activation function.

    k(z) = (1 + sqrt(5)|z| + 5/3 * z^2) * exp(-sqrt(5)|z|)

    This activation is smooth and non-negative, but it is not positively
    homogeneous, so its registry entry carries ``use_sphere=False``.
    """
    r = torch.abs(z)
    sqrt5 = 2.23606797749979
    return (1.0 + sqrt5 * r + (5.0 / 3.0) * r.pow(2)) * torch.exp(-sqrt5 * r)


def swish_beta(beta: float):
    return lambda x: x * torch.sigmoid(beta * x)


def softplus_beta(beta: float):
    return lambda x: F.softplus(x, beta=beta)


def mish_beta(beta: float):
    return lambda x: x * torch.tanh(F.softplus(x, beta=beta))


def gelu_beta(beta: float):
    return lambda x: 0.5 * x * (1.0 + torch.erf(beta * x / 1.41421356237))


def logcosh(x: torch.Tensor) -> torch.Tensor:
    return x + F.softplus(-2.0 * x) - 0.6931471805599453


def smooth_relu(beta: float):
    """0.5 * (x + sqrt(x^2 + 1/beta^2))."""
    inv_b2 = 1.0 / (beta * beta)
    return lambda x: 0.5 * (x + torch.sqrt(x.pow(2) + inv_b2))


def rcip(p: float):
    return lambda x: x / (1.0 + torch.abs(x).pow(p))


def tanh_swish(beta: float):
    return lambda x: x * torch.tanh(beta * x)


def shifted_softplus(beta: float):
    """softplus(beta*x)/beta - log(2)/beta — passes through origin."""
    log2 = 0.6931471805599453
    return lambda x: F.softplus(x, beta=beta) - log2 / beta


def abs_act(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x)


def softrelu_arctan(beta: float):
    """Smooth ReLU via arctan: 0.5*(x + (2/pi)*x*arctan(beta*x))."""
    inv_pi = 0.3183098861837907
    return lambda x: 0.5 * x + inv_pi * x * torch.atan(beta * x)


def softplus_squared(beta: float):
    """softplus(beta*x)^2/beta^2 — antiderivative-like."""
    return lambda x: F.softplus(x, beta=beta).pow(2)


def cubic(x: torch.Tensor) -> torch.Tensor:
    return x.pow(3)


def log1p_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.relu(x))


def quad_residual(alpha: float):
    return lambda x: x + alpha * x.pow(2)


def silu_quad(alpha: float):
    return lambda x: F.silu(x) + alpha * x.pow(2)


def relu_quad(alpha: float):
    return lambda x: torch.relu(x) + alpha * x.pow(2)


def softplus_quad(beta: float, alpha: float):
    return lambda x: F.softplus(x, beta=beta) + alpha * x.pow(2)


def x_absx(x: torch.Tensor) -> torch.Tensor:
    """x|x| — antisymmetric quadratic."""
    return x * torch.abs(x)


def quartic(x: torch.Tensor) -> torch.Tensor:
    return x.pow(4)


def gauss_centered(beta: float):
    """1 - exp(-beta*x^2) — RBF inverted, monotone in |x|."""
    return lambda x: 1.0 - torch.exp(-beta * x.pow(2))


def silu_b(beta: float):
    return lambda x: x * torch.sigmoid(beta * x)


def gelu_lo_softplus(beta: float):
    """gelu_b * exp(softplus(-x)) - smooth bump shape."""
    return lambda x: 0.5 * x * (1.0 + torch.erf(beta * x / 1.41421356237))


def smooth_max0(beta: float):
    """log(1 + exp(beta*x))/beta - approaches max(0,x) as beta->inf."""
    return lambda x: F.softplus(x, beta=beta)


def asym_mix(a: float):
    """relu(x) + a*relu(-x)*-1 + bias smoothness — leaky+quad mix."""
    return lambda x: torch.relu(x) - a * torch.relu(-x).pow(2)


def gaussian(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-x.pow(2))


def snake(x: torch.Tensor) -> torch.Tensor:
    return x + torch.sin(x).pow(2)


def bent_identity(x: torch.Tensor) -> torch.Tensor:
    return (torch.sqrt(x.pow(2) + 1.0) - 1.0) * 0.5 + x


def sigmoid_act(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def relu2(x: torch.Tensor) -> torch.Tensor:
    """Squared ReLU: positively homogeneous of degree 2."""
    return torch.relu(x).pow(2)


def leaky_relu2(alpha: float):
    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, x, alpha * x).pow(2)
    return fn


# ---------------------------------------------------------------------------- #
# Registry: name -> (callable, use_sphere)
#
# ``use_sphere`` is True only for positively-homogeneous (or sphere-sampled)
# activations; smooth/non-homogeneous activations carry False.
# ---------------------------------------------------------------------------- #
ACTIVATIONS: dict[str, tuple[Callable, bool]] = {
    "relu": (torch.relu, True),
    "tanh": (torch.tanh, False),
    "gelu": (F.gelu, False),
    "silu": (F.silu, False),
    "sin": (torch.sin, False),
    "softplus": (F.softplus, False),
    "matern52": (matern52, False),
    "leaky_relu": (F.leaky_relu, True),
    "elu": (F.elu, False),
    "selu": (F.selu, False),
    "mish": (F.mish, False),
    "hardswish": (F.hardswish, False),
    "celu": (F.celu, False),
    "softsign": (F.softsign, False),
    "swish_b2": (swish_beta(2.0), False),
    "swish_b0_5": (swish_beta(0.5), False),
    "swish_b0_25": (swish_beta(0.25), False),
    "swish_b0_1": (swish_beta(0.1), False),
    "softplus_b2": (softplus_beta(2.0), False),
    "softplus_b0_5": (softplus_beta(0.5), False),
    "softplus_b0_25": (softplus_beta(0.25), False),
    "softplus_b0_1": (softplus_beta(0.1), False),
    "softplus_b0_15": (softplus_beta(0.15), False),
    "softplus_b0_2": (softplus_beta(0.2), False),
    "softplus_b0_3": (softplus_beta(0.3), False),
    "softplus_b0_35": (softplus_beta(0.35), False),
    "softplus_b0_22": (softplus_beta(0.22), False),
    "softplus_b0_28": (softplus_beta(0.28), False),
    "softplus_b0_24": (softplus_beta(0.24), False),
    "softplus_b0_26": (softplus_beta(0.26), False),
    "softplus_b0_27": (softplus_beta(0.27), False),
    "softplus_b0_23": (softplus_beta(0.23), False),
    "mish_b0_5": (mish_beta(0.5), False),
    "mish_b0_25": (mish_beta(0.25), False),
    "gelu_b0_5": (gelu_beta(0.5), False),
    "gelu_b0_25": (gelu_beta(0.25), False),
    "gelu_b0_15": (gelu_beta(0.15), False),
    "gelu_b0_2": (gelu_beta(0.2), False),
    "gelu_b0_3": (gelu_beta(0.3), False),
    "gelu_b0_35": (gelu_beta(0.35), False),
    "gelu_b0_4": (gelu_beta(0.4), False),
    "swish_b0_3": (swish_beta(0.3), False),
    "swish_b0_2": (swish_beta(0.2), False),
    "swish_b0_15": (swish_beta(0.15), False),
    "logcosh": (logcosh, False),
    "smooth_relu_4": (smooth_relu(4.0), False),
    "smooth_relu_2": (smooth_relu(2.0), False),
    "smooth_relu_1": (smooth_relu(1.0), False),
    "rcip_2": (rcip(2.0), False),
    "rcip_3": (rcip(3.0), False),
    "tanh_swish_b0_25": (tanh_swish(0.25), False),
    "tanh_swish_b0_5": (tanh_swish(0.5), False),
    "shsp_b0_25": (shifted_softplus(0.25), False),
    "shsp_b0_5": (shifted_softplus(0.5), False),
    "abs_act": (abs_act, True),
    "sra_b0_5": (softrelu_arctan(0.5), False),
    "sra_b0_25": (softrelu_arctan(0.25), False),
    "sp2_b0_25": (softplus_squared(0.25), False),
    "sp2_b0_5": (softplus_squared(0.5), False),
    "sp2_b0_1": (softplus_squared(0.1), False),
    "sp2_b0_15": (softplus_squared(0.15), False),
    "sp2_b0_2": (softplus_squared(0.2), False),
    "sp2_b0_3": (softplus_squared(0.3), False),
    "sp2_b0_4": (softplus_squared(0.4), False),
    "sp2_b0_05": (softplus_squared(0.05), False),
    "sp2_b1": (softplus_squared(1.0), False),
    "sp2_b2": (softplus_squared(2.0), False),
    "cubic": (cubic, True),
    "log1p_relu": (log1p_relu, False),
    "qr_0_5": (quad_residual(0.5), False),
    "qr_0_25": (quad_residual(0.25), False),
    "qr_0_1": (quad_residual(0.1), False),
    "qr_1": (quad_residual(1.0), False),
    "qr_0_05": (quad_residual(0.05), False),
    "siluquad_0_25": (silu_quad(0.25), False),
    "reluquad_0_25": (relu_quad(0.25), False),
    "spquad_0_25_0_25": (softplus_quad(0.25,0.25), False),
    "x_absx": (x_absx, True),
    "quartic": (quartic, True),
    "gausscent_1": (gauss_centered(1.0), False),
    "gausscent_0_5": (gauss_centered(0.5), False),
    "gelu_b0_1": (gelu_beta(0.1), False),
    "gelu_b0_05": (gelu_beta(0.05), False),
    "swish_b1": (swish_beta(1.0), False),
    "swish_b1_5": (swish_beta(1.5), False),
    "softplus_b1": (softplus_beta(1.0), False),
    "atan": (torch.atan, False),
    "sqrt_signed": (lambda x: torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-8), False),
    "asinh": (torch.asinh, False),
    "tanh_b0_25": (lambda x: torch.tanh(0.25*x), False),
    "tanh_b0_5": (lambda x: torch.tanh(0.5*x), False),
    "geluquad_b0_25_0_05": (lambda x: 0.5*x*(1.0+torch.erf(0.25*x/1.41421356237))+0.05*x.pow(2), False),
    "geluquad_b0_25_0_1": (lambda x: 0.5*x*(1.0+torch.erf(0.25*x/1.41421356237))+0.1*x.pow(2), False),
    "geluquad_b0_25_0_25": (lambda x: 0.5*x*(1.0+torch.erf(0.25*x/1.41421356237))+0.25*x.pow(2), False),
    "softplus_b0_25_minus_log2_d_0_25": (lambda x: F.softplus(x, beta=0.25)-2.7725887222397812, False),
    "elish": (lambda x: torch.where(x>=0, F.silu(x), (torch.exp(x)-1)*torch.sigmoid(x)), False),
    "lisht": (lambda x: x*torch.tanh(x), False),
    "lisht_b0_5": (lambda x: x*torch.tanh(0.5*x), False),
    "lisht_b0_25": (lambda x: x*torch.tanh(0.25*x), False),
    "smoothy_relu": (lambda x: torch.where(x>=0.5, x-0.25, torch.where(x<=-0.5, torch.zeros_like(x), 0.5*(x+0.5).pow(2))), False),
    "smoothy_relu_w0_25": (lambda x: torch.where(x>=0.25, x-0.125, torch.where(x<=-0.25, torch.zeros_like(x), 2.0*(x+0.25).pow(2))), False),
    "smoothy_relu_w1": (lambda x: torch.where(x>=1.0,  x-0.5,   torch.where(x<=-1.0, torch.zeros_like(x), 0.25*(x+1.0).pow(2))), False),
    "smoothy_relu_w2": (lambda x: torch.where(x>=2.0,  x-1.0,   torch.where(x<=-2.0, torch.zeros_like(x), 0.125*(x+2.0).pow(2))), False),
    "smoothy_relu_w0_1": (lambda x: torch.where(x>=0.1,  x-0.05,  torch.where(x<=-0.1, torch.zeros_like(x), 5.0*(x+0.1).pow(2))), False),
    "smoothy_relu_w0_05": (lambda x: torch.where(x>=0.05, x-0.025, torch.where(x<=-0.05,torch.zeros_like(x),10.0*(x+0.05).pow(2))), False),
    "smoothy_relu_w0_125": (lambda x: torch.where(x>=0.125,x-0.0625,torch.where(x<=-0.125,torch.zeros_like(x),4.0*(x+0.125).pow(2))), False),
    "smoothy_relu_w0_3": (lambda x: torch.where(x>=0.3, x-0.15, torch.where(x<=-0.3, torch.zeros_like(x), (5.0/3.0)*(x+0.3).pow(2))), False),
    "smoothy_relu_w0_4": (lambda x: torch.where(x>=0.4, x-0.2,  torch.where(x<=-0.4, torch.zeros_like(x), 1.25*(x+0.4).pow(2))), False),
    "smoothy_relu_w0_2": (lambda x: torch.where(x>=0.2, x-0.1,  torch.where(x<=-0.2, torch.zeros_like(x), 2.5*(x+0.2).pow(2))), False),
    "gelu_b2": (gelu_beta(2.0), False),
    "gelu_b4": (gelu_beta(4.0), False),
    "gelu_b1": (gelu_beta(1.0), False),
    "softplus_b3": (softplus_beta(3.0), False),
    "softplus_b5": (softplus_beta(5.0), False),
    "swish_b3": (swish_beta(3.0), False),
    "swish_b5": (swish_beta(5.0), False),
    "asym_sp_tanh": (lambda x: F.softplus(x, beta=0.25) + 0.1*torch.tanh(x), False),
    "asym_sp_tanh_s": (lambda x: F.softplus(x, beta=0.25) - 0.5*torch.tanh(x), False),
    "sp025_quad": (lambda x: F.softplus(x, beta=0.25) + 0.05*x.pow(2), False),
    "smooth_leaky": (lambda x: torch.where(x>=0.25, x-0.125,
                          torch.where(x<=-0.25, 0.01*x + 0.0025,
                              2.0*(x+0.25).pow(2) + 0.01*x - 0.0025*0)), False),
    "smoothy_relu_sphere": (lambda x: torch.where(x>=0.25, x-0.125, torch.where(x<=-0.25, torch.zeros_like(x), 2.0*(x+0.25).pow(2))), True),
    "smoothy_relu_w0_25_b0_5": (lambda x: 0.5 * (torch.where(x>=0.25, x-0.125, torch.where(x<=-0.25, torch.zeros_like(x), 2.0*(x+0.25).pow(2)))), False),
    "smoothy_swish": (lambda x: torch.where(x>=0.25, x-0.125, torch.where(x<=-0.25, torch.zeros_like(x), 2.0*(x+0.25).pow(2))) * torch.sigmoid(0.25*x), False),
    "snake_b0_25": (lambda x: x + torch.sin(0.25*x).pow(2)/0.25, False),
    "snake_b0_5": (lambda x: x + torch.sin(0.5*x).pow(2)/0.5, False),
    "elu2": (lambda x: F.elu(x).pow(2), False),
    "selu_b0_25": (lambda x: F.selu(0.25*x), False),
    "leaky_relu2_a0_001_sphere": (leaky_relu2(0.001), True),
    "leaky_relu2_sphere": (leaky_relu2(0.01), True),
    "leaky_relu2_a0_015_sphere": (leaky_relu2(0.015), True),
    "leaky_relu2_a0_02_sphere": (leaky_relu2(0.02), True),
    "leaky_relu2_a0_025_sphere": (leaky_relu2(0.025), True),
    "leaky_relu2_a0_0375_sphere": (leaky_relu2(0.0375), True),
    "leaky_relu2_a0_05_sphere": (leaky_relu2(0.05), True),
    "leaky_relu2_a0_05": (leaky_relu2(0.05), True),
    "leaky_relu2_a0_0625_sphere": (leaky_relu2(0.0625), True),
    "leaky_relu2_a0_075_sphere": (leaky_relu2(0.075), True),
    "leaky_relu2_a0_1_sphere": (leaky_relu2(0.1), True),
    "gelu_squared": (lambda x: F.gelu(x).pow(2), False),
    "silu_squared": (lambda x: F.silu(x).pow(2), False),
    "softplus_squared_b1": (lambda x: F.softplus(x).pow(2), False),
    "elu2_b0_5": (lambda x: F.elu(0.5*x).pow(2), False),
    "elu2_b2": (lambda x: F.elu(2.0*x).pow(2), False),
    "mish_b0_15": (mish_beta(0.15), False),
    "mish_b0_1": (mish_beta(0.1), False),
    "gaussian": (gaussian, False),
    "snake": (snake, False),
    "bent_id": (bent_identity, False),
    "sigmoid": (sigmoid_act, False),
    "relu2": (relu2, True),
}


# ---------------------------------------------------------------------------- #
# Resolver
# ---------------------------------------------------------------------------- #
def _lookup(name: str) -> tuple[Callable, bool]:
    if name not in ACTIVATIONS:
        raise ValueError(
            f"unknown activation {name!r}; choices include "
            f"{sorted(ACTIVATIONS)[:8]}... ({len(ACTIVATIONS)} total)"
        )
    return ACTIVATIONS[name]


def get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Resolve a registry name to its activation callable."""
    return _lookup(name)[0]


def get_use_sphere(name: str) -> bool:
    """Resolve a registry name to its sphere-geometry flag.

    True only for positively-homogeneous (or sphere-sampled) activations, for
    which candidate directions may be sampled on S^d.
    """
    return _lookup(name)[1]


__all__ = ["ACTIVATIONS", "get_activation", "get_use_sphere", "matern52"]
