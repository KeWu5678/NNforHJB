"""Canonical activation registry + resolver.

The ``ACTIVATIONS`` table maps a name to its activation callable. The config
stores the activation as a **string** (OmegaConf holds only primitives) and
:func:`get_activation` resolves it to the callable at build time.

Sphere geometry (whether candidate directions are sampled on S^d, valid only for
positively-homogeneous activations) is a separate ``model.use_sphere`` config
field, set by hand — it is no longer bundled with the activation.

This module is the single home of the registry. ``scripts/run_activation_experiment.py``
re-imports ``ACTIVATIONS`` from here (``src`` must not import from ``scripts``).
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
    homogeneous, so runs using it should set ``model.use_sphere=false``.
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
# Registry: name -> callable
# ---------------------------------------------------------------------------- #
ACTIVATIONS: dict[str, Callable] = {
    "relu": torch.relu,
    "tanh": torch.tanh,
    "gelu": F.gelu,
    "silu": F.silu,
    "sin": torch.sin,
    "softplus": F.softplus,
    "matern52": matern52,
    "leaky_relu": F.leaky_relu,
    "elu": F.elu,
    "selu": F.selu,
    "mish": F.mish,
    "hardswish": F.hardswish,
    "celu": F.celu,
    "softsign": F.softsign,
    "swish_b2": swish_beta(2.0),
    "swish_b0_5": swish_beta(0.5),
    "swish_b0_25": swish_beta(0.25),
    "swish_b0_1": swish_beta(0.1),
    "softplus_b2": softplus_beta(2.0),
    "softplus_b0_5": softplus_beta(0.5),
    "softplus_b0_25": softplus_beta(0.25),
    "softplus_b0_1": softplus_beta(0.1),
    "softplus_b0_15": softplus_beta(0.15),
    "softplus_b0_2": softplus_beta(0.2),
    "softplus_b0_3": softplus_beta(0.3),
    "softplus_b0_35": softplus_beta(0.35),
    "softplus_b0_22": softplus_beta(0.22),
    "softplus_b0_28": softplus_beta(0.28),
    "softplus_b0_24": softplus_beta(0.24),
    "softplus_b0_26": softplus_beta(0.26),
    "softplus_b0_27": softplus_beta(0.27),
    "softplus_b0_23": softplus_beta(0.23),
    "mish_b0_5": mish_beta(0.5),
    "mish_b0_25": mish_beta(0.25),
    "gelu_b0_5": gelu_beta(0.5),
    "gelu_b0_25": gelu_beta(0.25),
    "gelu_b0_15": gelu_beta(0.15),
    "gelu_b0_2": gelu_beta(0.2),
    "gelu_b0_3": gelu_beta(0.3),
    "gelu_b0_35": gelu_beta(0.35),
    "gelu_b0_4": gelu_beta(0.4),
    "swish_b0_3": swish_beta(0.3),
    "swish_b0_2": swish_beta(0.2),
    "swish_b0_15": swish_beta(0.15),
    "logcosh": logcosh,
    "smooth_relu_4": smooth_relu(4.0),
    "smooth_relu_2": smooth_relu(2.0),
    "smooth_relu_1": smooth_relu(1.0),
    "rcip_2": rcip(2.0),
    "rcip_3": rcip(3.0),
    "tanh_swish_b0_25": tanh_swish(0.25),
    "tanh_swish_b0_5": tanh_swish(0.5),
    "shsp_b0_25": shifted_softplus(0.25),
    "shsp_b0_5": shifted_softplus(0.5),
    "abs_act": abs_act,
    "sra_b0_5": softrelu_arctan(0.5),
    "sra_b0_25": softrelu_arctan(0.25),
    "sp2_b0_25": softplus_squared(0.25),
    "sp2_b0_5": softplus_squared(0.5),
    "sp2_b0_1": softplus_squared(0.1),
    "sp2_b0_15": softplus_squared(0.15),
    "sp2_b0_2": softplus_squared(0.2),
    "sp2_b0_3": softplus_squared(0.3),
    "sp2_b0_4": softplus_squared(0.4),
    "sp2_b0_05": softplus_squared(0.05),
    "sp2_b1": softplus_squared(1.0),
    "sp2_b2": softplus_squared(2.0),
    "cubic": cubic,
    "log1p_relu": log1p_relu,
    "qr_0_5": quad_residual(0.5),
    "qr_0_25": quad_residual(0.25),
    "qr_0_1": quad_residual(0.1),
    "qr_1": quad_residual(1.0),
    "qr_0_05": quad_residual(0.05),
    "siluquad_0_25": silu_quad(0.25),
    "reluquad_0_25": relu_quad(0.25),
    "spquad_0_25_0_25": softplus_quad(0.25,0.25),
    "x_absx": x_absx,
    "quartic": quartic,
    "gausscent_1": gauss_centered(1.0),
    "gausscent_0_5": gauss_centered(0.5),
    "gelu_b0_1": gelu_beta(0.1),
    "gelu_b0_05": gelu_beta(0.05),
    "swish_b1": swish_beta(1.0),
    "swish_b1_5": swish_beta(1.5),
    "softplus_b1": softplus_beta(1.0),
    "atan": torch.atan,
    "sqrt_signed": lambda x: torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-8),
    "asinh": torch.asinh,
    "tanh_b0_25": lambda x: torch.tanh(0.25*x),
    "tanh_b0_5": lambda x: torch.tanh(0.5*x),
    "geluquad_b0_25_0_05": lambda x: 0.5*x*(1.0+torch.erf(0.25*x/1.41421356237))+0.05*x.pow(2),
    "geluquad_b0_25_0_1": lambda x: 0.5*x*(1.0+torch.erf(0.25*x/1.41421356237))+0.1*x.pow(2),
    "geluquad_b0_25_0_25": lambda x: 0.5*x*(1.0+torch.erf(0.25*x/1.41421356237))+0.25*x.pow(2),
    "softplus_b0_25_minus_log2_d_0_25": lambda x: F.softplus(x, beta=0.25)-2.7725887222397812,
    "elish": lambda x: torch.where(x>=0, F.silu(x), (torch.exp(x)-1)*torch.sigmoid(x)),
    "lisht": lambda x: x*torch.tanh(x),
    "lisht_b0_5": lambda x: x*torch.tanh(0.5*x),
    "lisht_b0_25": lambda x: x*torch.tanh(0.25*x),
    "smoothy_relu": lambda x: torch.where(x>=0.5, x-0.25, torch.where(x<=-0.5, torch.zeros_like(x), 0.5*(x+0.5).pow(2))),
    "smoothy_relu_w0_25": lambda x: torch.where(x>=0.25, x-0.125, torch.where(x<=-0.25, torch.zeros_like(x), 2.0*(x+0.25).pow(2))),
    "smoothy_relu_w1": lambda x: torch.where(x>=1.0,  x-0.5,   torch.where(x<=-1.0, torch.zeros_like(x), 0.25*(x+1.0).pow(2))),
    "smoothy_relu_w2": lambda x: torch.where(x>=2.0,  x-1.0,   torch.where(x<=-2.0, torch.zeros_like(x), 0.125*(x+2.0).pow(2))),
    "smoothy_relu_w0_1": lambda x: torch.where(x>=0.1,  x-0.05,  torch.where(x<=-0.1, torch.zeros_like(x), 5.0*(x+0.1).pow(2))),
    "smoothy_relu_w0_05": lambda x: torch.where(x>=0.05, x-0.025, torch.where(x<=-0.05,torch.zeros_like(x),10.0*(x+0.05).pow(2))),
    "smoothy_relu_w0_125": lambda x: torch.where(x>=0.125,x-0.0625,torch.where(x<=-0.125,torch.zeros_like(x),4.0*(x+0.125).pow(2))),
    "smoothy_relu_w0_3": lambda x: torch.where(x>=0.3, x-0.15, torch.where(x<=-0.3, torch.zeros_like(x), (5.0/3.0)*(x+0.3).pow(2))),
    "smoothy_relu_w0_4": lambda x: torch.where(x>=0.4, x-0.2,  torch.where(x<=-0.4, torch.zeros_like(x), 1.25*(x+0.4).pow(2))),
    "smoothy_relu_w0_2": lambda x: torch.where(x>=0.2, x-0.1,  torch.where(x<=-0.2, torch.zeros_like(x), 2.5*(x+0.2).pow(2))),
    "gelu_b2": gelu_beta(2.0),
    "gelu_b4": gelu_beta(4.0),
    "gelu_b1": gelu_beta(1.0),
    "softplus_b3": softplus_beta(3.0),
    "softplus_b5": softplus_beta(5.0),
    "swish_b3": swish_beta(3.0),
    "swish_b5": swish_beta(5.0),
    "asym_sp_tanh": lambda x: F.softplus(x, beta=0.25) + 0.1*torch.tanh(x),
    "asym_sp_tanh_s": lambda x: F.softplus(x, beta=0.25) - 0.5*torch.tanh(x),
    "sp025_quad": lambda x: F.softplus(x, beta=0.25) + 0.05*x.pow(2),
    "smooth_leaky": lambda x: torch.where(x>=0.25, x-0.125,
                          torch.where(x<=-0.25, 0.01*x + 0.0025,
                              2.0*(x+0.25).pow(2) + 0.01*x - 0.0025*0)),
    "smoothy_relu_sphere": lambda x: torch.where(x>=0.25, x-0.125, torch.where(x<=-0.25, torch.zeros_like(x), 2.0*(x+0.25).pow(2))),
    "smoothy_relu_w0_25_b0_5": lambda x: 0.5 * (torch.where(x>=0.25, x-0.125, torch.where(x<=-0.25, torch.zeros_like(x), 2.0*(x+0.25).pow(2)))),
    "smoothy_swish": lambda x: torch.where(x>=0.25, x-0.125, torch.where(x<=-0.25, torch.zeros_like(x), 2.0*(x+0.25).pow(2))) * torch.sigmoid(0.25*x),
    "snake_b0_25": lambda x: x + torch.sin(0.25*x).pow(2)/0.25,
    "snake_b0_5": lambda x: x + torch.sin(0.5*x).pow(2)/0.5,
    "elu2": lambda x: F.elu(x).pow(2),
    "selu_b0_25": lambda x: F.selu(0.25*x),
    "leaky_relu2_a0_001_sphere": leaky_relu2(0.001),
    "leaky_relu2_sphere": leaky_relu2(0.01),
    "leaky_relu2_a0_015_sphere": leaky_relu2(0.015),
    "leaky_relu2_a0_02_sphere": leaky_relu2(0.02),
    "leaky_relu2_a0_025_sphere": leaky_relu2(0.025),
    "leaky_relu2_a0_0375_sphere": leaky_relu2(0.0375),
    "leaky_relu2_a0_05_sphere": leaky_relu2(0.05),
    "leaky_relu2_a0_05": leaky_relu2(0.05),
    "leaky_relu2_a0_0625_sphere": leaky_relu2(0.0625),
    "leaky_relu2_a0_075_sphere": leaky_relu2(0.075),
    "leaky_relu2_a0_1_sphere": leaky_relu2(0.1),
    "gelu_squared": lambda x: F.gelu(x).pow(2),
    "silu_squared": lambda x: F.silu(x).pow(2),
    "softplus_squared_b1": lambda x: F.softplus(x).pow(2),
    "elu2_b0_5": lambda x: F.elu(0.5*x).pow(2),
    "elu2_b2": lambda x: F.elu(2.0*x).pow(2),
    "mish_b0_15": mish_beta(0.15),
    "mish_b0_1": mish_beta(0.1),
    "gaussian": gaussian,
    "snake": snake,
    "bent_id": bent_identity,
    "sigmoid": sigmoid_act,
    "relu2": relu2,
}


# ---------------------------------------------------------------------------- #
# Resolver
# ---------------------------------------------------------------------------- #
def get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Resolve a registry name to its activation callable.

    The sphere geometry (whether candidate directions are sampled on S^d) is no
    longer bundled with the activation — set ``model.use_sphere`` in the config.
    """
    if name not in ACTIVATIONS:
        raise ValueError(
            f"unknown activation {name!r}; choices include "
            f"{sorted(ACTIVATIONS)[:8]}... ({len(ACTIVATIONS)} total)"
        )
    return ACTIVATIONS[name]


__all__ = ["ACTIVATIONS", "get_activation", "matern52"]
