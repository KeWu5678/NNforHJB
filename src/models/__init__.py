"""Parametric value-function models for the PDAP outer loop.

Each model represents V(x):
  * ``SignedModel``      — pure signed shallow network  V = sum c_i sigma(w.x+b)^p
  * ``SemiconcaveModel`` — semiconcave  V = 0.5 C ||x||^2 - g(x), convex g
"""

from .net import ShallowNetwork
from .signed import SignedModel
from .semiconcave import SemiconcaveModel
from .build import build_model

__all__ = ["ShallowNetwork", "SignedModel", "SemiconcaveModel", "build_model"]
