"""Parametric value-function models for the PDAP outer loop.

Each model represents V(x) and conforms to the :class:`base.Model` protocol:
  * ``SignedModel``      — pure signed shallow network  V = sum c_i sigma(w.x+b)^p
  * ``SemiconcaveModel`` — semiconcave  V = 0.5 C ||x||^2 - g(x), convex g
"""

from .base import Model
from .net import ShallowNetwork
from .signed import SignedModel
from .semiconcave import SemiconcaveModel

__all__ = ["Model", "ShallowNetwork", "SignedModel", "SemiconcaveModel"]
