"""Open-loop optimal-control data generation.

The package keeps the shared PDAP-facing value-sample contract at this level and
groups domain-specific solvers below ``vdp`` and ``pendulum``.
"""

from src.OpenLoop.value_samples import ValueSamples

__all__ = ["ValueSamples"]
