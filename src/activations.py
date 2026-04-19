"""Custom activation functions for shallow neural networks."""

import torch


def matern52(z: torch.Tensor) -> torch.Tensor:
    """Matérn 5/2 activation function.

    k(z) = (1 + sqrt(5)|z| + 5/3 * z^2) * exp(-sqrt(5)|z|)

    Properties:
      - Smooth and non-negative (safe for power p).
      - k(0) = 1, decays to 0 as |z| -> inf.
      - NOT positively homogeneous: must use use_sphere=False.
    """
    r = torch.abs(z)
    sqrt5 = 2.23606797749979
    return (1.0 + sqrt5 * r + (5.0 / 3.0) * r.pow(2)) * torch.exp(-sqrt5 * r)
