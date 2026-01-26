import os
from typing import Optional

import numpy as np


def gaussian_window(x: np.ndarray) -> np.ndarray:
    """f(x) = exp(-x^2/2) * |sin(7 * sqrt(1 + x^2))|"""
    return np.exp(-0.5 * x * x) * np.abs(np.sin(7.0 * np.sqrt(1.0 + x * x)))


def gaussian_window_grad(x: np.ndarray) -> np.ndarray:
    """
    Analytic derivative of gaussian_window with respect to x.

    f(x) = exp(-x^2/2) * |sin(7 * sqrt(1 + x^2))|

    Note: the absolute value makes f non-differentiable where sin(...) == 0.
    We use sign(sin(.)) with 0 derivative at exactly 0.
    """
    g = np.exp(-0.5 * x * x)
    r = np.sqrt(1.0 + x * x)
    t = 7.0 * r
    s = np.sin(t)
    c = np.cos(t)
    a = np.abs(s)

    # da/dx = sign(s) * cos(t) * 7 * d(r)/dx, with d(r)/dx = x/r
    sign_s = np.sign(s)
    drdx = np.divide(x, r, out=np.zeros_like(x), where=r != 0)
    da = sign_s * c * 7.0 * drdx

    # dg/dx = -x * g
    return g * (-x * a + da)


def generate_gaussian_window_grid(
    n_points: int = 1000,
    *,
    low: float = -1.0,
    high: float = 1.0,
    seed: Optional[int] = 0,
    shuffle: bool = True,
) -> np.ndarray:
    """
    Generate n_points on a uniform grid in [low, high] and shuffle them.

    Returns a structured array with dtype:
      [('x', float64, (1,)), ('dv', float64, (1,)), ('v', float64)]
    """
    x = np.linspace(low, high, int(n_points))
    v = gaussian_window(x)
    dvx = gaussian_window_grad(x)

    dtype = np.dtype([
        ("x", np.float64, (1,)),
        ("dv", np.float64, (1,)),
        ("v", np.float64),
    ])
    data = np.zeros(x.shape[0], dtype=dtype)
    data["x"] = x[:, None]
    data["dv"] = dvx[:, None]
    data["v"] = v

    if shuffle:
        rng = np.random.default_rng(seed)
        data = data[rng.permutation(data.shape[0])]

    return data


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gaussian_window_1000.npy")
    data = generate_gaussian_window_grid(n_points=1000, low=-1.0, high=1.0, seed=42, shuffle=True)
    np.save(out_path, data)
    print(f"Saved dataset with shape {data.shape}, dtype: {data.dtype} to {out_path}")


if __name__ == "__main__":
    main()
