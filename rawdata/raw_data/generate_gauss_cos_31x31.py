import os
import numpy as np
from typing import Optional


def generate_gauss_cos_grid(n_points: int = 31) -> np.ndarray:
    """
    Generate a structured dataset on a n_points x n_points grid over [-1, 1]^2
    for the function f(x1, x2) = exp(- (x1^2 + x2^2)/2) * cos(10*x1*x2),
    including its gradient with respect to (x1, x2).

    Returns a NumPy structured array with dtype:
      [('x', float64, (2,)), ('dv', float64, (2,)), ('v', float64)]
    """
    # Grid
    x1 = np.linspace(-1.0, 1.0, n_points)
    x2 = np.linspace(-1.0, 1.0, n_points)
    X1, X2 = np.meshgrid(x1, x2, indexing='xy')

    # Flatten
    x = X1.ravel()
    y = X2.ravel()

    # Function components
    g = np.exp(-0.5 * (x**2 + y**2))
    xy = 10.0 * x * y
    h = np.cos(xy)

    # Function value
    f = g * h

    # Gradient
    # df/dx = g * [ -x*h - 10*y*sin(10*x*y) ]
    # df/dy = g * [ -y*h - 10*x*sin(10*x*y) ]
    s = np.sin(xy)
    fx = g * (-x * h - 10.0 * y * s)
    fy = g * (-y * h - 10.0 * x * s)

    # Structured dtype to match existing datasets
    dtype = np.dtype([
        ('x', np.float64, (2,)),
        ('dv', np.float64, (2,)),
        ('v', np.float64),
    ])

    data = np.zeros(x.shape[0], dtype=dtype)
    data['x'] = np.stack([x, y], axis=1)
    data['dv'] = np.stack([fx, fy], axis=1)
    data['v'] = f

    return data


def generate_gauss_cos_random(
    n_samples: int = 1000,
    *,
    seed: Optional[int] = 0,
    low: float = -1.0,
    high: float = 1.0,
) -> np.ndarray:
    """
    Generate a random dataset with n_samples points uniformly sampled over [low, high]^2
    for the function f(x1, x2) = exp(- (x1^2 + x2^2)/2) * cos(10*x1*x2),
    including its gradient with respect to (x1, x2).

    Returns a NumPy structured array with dtype:
      [('x', float64, (2,)), ('dv', float64, (2,)), ('v', float64)]

    Notes:
      - Set seed=None to get different samples each run.
    """
    rng = np.random.default_rng(seed)

    # Sample points
    x = rng.uniform(low, high, size=n_samples)
    y = rng.uniform(low, high, size=n_samples)

    # Function components
    g = np.exp(-0.5 * (x**2 + y**2))
    xy = 10.0 * x * y
    h = np.cos(xy)

    # Function value
    f = g * h

    # Gradient
    # df/dx = g * [ -x*h - 10*y*sin(10*x*y) ]
    # df/dy = g * [ -y*h - 10*x*sin(10*x*y) ]
    s = np.sin(xy)
    fx = g * (-x * h - 10.0 * y * s)
    fy = g * (-y * h - 10.0 * x * s)

    # Structured dtype to match existing datasets
    dtype = np.dtype([
        ('x', np.float64, (2,)),
        ('dv', np.float64, (2,)),
        ('v', np.float64),
    ])

    data = np.zeros(x.shape[0], dtype=dtype)
    data['x'] = np.stack([x, y], axis=1)
    data['dv'] = np.stack([fx, fy], axis=1)
    data['v'] = f

    return data


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, 'gauss_cos_31x31.npy')
    # Generate 1000 random points (keeps filename for notebook compatibility).
    data = generate_gauss_cos_random(n_samples=1000, seed=0, low=-1.0, high=1.0)
    np.save(out_path, data)
    print(f"Saved dataset with shape {data.shape}, dtype: {data.dtype} to {out_path}")


if __name__ == '__main__':
    main()


