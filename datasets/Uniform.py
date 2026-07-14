"""Smoothed-uniform benchmark: ``Y = X + noise`` with uniform marginals.

Each coordinate of ``X`` is drawn uniformly on ``[0, 1]`` and observed through
additive uniform noise on ``[-eps, eps]``. The coordinates are independent, so
the total mutual information is ``n_dims`` times the per-coordinate MI, which has
the closed form in :func:`compute_MI_given_eps_unif`.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class Uniform(Dataset):
    """Independent smoothed-uniform pairs with a closed-form MI.

    Args:
        n_samples: number of pairs drawn at construction.
        n_dims: number of independent coordinates in each of X and Y.
        eps: half-width of the uniform additive noise ``[-eps, eps]``.
    """

    def __init__(self, n_samples=100000, n_dims=80, eps=0.5):
        self.n_dims = n_dims
        self.eps = eps
        self.data = self.sample_data(100000)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def sample_data(self, n_samples=10000):
        """Sample ``(X, Y)`` torch tensors of shape ``(n_samples, n_dims)``."""
        X, Y = sample_uniform(n_samples, self.n_dims, self.eps)
        return torch.Tensor(X), torch.Tensor(Y)

    def true_mutual_info(self):
        """Exact total MI in nats: per-coordinate MI times ``n_dims``."""
        return compute_MI_given_eps_unif(self.eps) * self.n_dims


def sample_uniform(n, d, eps):
    """Draw ``x ~ U(0, 1)^d`` and ``y = x + U(-eps, eps)^d``, both ``(n, d)``."""
    x = np.random.uniform(0, 1, (n, d))
    noise = np.random.uniform(-eps, eps, (n, d))
    y = x + noise
    return x, y


def compute_MI_given_eps_unif(eps):
    """Per-coordinate MI of ``Y = X + U(-eps, eps)`` with ``X ~ U(0, 1)``, in nats."""
    if eps > 0.5:
        return 1 / (4 * eps)
    else:
        return eps - np.log(2 * eps)
