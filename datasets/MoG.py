"""Mixture-of-Gaussians benchmark built from correlated Gaussian components.

The joint ``(X, Y)`` is an equally-weighted mixture of ``K`` block-correlated
Gaussians (each a :class:`~datasets.NonlinearGaussian.NonlinearGaussian`), where
component ``i`` has its own shift and within-block correlation. Because the
mixture breaks the Gaussian form, there is no closed-form MI; instead the MI is
estimated by Monte-Carlo through :meth:`MoG.empirical_mutual_info`, using the
log-ratio between the mixture joint and the product of its (mixture) marginals.
"""

import numpy as np
import torch
from scipy.special import logsumexp
from torch.utils.data import Dataset

from datasets.NonlinearGaussian import NonlinearGaussian


class MoG(Dataset):
    """Equally-weighted mixture of ``K`` block-correlated Gaussians.

    Args:
        n_samples: number of pairs drawn at construction.
        n_dims: total dimensionality (even); X and Y each get ``n_dims // 2``.
        K: number of mixture components.
        shifts: length-``K`` per-component mean shifts.
        rhos: length-``K`` per-component within-block correlations.
    """

    def __init__(self, n_samples=100000, n_dims=80, K=5, shifts=np.random.rand(1, 100) * 0, rhos=np.random.rand(1, 100)):
        self.n_dims = n_dims
        self.K = K
        # Full correlated joint component, plus the X-only / Y-only components
        # used to evaluate the product-of-marginals density.
        self.gaussians = [NonlinearGaussian(n_samples, n_dims, rhos[i], shifts[i]) for i in range(K)]
        self.gaussians_x = [NonlinearGaussian(n_samples, n_dims // 2, 0, shifts[i]) for i in range(K)]
        self.gaussians_y = [NonlinearGaussian(n_samples, n_dims // 2, 0, shifts[i]) for i in range(K)]
        self.data = self.sample_data(n_samples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def sample_data(self, n_samples):
        """Sample ``n_samples`` pairs by pooling equal shares of each component."""
        X, Y = [], []
        for i in range(self.K):
            X_i, Y_i = self.gaussians[i].sample_data(n_samples // self.K)
            X.append(X_i)
            Y.append(Y_i)
        X, Y = torch.cat(X, dim=0), torch.cat(Y, dim=0)
        idx = torch.randperm(n_samples)
        return X[idx], Y[idx]

    def _numerator_log_prob(self, u):
        """Log density of the mixture joint ``p(x, y)`` at interleaved points ``u``."""
        log_prob = np.zeros((len(u), self.K))
        for i in range(self.K):
            log_prob[:, i] = self.gaussians[i]._numerator_log_prob(u)
        return logsumexp(log_prob, 1) - np.log(self.K)

    def _denominator_log_prob(self, u):
        """Log density of the product of mixture marginals ``p(x) p(y)`` at ``u``."""
        log_prob_x = np.zeros((len(u), self.K))
        log_prob_y = np.zeros((len(u), self.K))
        x, y = NonlinearGaussian.u2xy(u)
        for i in range(self.K):
            log_prob_x[:, i] = self.gaussians_x[i]._numerator_log_prob(x)
            log_prob_y[:, i] = self.gaussians_y[i]._numerator_log_prob(y)
        return logsumexp(log_prob_x, 1) - np.log(self.K) + logsumexp(log_prob_y, 1) - np.log(self.K)

    def log_ratio(self, X, Y):
        """Pointwise ``log p(x, y) / (p(x) p(y))`` for the mixture."""
        samples = np.zeros((len(X), self.n_dims))
        samples[:, ::2], samples[:, 1::2] = X, Y
        return self._numerator_log_prob(samples) - self._denominator_log_prob(samples)

    def log_prob(self, X, Y):
        """Log density of the mixture joint for torch tensors ``X``, ``Y``."""
        samples = np.zeros((len(X), self.n_dims))
        samples[:, ::2], samples[:, 1::2] = X.cpu().numpy(), Y.cpu().numpy()
        return self._numerator_log_prob(samples)

    def true_mutual_info(self):
        """No closed form for a mixture; use :meth:`empirical_mutual_info`."""
        return "not implemented true MI"

    def empirical_mutual_info(self):
        """Monte-Carlo MI estimate (mean log-ratio over 100k fresh samples)."""
        X, Y = self.sample_data(100000)
        samples = np.zeros((len(X), self.n_dims))
        samples[:, ::2], samples[:, 1::2] = X, Y
        return np.mean(self._numerator_log_prob(samples) - self._denominator_log_prob(samples))
