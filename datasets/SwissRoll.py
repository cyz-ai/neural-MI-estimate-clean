"""Swiss-roll benchmark: a Gaussian copula embedded on a rolled 2D manifold.

The latent ``(X, Y)`` is a block-correlated Gaussian. Each coordinate is pushed
through its Gaussian CDF to a uniform, and the X-uniform is embedded onto a swiss
-roll curve in the plane while the Y-uniform is padded with independent noise
(see :meth:`SwissRoll.transformation`). The reported ``true_mutual_info`` is the
closed-form MI of the underlying Gaussian copula; the manifold embedding makes
the estimation task hard without changing that reference value.
"""

import numpy as np
import torch
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

import torch.distributions as distribution


class SwissRoll(Dataset):
    """Block-correlated Gaussian embedded on a swiss-roll manifold.

    Args:
        n_samples: number of latent samples drawn at construction.
        n_dims: total dimensionality (even); X and Y each get ``n_dims // 2``.
        rho: within-block correlation coefficient in ``(-1, 1)``.
        mu: scalar (or broadcastable) added to the zero mean.
    """

    def __init__(self, n_samples=100000, n_dims=80, rho=0.80, mu=0, jitter=0.05):
        self.n_dims = n_dims
        self.mu = np.zeros(self.n_dims) + mu
        self.rho = rho
        self.jitter = jitter                 # off/along-manifold noise added to the embedding
        self.rhos = np.ones(n_dims // 2) * self.rho
        self.cov_matrix = block_diag(*[[[1, self.rho], [self.rho, 1]] for _ in range(n_dims // 2)])
        self.data = self._sample_gaussian(n_samples, self.cov_matrix).astype(dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _sample_gaussian(self, n_samples, cov_matrix):
        """Draw ``n_samples`` rows from ``N(mu, cov_matrix)`` as a NumPy array."""
        mvn = multivariate_normal(mean=np.zeros(self.n_dims) + self.mu, cov=cov_matrix)
        return mvn.rvs(n_samples)

    @staticmethod
    def _get_rho_from_mi(mi, n_dims):
        """Correlation ``rho`` that yields mutual information ``mi`` at ``n_dims``."""
        x = (4 * mi) / n_dims
        return (1 - np.exp(-x)) ** 0.5

    @staticmethod
    def _get_mi_from_rho(rho, n_dims):
        """Closed-form MI of the block-correlated Gaussian with correlation ``rho``."""
        a = np.log(1 - rho**2)
        return -1/4.0*n_dims*a

    @staticmethod
    def u2xy(u):
        """Split an interleaved latent ``u`` into ``(X, Y)`` (even/odd columns)."""
        X, Y = u[:, ::2], u[:, 1::2]
        return X, Y

    @staticmethod
    def xy2u(X, Y):
        """Interleave ``(X, Y)`` back into a single latent ``u``."""
        n, d = X.shape
        samples = np.zeros((len(X), d*2))
        samples[:, ::2], samples[:, 1::2] = X, Y
        return samples

    def _numerator_log_prob(self, u):
        """Log density of the joint ``p(x, y)`` at interleaved points ``u``."""
        mvn = multivariate_normal(mean=np.zeros(self.n_dims) + self.mu, cov=self.cov_matrix)
        return mvn.logpdf(u)

    def _denominator_log_prob(self, u):
        """Log density of the product of marginals ``p(x) p(y)`` at ``u``."""
        mvn = multivariate_normal(mean=np.zeros(self.n_dims) + self.mu, cov=np.diag(np.ones(self.n_dims)))
        return mvn.logpdf(u)

    def sample_data(self, n_samples, mode='joint'):
        """Sample the latent and return its swiss-roll embedding ``(X, Y)``.

        Args:
            n_samples: number of rows to draw.
            mode: ``'joint'`` uses the correlated covariance; anything else
                samples X and Y independently (identity covariance).
        """
        cov = self.cov_matrix if mode == 'joint' else np.diag(np.ones(self.n_dims))
        data = self._sample_gaussian(n_samples, cov)
        X, Y = torch.Tensor(data[:, ::2]).clone(), torch.Tensor(data[:, 1::2]).clone()
        return self.transformation(X, Y)

    def log_ratio(self, X, Y):
        """Pointwise ``log p(x, y) / (p(x) p(y))`` for the latent Gaussian."""
        samples = np.zeros((len(X), self.n_dims))
        samples[:, ::2], samples[:, 1::2] = X, Y
        return self._numerator_log_prob(samples) - self._denominator_log_prob(samples)

    def true_mutual_info(self):
        """Exact MI in nats of the latent copula (the embedding preserves it)."""
        return self._get_mi_from_rho(self.rho, self.n_dims)

    def empirical_mutual_info(self):
        """Monte-Carlo MI estimate on the latent (mean log-ratio over 100k samples)."""
        samples = self._sample_gaussian(100000, self.cov_matrix)
        return np.mean(self._numerator_log_prob(samples) - self._denominator_log_prob(samples))

    @staticmethod
    def _roll(u):
        """Place a uniform ``u`` in ``[0, 1]`` on the swiss-roll curve ``(t cos t, t sin t)/21``."""
        t = 3*np.pi/2*(1 + 2*u)
        return t*torch.cos(t)/21, t*torch.sin(t)/21

    def transformation(self, x, y):
        """Embed latent ``(x, y)`` onto the swiss-roll manifold.

        Both sides are pushed through their Gaussian CDF to uniforms and placed on
        the swiss-roll curve ``(t cos t, t sin t)``: X carries ``ux``, Y carries
        ``uy``. Both embeddings are bijective, so MI is preserved; the symmetric
        rolling makes the task strictly harder. Small jitter is added to both.
        """
        # x, y -> F(x), F(y)
        mu, sigma = x*0, x*0 + 1
        normal = distribution.normal.Normal(mu, sigma)
        ux, uy = normal.cdf(x), normal.cdf(y)
        # both sides: swiss-roll embedding of their uniform
        e1, e2 = self._roll(ux)
        f1, f2 = self._roll(uy)
        X = torch.cat([e1, e2], dim=1)
        Y = torch.cat([f1, f2], dim=1)
        return X + self.jitter*torch.randn_like(X), Y + self.jitter*torch.rand_like(Y)

    def plot(self, X, Y):
        """3D scatter of the swiss-roll embedding for the first 1000 samples."""
        # Split the data into three arrays for plotting
        x = X[0:1000, 0].cpu().numpy()
        z = X[0:1000, 1].cpu().numpy()
        y = Y[0:1000, 0].cpu().numpy()

        # Create a new figure and a 3D subplot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the data
        ax.scatter(x, y, z, s=2)

        # grid and view angle
        ax.grid(False)
        ax.view_init(elev=35, azim=125)

        # Set labels for axes
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()
