"""Dimensionality-reduction / compression facilities.

- ``Autoencoder`` : a Softplus MLP autoencoder with MSE reconstruction loss and a
  shared compressibility metric.
- ``PCA``         : low-rank PCA projection reporting the same compressibility metric.
"""

from .autoencoder import Autoencoder
from .pca import PCA

__all__ = ["Autoencoder", "PCA"]
