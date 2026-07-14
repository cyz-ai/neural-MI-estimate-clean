"""Synthetic and real-data benchmarks with known (or exactly computable) MI.

Each module defines one dataset whose ground-truth mutual information is either
closed-form or exactly computable, for benchmarking neural MI estimators:

- ``NonlinearGaussian`` : Gaussian copula under invertible per-coordinate warps.
- ``MoG``               : mixture of block-correlated Gaussians.
- ``SwissRoll``         : Gaussian copula embedded on a swiss-roll manifold.
- ``Spiral``            : Gaussian copula bent by a norm-dependent rotation.
- ``Uniform``           : independent smoothed-uniform pairs.
- ``Student_t``         : multivariate Student-t (heavy-tailed dependence).
- ``NLP``               : IMDb/BERT text pairs with a tunable label-MI.
- ``image``             : Gaussian-image / rectangle image pairs with a target MI
                          (a package of helpers, not a single class). Ported from
                          the ICLR 2023 lossy-compression IB paper (see
                          ``image/README.md``).

Each concrete class is re-exported at the package level, so either form works::

    from datasets import NonlinearGaussian, MultivariateStudentT   # concise
    from datasets.NonlinearGaussian import NonlinearGaussian       # explicit module path

Both resolve to the same class; the explicit module path keeps working because the
submodule stays importable via ``sys.modules``. ``image`` is a package of helpers
(not a single class) and stays available as the submodule ``datasets.image``.
"""

from .NonlinearGaussian import NonlinearGaussian
from .MoG import MoG
from .SwissRoll import SwissRoll
from .Spiral import Spiral
from .Uniform import Uniform
from .Student_t import MultivariateStudentT
from .NLP import TextDataset
from . import image

__all__ = [
    "NonlinearGaussian",
    "MoG",
    "SwissRoll",
    "Spiral",
    "Uniform",
    "MultivariateStudentT",
    "TextDataset",
    "image",
]
