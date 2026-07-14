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

Import the concrete classes from their modules, e.g.::

    from datasets.NonlinearGaussian import NonlinearGaussian
    from datasets.Student_t import MultivariateStudentT

Note: this package is intentionally a thin marker -- the class names collide
with their module names, so the classes are not re-exported at the package level
(doing so would shadow the submodules that some call sites import directly).
"""

__all__ = [
    "NonlinearGaussian",
    "MoG",
    "SwissRoll",
    "Spiral",
    "Uniform",
    "Student_t",
    "NLP",
]
