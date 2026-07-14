"""Neural mutual-information estimators.

Each module defines one estimator as an ``nn.Module`` with a shared interface
(``objective_func``, ``MI``, ``learn``):

- ``MINE``   : Mutual Information Neural Estimation (Donsker-Varadhan bound).
- ``InfoNCE``: contrastive lower bound (a.k.a. NCE / CPC).
- ``SMILE``  : Smoothed MI Lower-bound Estimator (clipped DV).
- ``MRE``    : Mutual-information Ratio Estimator.
- ``DoE``    : Difference-of-Entropies, ``I = H(X) - H(X|Y)`` via two flows.
- ``MINDE``  : diffusion / score-based MI (see the ``minde`` subpackage).
- ``MIENF``  : MI via a normalizing-flow density (single Gaussian-copula base).
- ``VCE``    : two-stage Vector Copula Estimator (marginal flows + MoG copula).

Import the concrete classes from their modules, e.g.::

    from estimators.VCE import VCE
    from estimators.MINE import MINE

Note: this package is intentionally a thin marker -- the class names collide
with their module names, so the classes are not re-exported at the package level
(doing so would shadow the submodules that call sites import directly).
"""

__all__ = [
    "MINE",
    "InfoNCE",
    "SMILE",
    "MRE",
    "DoE",
    "MINDE",
    "MIENF",
    "VCE",
]
