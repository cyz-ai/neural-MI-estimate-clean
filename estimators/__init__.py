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

Each class is re-exported at the package level, so either form works::

    from estimators import VCE          # concise
    from estimators.VCE import VCE      # explicit module path

(The re-export binds the ``estimators.VCE`` attribute to the class, but the
submodule stays importable via ``sys.modules``, so the explicit form below keeps
working too -- both resolve to the same class.)
"""

from .MINE import MINE
from .InfoNCE import InfoNCE
from .SMILE import SMILE
from .MRE import MRE
from .DoE import DoE
from .MINDE import MINDE
from .MIENF import MIENF
from .VCE import VCE

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
