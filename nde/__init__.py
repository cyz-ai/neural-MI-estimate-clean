"""Neural density estimators.

Each module's main class is re-exported here, so ``from nde import MoG`` and
``from nde.MoG import MoG`` both work (they resolve to the same class):

- ``NAF`` : Neural Autoregressive Flow.
- ``MAF`` : Masked Autoregressive Flow.
- ``NFM`` : Neural flow-matching model (module ``nde.FM``).
- ``MoG`` : Mixture of Gaussians (full-covariance, learnable weights; also usable
            as a copula base).
- ``VGC`` : Vector Gaussian Copula (per-side marginal flows + a joint MoG base).

All modules train through the single shared ``optimizer.py``. VGC does
``from nde import NAF, MoG``, so NAF and MoG are imported before it.
"""

from .NAF import NAF
from .MoG import MoG
from .VGC import VGC
from .MAF import MAF
from .FM import NFM

__all__ = ["NAF", "MAF", "NFM", "MoG", "VGC"]
