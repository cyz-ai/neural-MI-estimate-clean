import torch
import torch.nn as nn

import optimizer
from nde.FM import NFM
from nde.NAF import NAF
from nde.MoG import MoG


class VCE(nn.Module):
    """VCE — the two-stage Vector Copula Estimator

    Stage A: two independent marginal flows (``flow_x`` on X, ``flow_y`` on Y) Gaussianize each side.
    The flow family is set by ``hyperparams.marginal_flow`` -- 'FM' (flow-matching :class:`nde.FM.NFM`,
    default) or 'NAF' (:class:`nde.NAF.NAF`).

    Stage B: a best-of-N Mixture-of-Gaussian-Copulas (``nde.MoG.MoG`` with ``M=n_restarts``,
    ``copula=True``) fits the joint copula on the shared latents; ``n_restarts`` members are trained
    JOINTLY (one vectorized optimizer) from DIFFERENT initializations and the lowest-NLL member is
    kept. MI is read off that mixture via KL_joint_marginal.
    """
    def __init__(self, hyperparams):
        super().__init__()
        self.d = None                      # per-side dim, inferred at learn() time (square copula)

        # Only K and the restart count are configurable; the copula's optimizer hyperparams
        # (lr=0.02, bs=1000) are hardcoded in nde.MoG.MoG (established sweet spot, not tuned per-run).
        self.K_components = 32 if not hasattr(hyperparams, 'K_components') else hyperparams.K_components            # <- number of mixture components in the copula MoG
        self.n_restarts = 4 if not hasattr(hyperparams, 'n_restarts') else hyperparams.n_restarts                   # <- number of copula restarts (jointly trained, lowest-NLL one kept)

        self.bon_selection = True if not hasattr(hyperparams, 'bon_selection') else hyperparams.bon_selection
        self.marginal_flow = 'FM' if not hasattr(hyperparams, 'marginal_flow') else hyperparams.marginal_flow

        self.flow_x = None                                                                                          # <- marginal flow on X (NFM or NAF), set in learn()
        self.flow_y = None                                                                                          # <- marginal flow on Y (NFM or NAF), set in learn()
        self.mog = None                                                                                             # <- the MoG copula fit on the flow latents, set in learn()
        self._cached_latents = None                                                                                 # <- (v, w) flow latents cached in learn() for _refit_copula()

    def MI(self, x, y, reused_latent=True):
        self.eval()
        if reused_latent and self._cached_latents is not None:
            v, w = self._cached_latents          # skip the flow forward; reuse learn()'s latents
        else:
            with torch.no_grad():
                v, w = self._flow_latents(x, y)
        return self.mog.KL_joint_marginal(v, w)

    def learn(self, x, y):
        assert x.size(1) == y.size(1), "VCE assumes dim_x == dim_y (square copula)."
        self.d = x.size(1)                 # per-side dim inferred here
        # A. two marginal flows (once; shared across all copula restarts)
        self.learn_flows(x, y)
        with torch.no_grad():
            v, w = self._flow_latents(x, y)
            v, w = v.clone().detach(), w.clone().detach()
        self._cached_latents = (v, w)          # cache the shared latents for cheap copula-only refits
        # B. jointly train n_restarts copula members on the shared latents; keep lowest-NLL one
        self.mog = self._fit_best_mog(v, w)

    def learn_flows(self, x, y):
        """Train the two independent marginal flows (flow_x on X, flow_y on Y). ``marginal_flow``
        selects the family: 'FM' (flow-matching NFM, default) or 'NAF' (neural autoregressive flow).
        Only the flow construction is gated; everything downstream is identical."""
        if self.marginal_flow == 'NAF':
            self.flow_x = NAF(n_blocks=3, n_inputs=self.d, n_hidden=400, n_cond_inputs=2).to(x.device)
            self.flow_y = NAF(n_blocks=3, n_inputs=self.d, n_hidden=400, n_cond_inputs=2).to(y.device)
        else:
            self.flow_x = NFM(n_inputs=self.d).to(x.device)
            self.flow_y = NFM(n_inputs=self.d).to(y.device)
        self.flow_x.max_iteration = 2500
        self.flow_y.max_iteration = 2500
        self.flow_x.learn(x)
        self.flow_y.learn(y)

    def _flow_latents(self, x, y):
        """Gaussianize each side through its marginal flow. Returns (z_x, z_y)."""
        v, _ = self.flow_x.forward(x)
        w, _ = self.flow_y.forward(y)
        return v, w

    def _refit_copula(self, K=None, n_restarts=None):
        """Re-fit ONLY the MoG copula on the cached flow latents from the last ``learn()``, skipping
        the expensive flow stage (~84% of a run). Makes copula / read-off experiments nearly free.

        Optional ``K`` / ``n_restarts`` override the stored settings (and are persisted, so ``MI`` reads
        stay consistent). Updates ``self.mog`` and returns it."""
        assert self._cached_latents is not None, "call learn() before _refit_copula()"
        if K is not None:
            self.K_components = K
        if n_restarts is not None:
            self.n_restarts = n_restarts
        v, w = self._cached_latents
        self.mog = self._fit_best_mog(v, w)
        return self.mog

    def _fit_best_mog(self, v, w):
        """Fit the best-of-N copula mixture on the shared latents (v, w) and return it. With
        ``bon_selection`` (default), prune the fitted mixture on held-out data to curb the plug-in
        K-over-read; otherwise keep the full-K fit. See :class:`nde.MoG.MoG` for the mechanics."""
        mog = MoG(d=self.d, K=self.K_components, M=self.n_restarts, copula=True).to(v.device)
        mog.learn(v, w)
        if self.bon_selection:                          # test-time held-out pruning of the mixture
            _, _, v_val, w_val = optimizer.NNOptimizer.divide_train_val(v, w)
            mog.test_time_BoN_selection(v_val, w_val)
        self.restart_nlls = mog.member_nlls
        self.best_nll = mog.best_nll
        return mog
