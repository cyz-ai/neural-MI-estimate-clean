import torch
import torch.nn as nn
import optimizer
from nde import NAF, MoG


class VGC(nn.Module):
    """Vector Gaussian Copula: per-side marginal flows + one joint MoG base for MI estimation.

    The marginal flows (NAF) Gaussianize each side; a single fixed-K Mixture-of-Gaussians
    models the joint copula density in the shared latent, and MI is read off as that density's
    log-ratio (the 'mog' read-off).

    ``learning_mode`` is the single knob controlling how flows and base are fit:
      - 'joint': flows + base trained end-to-end, maximizing the joint log-likelihood log p(x,y);
        the base gradient backprops through the flows.
      - 'separate': each marginal flow is pre-trained on its own, then frozen while the base fits
        (the shared latents are detached).

    Supports rectangular MI (dim_x != dim_y): the two marginal flows and the joint base are
    sized from ``n_inputs`` (X-side) and ``n_inputs_y`` (Y-side, defaults to ``n_inputs``).
    """
    def __init__(self, n_blocks, n_inputs, n_hidden, n_cond_inputs=2, K=16,
                 learning_mode='joint', n_inputs_y=None, lr=1e-3, bs=250, max_iteration=200):
        super().__init__()
        assert learning_mode in ('joint', 'separate'), \
            f"Unknown learning_mode: {learning_mode!r}. Use 'joint' or 'separate'."
        # n_inputs is the X-side dim; n_inputs_y the Y-side dim (defaults to n_inputs for the
        # common square case). Marginal flows and the joint base are sized accordingly.
        n_inputs_y = n_inputs if n_inputs_y is None else n_inputs_y
        self.dim_x, self.dim_y = n_inputs, n_inputs_y
        self.maf1 = NAF(n_blocks, n_inputs, n_hidden, n_cond_inputs)
        self.maf2 = NAF(n_blocks, n_inputs_y, n_hidden, n_cond_inputs)
        # single fixed-K joint Mixture-of-Gaussians copula base over the shared latent
        self.base = MoG(n_in=2, n_hidden=10, n_out=n_inputs + n_inputs_y, K=K)
        self.K = K
        self.max_iteration = max_iteration
        self.lr = lr
        self.bs = bs
        self.learning_mode = learning_mode

    def forward(self, x, y):
        """Transform (x, y) through the marginal flows. Returns (z_x, z_y)."""
        xx, _ = self.maf1.forward(x)
        yy, _ = self.maf2.forward(y)
        return xx, yy

    def _forward_latents(self, xy):
        """Push (x, y) through the marginal flows once. Returns (xxyy, log_jacob).

        In 'joint' mode the latents keep their graph so the base's log-likelihood backprops into
        the flows; in 'separate' mode they are detached (the flows are pre-trained and frozen).
        """
        n, d = xy.size()
        x, y = xy[:, :self.dim_x], xy[:, self.dim_x:]
        xx, log_jacob_xx = self.maf1.forward(x)
        yy, log_jacob_yy = self.maf2.forward(y)
        if self.learning_mode == 'separate':
            xx, yy = xx.detach(), yy.detach()
            log_jacob_xx, log_jacob_yy = log_jacob_xx.detach(), log_jacob_yy.detach()
        xxyy = torch.cat([xx, yy], dim=1)
        log_jacob = (log_jacob_xx + log_jacob_yy).view(n, -1)
        return xxyy, log_jacob

    def _base_log_prob(self, xxyy, log_jacob):
        """Joint log p(x,y) = log p_base(z) + log|J| on the shared latents."""
        n = xxyy.size(0)
        t = torch.ones(n, 2, device=xxyy.device)
        log_base_prob = self.base.log_probs(inputs=xxyy, cond_inputs=t).view(n, -1)
        return (log_base_prob + log_jacob).view(-1)

    def log_prob(self, xy):
        """Joint log p(x,y) = log p_base(f(x), g(y)) + log|J_f| + log|J_g|."""
        xxyy, log_jacob = self._forward_latents(xy)
        return self._base_log_prob(xxyy, log_jacob)

    def objective_func(self, x, y):
        """Training objective: the joint log-likelihood log p(x,y) under the base (base density +
        flow log-Jacobians). In 'joint' mode this backprops into the flows; in 'separate' mode the
        latents are detached in ``_forward_latents`` so only the base is fit."""
        xy = torch.cat([x, y], dim=1)
        xxyy, log_jacob = self._forward_latents(xy)
        return self._base_log_prob(xxyy, log_jacob).mean()

    def learn(self, x, y):
        """Fit the VGC. In 'separate' mode each marginal flow is pre-trained first (then frozen);
        both modes then fit the base (jointly with the flows in 'joint' mode) by maximizing the
        joint log-likelihood."""
        if self.learning_mode == 'separate':
            if self.maf1.max_iteration > 0:
                self.maf1.learn(x)
            if self.maf2.max_iteration > 0:
                self.maf2.learn(y)
        if self.max_iteration > 0:
            optimizer.NNOptimizer.learn(self, x=x, y=y)

    def MI(self, x, y, inner=True):
        """MI in the latent space via the joint MoG (mog read-off):
            MI = E[log q(z_x, z_y) - log q_x(z_x) - log q_y(z_y)]
        where q_x, q_y are the base's analytic marginals. ``inner=False`` first pushes (x, y)
        through the marginal flows (MI is invariant to those invertible maps)."""
        if inner is False:
            x, y = self.forward(x, y)
        xy = torch.cat([x, y], dim=1)
        n, dx = x.size()
        dy = y.size(1)
        t = torch.ones(n, 2, device=xy.device)
        log_joint = self.base.log_probs(inputs=xy, cond_inputs=t)
        lx = self.base.log_probs_marginal(inputs=xy, cond_inputs=t, marginals=list(range(dx)))
        ly = self.base.log_probs_marginal(inputs=xy, cond_inputs=t, marginals=list(range(dx, dx + dy)))
        return (log_joint - lx - ly).mean().item()
