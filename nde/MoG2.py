import math
import torch
import torch.nn as nn
import torch.distributions as distribution

import optimizer



class MoG2(nn.Module):
    """EXPERIMENTAL sandbox copy of :class:`nde.MoG.MoG` -- safe to modify.

    Byte-for-byte the production MoG at fork time (same params, training, read-off, u-space view).
    Put experimental things here (copula-likelihood objective, Gaussian-copula / correlation-matrix
    components, shrinkage regularization, iterative Gaussianization, ...) so the production MoG's
    training facilities stay untouched. Wire it into VCE via a small experimental hook, not by
    editing MoG.

    ---- original docstring ----
    Mixture of Gaussians with full covariances and learnable weights, vectorized over M
    parallel members and optionally used as a copula.

    A single member is an unconditional K-component Gaussian mixture, fully described by K weights,
    means, and Cholesky-of-*precision* factors. All M members' parameters are stacked on a leading
    member axis (``[M,K,...]``) so every member's density is evaluated in one batched pass -- M
    restarts cost ~one fit. Two roles, one class:

    * **VGC's base** (``M=1``, ``copula=False``): a single joint mixture over the flow latents. VGC
      queries :meth:`log_probs` / :meth:`log_probs_marginal` and trains these params end-to-end via
      its own objective. It is unconditional -- ``cond_inputs`` is accepted for interface
      compatibility and ignored.

    * **The best-of-M copula** (``M>1``, ``copula=True``): :meth:`learn` fits M members JOINTLY
      from different inits on the (rank-transformed) inputs and keeps the lowest-NLL member; the
      mutual information is read off that member's copula ratio
      E[log q(x,y) - log q(x) - log q(y)] via :meth:`KL_joint_marginal`. Best-of-M is needed
      because the fit is bistable (~30% of inits collapse to a single Gaussian, distinctly higher
      NLL); the lowest NLL is a ground-truth-free collapse detector (all-collapse chance ~0.3**M).
    """
    def __init__(self, n_in=None, n_hidden=None, n_out=None, K=1, M=1, copula=False, d=None,
                 mean_init=2.0, gaussian_copula=False, copula_shrink=0.1):
        super().__init__()
        # Accept both the historical MoG signature (n_out = total joint dim, plus the now-unused
        # n_in / n_hidden of the old conditional scaffold) and a per-side ``d`` (D = 2d) shorthand.
        if n_out is not None:
            D = n_out
        elif d is not None:
            D = 2 * d
        else:
            raise ValueError("MoG2 needs n_out (total dim) or d (per-side dim).")
        self.D = D
        self.d = D // 2                     # per-side dim (for the copula halves)
        self.K, self.M = K, M
        self.copula = copula                # apply the rank/copula transform in forward()/learn()
        # EXPERIMENTAL "copula-based training" mode. gaussian_copula=True pins every component to
        # zero mean with a CORRELATION-matrix covariance (unit diagonal), so each per-dim marginal
        # is exactly N(0,1). The mixture is then a proper Gaussian copula: its marginals never drift,
        # the marginal terms in log p(u)=log q_J-sum_i log q_i are constant, and ordinary joint MLE
        # ALREADY maximizes the copula likelihood -- no separate (divergent) copula objective needed.
        # copula_shrink in [0,1) shrinks each R toward I (R<-(1-s)R+sI); it keeps the unit diagonal
        # (marginals stay N(0,1)) while bounding |R| away from 0, taming the log|R|->-inf blow-up on
        # near-collinear latents (e.g. student-t df=1). See _corr_chol_logdet. Default 0.1: at s=0 the
        # copula MLE diverges (mog full-K read blew up to ~40 vs GT 6.9); s=0.05-0.1 reads GT with an
        # NLL matching the free mixture (facility test 2026-07-14).
        self.gaussian_copula = gaussian_copula
        self.copula_shrink = copula_shrink

        # mixture parameters, stacked on the leading member axis; init follows the retired MDN's
        # nn.Linear(1, .) bias init -- U(-1, 1) -- EXCEPT the means, drawn from U(-mean_init, mean_init).
        # The rank-space data lives on a shell at radius ~sqrt(D); U(-1,1) means sit well inside it
        # (radius ~sqrt(D/3)), so most components are pulled together and collapse during training.
        # A wider mean range (mean_init=2) pushes the means out toward the data shell to resist that
        # collapse. logits/cov_raw stay at U(-1,1): a wider logit range would spread the initial
        # weights and start some components closer to dead, the opposite of the intent.
        self.logits = nn.Parameter(torch.empty(M, K).uniform_(-1, 1))                      # weights
        if gaussian_copula:
            # zero-mean components: means pinned to 0 (non-trainable) so per-dim marginals stay N(0,1)
            self.register_buffer('means', torch.zeros(M, K, D))
        else:
            self.means = nn.Parameter(torch.empty(M, K, D).uniform_(-mean_init, mean_init))  # means
        self.cov_raw = nn.Parameter(torch.empty(M, K, D, D).uniform_(-1, 1))               # raw Chol

        # NNOptimizer hyperparams for the copula fit -- HARDCODED, not tuned per-run. bs=1000, lr=0.02
        # is the established sweet spot: lr<=5e-3 collapses to the single-Gaussian basin, and some
        # bs/lr corners fit poorly, so these are fixed (session_2026_07_12 lr/bs study).
        self.bs = 5000
        self.lr = 0.02
        self.wd = 0e-5
        self.max_iteration = 5000
        self.best_idx = 0                   # selected member (set in learn); member 0 for M=1
        self.active = None                  # EXPERIMENTAL per-member component mask [M,K] bool
                                            # (None = all K active; set by test_time_BoN_selection)

    # ---------- shared Gaussian-mixture kernel ----------
    def _chol_logdet(self):
        """Lower-triangular Cholesky-of-precision factors C [M,K,D,D] (exp on the diagonal) and
        log|det C| [M,K]."""
        if self.gaussian_copula:
            return self._corr_chol_logdet()
        tril = torch.tril(self.cov_raw, diagonal=-1)
        diag_raw = torch.diagonal(self.cov_raw, dim1=-2, dim2=-1)             # [M,K,D]
        C = tril + torch.diag_embed(torch.exp(diag_raw))                     # [M,K,D,D]
        return C, diag_raw.sum(-1)                                            # C, logdet [M,K]

    def _corr_chol_logdet(self):
        """Cholesky-of-precision C [M,K,D,D] and logdet(C) [M,K] for zero-mean components whose
        covariance is a CORRELATION matrix (unit diagonal), optionally shrunk toward I.

        cov_raw is read as a Cholesky-of-covariance G (lower-tri, exp on the diagonal); dividing each
        row of G by its own norm sqrt(diag(G G^T)) yields L_R, the Cholesky of the correlation matrix
        R = diag(Sigma)^-1/2 Sigma diag(Sigma)^-1/2 (unit diagonal by construction). Shrinkage
        R <- (1-s)R + sI is applied on R (still unit diagonal, so marginals stay N(0,1)) and R is
        re-factored. The precision Cholesky is C = L_R^-1 (lower-triangular), giving
        C^T C = R^-1 and logdet(C) = -sum log diag(L_R) = -1/2 log|R|."""
        G = torch.tril(self.cov_raw, diagonal=-1)
        diag_raw = torch.diagonal(self.cov_raw, dim1=-2, dim2=-1)             # [M,K,D]
        G = G + torch.diag_embed(torch.exp(diag_raw))                        # Cholesky-of-covariance
        var = (G ** 2).sum(-1).clamp_min(1e-12)                              # [M,K,D] = diag(G G^T)
        L_R = G * var.rsqrt()[..., None]                                     # Cholesky-of-correlation
        if self.copula_shrink > 0:
            eye = torch.eye(self.D, device=G.device, dtype=G.dtype)
            R = L_R @ L_R.transpose(-1, -2)
            R = (1 - self.copula_shrink) * R + self.copula_shrink * eye
            L_R = torch.linalg.cholesky(R)
        eye = torch.eye(self.D, device=G.device, dtype=G.dtype).expand_as(L_R)
        C = torch.linalg.solve_triangular(L_R, eye, upper=False)            # L_R^-1 (lower-tri)
        logdet = -torch.log(torch.diagonal(L_R, dim1=-2, dim2=-1)).sum(-1)   # logdet(C) = -1/2 log|R|
        return C, logdet                                                     # C, logdet [M,K]

    def _member_logprobs(self, Z, active=None):
        """Per-member log-density of Z [n, D]: returns [M, n] (white = C @ (z - mu), standard
        normal density, log-sum-exp over the K components).

        ``active`` [M,K] bool restricts each member to a component subset (weights renormalized
        over the subset); defaults to ``self.active`` (None = all K components, the canonical path)."""
        active = self.active if active is None else active
        C, logdet = self._chol_logdet()
        diff = Z[None, None] - self.means[:, :, None, :]                      # [M,K,n,D]
        white = torch.einsum('mkij,mknj->mkni', C, diff)                     # [M,K,n,D]
        log_base = -0.5 * (white ** 2).sum(-1) - 0.5 * self.D * math.log(2 * math.pi)   # [M,K,n]
        logits = self.logits if active is None else self.logits.masked_fill(~active, float('-inf'))
        log_w = torch.log_softmax(logits, dim=-1)                            # [M,K] (renorm over subset)
        log_comp = log_base + logdet[:, :, None] + log_w[:, :, None]         # [M,K,n]
        return torch.logsumexp(log_comp, dim=1)                              # [M,n]

    def _member_cov(self, m):
        """Covariance [K,D,D] of member m (invert the precision = C^T C, symmetrized).

        Uses the non-throwing ``inv_ex``, retrying on a jittered precision if it reports singular.
        A heavy-tailed base (e.g. student-t df=1) can drive the Cholesky diagonal to exp
        over/underflow, making ``prec`` numerically singular; ``torch.inverse`` would raise there.
        Any residual inf/nan is sanitized so the read-out stays finite (possibly biased) rather
        than crashing. Read-out only -- not used in training."""
        C, _ = self._chol_logdet()
        prec = torch.nan_to_num(C[m].transpose(-1, -2) @ C[m])              # [K,D,D]
        cov, info = torch.linalg.inv_ex(prec)
        if int(info.max()) != 0:
            eye = torch.eye(prec.size(-1), device=prec.device, dtype=prec.dtype)
            cov, _ = torch.linalg.inv_ex(prec + 1e-6 * eye)
        cov = torch.nan_to_num(cov)
        return 0.5 * (cov + cov.transpose(-1, -2))

    def _marginal_logprob(self, Zs, dims, m, active=None):
        """Log-density of member m MARGINALIZED to ``dims``, at Zs [n, len(dims)]. Marginal of a
        Gaussian mixture = same weights over the covariance sub-block Gaussians.

        ``active`` [M,K] bool restricts member m to a component subset (weights renormalized);
        defaults to ``self.active``. Inactive components are SLICED OUT (not masked) so a degenerate
        collapsed member cannot poison the sum with a nan before its weight is zeroed."""
        active = self.active if active is None else active
        mean = self.means[m][:, dims]                        # [K,s]
        cov = self._member_cov(m)[:, dims][:, :, dims]       # [K,s,s]
        logits_m = self.logits[m]                            # [K]
        if active is not None:
            sel = active[m]
            mean, cov, logits_m = mean[sel], cov[sel], logits_m[sel]
        logw = torch.log_softmax(logits_m, dim=-1)           # [V] (renorm over subset)
        L = _psd_cholesky(0.5 * (cov + cov.transpose(-1, -2)))
        mvn = distribution.MultivariateNormal(mean, scale_tril=L, validate_args=False)
        logp_k = mvn.log_prob(Zs[:, None, :])                # [n,K]
        return torch.logsumexp(logp_k + logw, dim=1)         # [n]

    # ---------- VGC-facing interface (single mixture, member 0, unconditional) ----------
    def log_probs(self, inputs, cond_inputs=None):
        """Joint mixture log-density [n]. ``cond_inputs`` is accepted for interface compatibility
        and ignored (this base is unconditional). Uses member 0 (VGC's base is M=1)."""
        return self._member_logprobs(inputs)[0]

    def log_probs_marginal(self, inputs, cond_inputs=None, marginals=None):
        """Log-density [n] of the ``marginals`` dimensions under member 0's mixture."""
        dims = list(marginals)
        return self._marginal_logprob(inputs[:, dims], dims, m=0)

    # ---------- copula facilities (M-member best-of-N + rank transform + ratio read-off) ----------
    def forward(self, x, y):
        """Copula rank-transform (empirical CDF -> N(0,1)) each dim; returns the two halves.
        Identity when ``copula`` is False (e.g. VGC's base)."""
        if not self.copula:
            return x, y
        data = torch.cat([x, y], dim=1)
        _, idx = torch.sort(data, dim=0)
        _, idx2 = torch.sort(idx, dim=0)
        u = (idx2.float() + 1) / (len(data) + 1)
        z = distribution.Normal(torch.zeros_like(data), torch.ones_like(data)).icdf(u)
        n, d = z.size()
        return z[:, :d // 2], z[:, d // 2:]

    def objective_func(self, z, cond=None):
        """MLE over all M members (one optimizer trains them jointly)."""
        return self._member_logprobs(z).mean(dim=1).mean()

    def learn(self, x, y):
        """Train M members jointly on the (rank-transformed, if copula) latents and select the
        lowest-NLL member. Stores ``member_nlls`` / ``best_nll`` / ``best_idx`` for inspection.

        Follows the centralized protocol -- the optimizer owns the 80/20 split and early stopping.
        The split is DETERMINISTIC (no shuffle, matching every other NDE), so a caller can re-derive
        the exact held-out val with ``optimizer.NNOptimizer.divide_train_val`` -- VCE relies on this
        to score its test-time BoN selection on data the copula gradient never saw."""
        vv, ww = self.forward(x, y)
        z = torch.cat([vv, ww], dim=1)
        optimizer.NNOptimizer.learn(self, x=z, y=torch.ones_like(z))
        with torch.no_grad():
            self.member_nlls = [float(v) for v in (-self._member_logprobs(z).mean(dim=1)).tolist()]
            self.best_idx = int(min(range(self.M), key=lambda i: self.member_nlls[i]))
            self.best_nll = self.member_nlls[self.best_idx]
        return self

    def test_time_BoN_selection(self, x, y):
        """EXPERIMENTAL -- test-time best-of-N selection of the learned mixture, scored on ``(x, y)``.

        Pass a HELD-OUT split (data the mixture was NOT fit on). For each of the M members, sort its
        K components by mixture weight and evaluate the nested ``top-V`` sub-mixtures (V = 1..K,
        weights renormalized to sum to 1) by NLL on ``(x, y)``, keeping the best V per member, then
        re-run best-of-M on those truncated NLLs. Sets ``self.active`` [M,K] and ``self.best_idx``;
        returns a diagnostics dict. Call ``reset_BoN_selection()`` to restore full K.

        Held-out scoring is what makes this move the estimate: an overfit component that lowers
        in-sample NLL but RAISES held-out NLL can be pruned -- the one lever an in-sample criterion is
        forbidden to pull. (Scoring on the training data would be a provable IDENTITY OP, since the
        mixture already minimizes exactly that in-sample NLL.)

        The nested top-V NLLs are computed in one shot per member with ``logcumsumexp`` over the
        weight-sorted components -- O(K) candidates, so this scales to large K (unlike an exhaustive
        2**K subset search)."""
        with torch.no_grad():
            v, w = self.forward(x, y)
            z = torch.cat([v, w], dim=1)
            C, logdet = self._chol_logdet()
            diff = z[None, None] - self.means[:, :, None, :]                  # [M,K,n,D]
            white = torch.einsum('mkij,mknj->mkni', C, diff)
            log_base = -0.5 * (white ** 2).sum(-1) - 0.5 * self.D * math.log(2 * math.pi)  # [M,K,n]
            A = log_base + logdet[:, :, None] + self.logits[:, :, None]       # [M,K,n] unnormalized

            active = torch.zeros(self.M, self.K, dtype=torch.bool, device=z.device)
            best_nll, best_V = [], []
            for m in range(self.M):
                order = torch.argsort(self.logits[m], descending=True)        # by weight (softmax monotone)
                As = A[m][order]                                             # [K,n] weight-sorted
                lg = self.logits[m][order]                                   # [K]
                t1 = torch.logcumsumexp(As, dim=0)                           # [K,n]  top-V numerator
                t2 = torch.logcumsumexp(lg, dim=0)                           # [K]    top-V normalizer
                nll = -(t1 - t2[:, None]).mean(dim=1)                        # [K]    NLL for V=1..K
                V = int(torch.argmin(nll)) + 1
                active[m, order[:V]] = True
                best_nll.append(float(nll[V - 1]))
                best_V.append(V)
            self.active = active
            self.trunc_nlls = best_nll
            self.trunc_V = best_V
            self.best_idx = int(min(range(self.M), key=lambda i: best_nll[i]))
            self.best_nll_trunc = best_nll[self.best_idx]
        return {
            'best_idx': self.best_idx,
            'V_per_member': best_V,               # chosen subset size per member
            'trunc_nlls': best_nll,               # best top-V NLL per member (on the scoring data)
            'full_nlls': getattr(self, 'member_nlls', None),   # full-K NLL per member (from learn)
        }

    def reset_BoN_selection(self):
        """Restore the full-K mixture (undo :meth:`test_time_BoN_selection`)."""
        self.active = None
        return self

    def fake_critic(self, x, y):
        """Copula log-ratio log q(x,y) - log q(x) - log q(y) of the selected member, on the
        (already rank-transformed) halves x, y."""
        d = self.d
        l_xy = self._member_logprobs(torch.cat([x, y], dim=1))[self.best_idx]
        l_x = self._marginal_logprob(x, list(range(d)), self.best_idx)
        l_y = self._marginal_logprob(y, list(range(d, 2 * d)), self.best_idx)
        return l_xy - l_x - l_y

    def KL_joint_marginal(self, x, y):
        x, y = self.forward(x, y)
        return self.fake_critic(x, y).mean().item()

    # ---------- copula (u-space) view ----------
    # The fitted joint factorizes as q_J(z) = p(u) * prod_i q_i(z_i), where u_i = Q_i(z_i) is the
    # element-wise PIT under the member's own per-dim marginal Q_i, and p(u) is the copula density
    # (uniform marginals by construction). These helpers expose q_i, u, and p(u). NOTE: for MI between
    # the X/Y blocks the per-dim marginals cancel, so this view does NOT change KL_joint_marginal.
    def _marginal_mixture(self, m=None, active=None):
        """(w [V], mu [V,D], sigma [V,D]) of member m's per-dim marginals: weights, and each active
        component's mean and per-dim std (sqrt of the COVARIANCE diagonal, i.e. the true 1-D scale)."""
        m = self.best_idx if m is None else m
        active = self.active if active is None else active
        mu = self.means[m]                                            # [K,D]
        var = self._member_cov(m).diagonal(dim1=-2, dim2=-1)          # [K,D] per-dim variances
        logits_m = self.logits[m]                                     # [K]
        if active is not None:
            sel = active[m]
            mu, var, logits_m = mu[sel], var[sel], logits_m[sel]
        w = torch.softmax(logits_m, dim=-1)                           # [V]
        return w, mu, var.clamp_min(1e-12).sqrt()                     # [V], [V,D], [V,D]

    def marginal_logpdf(self, z, m=None, active=None):
        """Per-dim marginal log-density log q_i(z_i) of the fitted member. z [n,D] -> [n,D]."""
        w, mu, sigma = self._marginal_mixture(m, active)
        zc = z[:, None, :]                                            # [n,1,D]
        logphi = (-0.5 * ((zc - mu[None]) / sigma[None]) ** 2
                  - torch.log(sigma[None]) - 0.5 * math.log(2 * math.pi))   # [n,V,D]
        return torch.logsumexp(logphi + torch.log(w).view(1, -1, 1), dim=1)  # [n,D]

    def marginal_cdf(self, z, m=None, active=None):
        """Element-wise PIT u_i = Q_i(z_i) under the member's per-dim marginals. z [n,D] -> u [n,D]
        in (0,1). This is the model's own 'element-wise rank'; u has uniform marginals by construction."""
        w, mu, sigma = self._marginal_mixture(m, active)
        Phi = 0.5 * (1 + torch.erf((z[:, None, :] - mu[None]) / (sigma[None] * math.sqrt(2))))  # [n,V,D]
        return (w.view(1, -1, 1) * Phi).sum(dim=1)                    # [n,D]

    def copula_logprob(self, x, y, m=None, active=None):
        """Log copula density log p(u) = log q_J(z) - sum_i log q_i(z_i) at u = PIT(z), for the fitted
        member. Takes the ORIGINAL (pre-rank) halves (rank-transformed internally via forward), matching
        KL_joint_marginal. Returns (u [n,D], log_p_u [n])."""
        m = self.best_idx if m is None else m
        x, y = self.forward(x, y)
        z = torch.cat([x, y], dim=1)
        log_qj = self._member_logprobs(z, active=active)[m]           # [n]
        log_marg = self.marginal_logpdf(z, m, active).sum(dim=1)      # [n]
        u = self.marginal_cdf(z, m, active)
        return u, log_qj - log_marg


def _psd_cholesky(cov):
    """Lower Cholesky of each [..., s, s] covariance, made positive-definite robustly.

    Deliberately a MODULE-LEVEL function, not a method/@staticmethod: `%autoreload` mishandles
    changes to a method's descriptor and intermittently desyncs it from the body, throwing
    "takes 1 positional argument but 2 were given" / "missing 1 required positional argument".
    A plain module function has no class descriptor, so autoreload swaps it cleanly.

    A collapsed member's precision has huge eigenvalues, so (after inversion) its covariance
    sub-block is near-singular or ill-scaled and a fixed jitter over-/under-shoots. We escalate a
    jitter with both a relative and an absolute floor via the non-throwing ``cholesky_ex``; if that
    still fails (heavy-tailed bases, e.g. student-t), we fall back to a symmetric eigenvalue clamp,
    which is guaranteed PD. This never returns None -- returning None was the source of the
    ``scale_tril=None`` MultivariateNormal crash."""
    cov = torch.nan_to_num(cov)                          # guard against inf/nan from inversion
    eye = torch.eye(cov.size(-1), device=cov.device, dtype=cov.dtype)
    scale = cov.diagonal(dim1=-2, dim2=-1).mean(-1).clamp(min=1e-12)     # cov magnitude
    jitter = (1e-6 * scale).clamp(min=1e-8)              # absolute floor so tiny scale escalates
    for _ in range(10):
        chol, info = torch.linalg.cholesky_ex(cov + jitter[..., None, None] * eye)
        if int(info.max()) == 0:
            return chol
        jitter = jitter * 10
    # fallback: clamp the symmetric matrix's eigenvalues to a positive floor (guaranteed PD
    # when eigh converges)
    try:
        w, V = torch.linalg.eigh(cov)
        cov_pd = (V * w.clamp(min=1e-8)[..., None, :]) @ V.transpose(-1, -2)
        chol, info = torch.linalg.cholesky_ex(cov_pd + 1e-8 * eye)
        if int(info.max()) == 0:
            return chol
    except Exception:
        pass
    # last resort (eigh failed to converge on an extreme matrix): a diagonal factor from the
    # floored variances -- always a valid lower-triangular PD scale_tril
    diag = cov.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8)
    return torch.diag_embed(diag.sqrt())
