import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import optimizer


class InfoNCE(nn.Module):
    """InfoNCE — contrastive MI lower bound (van den Oord et al. 2018 / CPC).

    Reads MI = log(M) + the contrastive log-softmax at the positive pair, so the bound saturates
    at log(M) where M is the contrastive-set size.

    Two regimes, selected by ``K_negatives``:
      * ``K_negatives=None`` (default) — **full in-batch** softmax: every other sample is a
        negative, M = m (the batch), cost **O(m^2)** critic evaluations per batch.
      * ``K_negatives=K`` — **sampled softmax**: each anchor is scored against K random negatives,
        M = K+1, cost **O(m*K)**. Trades the MI ceiling (log(K+1)) for speed; K >= m-1 is
        equivalent to the full path. Set K so that log(K+1) comfortably exceeds the expected MI.
    """

    def __init__(self, architecture_critic, hyperparams):
        super().__init__()

        # hyperparameters
        self.max_iteration = 4000 if not hasattr(hyperparams, 'max_iteration') else hyperparams.max_iteration
        self.lr = 5e-4 if not hasattr(hyperparams, 'lr') else hyperparams.lr
        self.bs = 500 if not hasattr(hyperparams, 'bs') else hyperparams.bs   # in-batch contrastive-set size; MI ceiling = log(bs)
        self.wd = 1e-5 if not hasattr(hyperparams, 'wd') else hyperparams.wd
        # None -> full O(m^2) in-batch softmax; int K -> O(m*K) sampled softmax (ceiling log(K+1))
        self.K_negatives = getattr(hyperparams, 'K_negatives', None)
        # eval-time contrastive-set size; None -> full data (tightest read), else sampled with K_eval
        self.K_eval = getattr(hyperparams, 'K_eval', None)

        # layers
        self.critic_layer = CriticLayer(architecture_critic, 1)

    def MI(self, x, y):
        self.eval()
        with torch.no_grad():
            v = self._objective(x, y, self.K_eval).item()   # eval reads the (tight) full-set bound by default
            if not math.isfinite(v):
                return 0.0
            return float(v)

    def log_ratio(self, x, y):
        zy = torch.cat([x, y], dim=1)
        t = self.critic_layer(zy)
        return t.view(-1)

    def objective_func(self, x, y):
        # training / early-stopping objective; uses the sampled path iff K_negatives is set
        return self._objective(x, y, self.K_negatives)

    def _objective(self, x, y, K):
        # Dispatch: full in-batch (O(m^2)) when K is None or as large as the batch, else the
        # O(m*K) sampled-softmax path. Both estimate MI = log(M) + E[f(pos) - logsumexp(candidates)].
        m = x.size(0)
        if K is None or K >= m - 1:
            return self._objective_full(x, y)
        return self._objective_sampled(x, y, int(K))

    def _objective_full(self, x, y):
        # In-batch InfoNCE (van den Oord et al. 2018 / CPC): every OTHER sample in the batch
        # is a negative, so for anchor i the contrastive set is {(x_i, y_j) : j in batch} with
        # the positive at j=i. The contrastive set size — and the MI ceiling = log(m) — is thus
        # the batch size m (batch at train time, the full dataset at eval time).
        #   MI = log(m) + mean_i [ f(x_i, y_i) - logsumexp_j f(x_i, y_j) ]
        # Vectorized, chunked over anchors so the m x m score matrix never fully materializes.
        m, d = x.size()
        CH = 256                                      # anchor block size (memory cap)
        acc = x.new_zeros(())
        pos_cols = torch.arange(m, device=x.device)
        for s in range(0, m, CH):
            b = min(CH, m - s)
            x_rep = x[s:s + b].unsqueeze(1).expand(b, m, d).reshape(-1, d)
            y_rep = y.unsqueeze(0).expand(b, m, d).reshape(-1, d)
            scores = self.critic_layer(torch.cat([x_rep, y_rep], dim=1)).view(b, m)   # [b, m]
            diag = scores[torch.arange(b, device=x.device), pos_cols[s:s + b]]
            acc = acc + (diag - scores.logsumexp(dim=1)).sum()
        return math.log(m) + acc / m

    def _objective_sampled(self, x, y, K):
        # Sampled-softmax InfoNCE: for anchor i the contrastive set is {y_i} + K random negatives
        # drawn from the batch, so M = K+1 and the ceiling is log(K+1). Cost is O(m*K) critic evals.
        #   MI = log(K+1) + mean_i [ f(x_i, y_i) - logsumexp_{c in {pos}+negs} f(x_i, y_c) ]
        # Negatives are index-offset by 1..m-1 (mod m) so a sampled negative is never the positive.
        m, d = x.size()
        ar = torch.arange(m, device=x.device)
        offsets = torch.randint(1, m, (m, K), device=x.device)      # 1..m-1 -> guaranteed != anchor
        neg_idx = (ar.unsqueeze(1) + offsets) % m                   # [m, K]
        CH = 256                                                    # anchor block size (memory cap)
        acc = x.new_zeros(())
        for s in range(0, m, CH):
            b = min(CH, m - s)
            xb = x[s:s + b]                                         # [b, d]
            cand = torch.cat([y[s:s + b].unsqueeze(1),             # positive at column 0
                              y[neg_idx[s:s + b]]], dim=1)          # [b, K+1, d]
            x_rep = xb.unsqueeze(1).expand(b, K + 1, d)            # [b, K+1, d]
            scores = self.critic_layer(
                torch.cat([x_rep, cand], dim=2).reshape(b * (K + 1), -1)).view(b, K + 1)
            acc = acc + (scores[:, 0] - scores.logsumexp(dim=1)).sum()
        return math.log(K + 1) + acc / m

    def learn(self, x, y):
        return optimizer.NNOptimizer.learn(self, x, y)


class CriticLayer(nn.Module):
    '''
        Dense concat-shortcut MLP critic (same design as MINE): maps a concatenated [x, y]
        to K scalar logits, with a skip connection from the raw input to the output layer.
    '''
    def __init__(self, architecture, K):
        super().__init__()
        in_dims = architecture[0]
        dim_hidden = architecture[1]
        self.input = nn.Sequential(
            nn.Linear(in_dims, dim_hidden),
        )
        self.BN = False                                   # BN does not work well for this critic
        self.bn1 = nn.BatchNorm1d(dim_hidden)
        self.bn2 = nn.BatchNorm1d(dim_hidden)
        self.main = nn.Sequential(*[nn.Linear(dim_hidden, dim_hidden) for _ in range(len(architecture)-3)])
        self.out = nn.Linear(in_dims + dim_hidden, K)
        self.dropout = nn.Dropout(0.25)

    def forward(self, xy):
        h = self.input(xy)
        h = self.bn1(h) if self.BN else h
        for i, layer in enumerate(self.main):
            h = layer(F.leaky_relu(h, 0.2))
        h = self.bn2(h) if self.BN else h
        h = F.leaky_relu(h, 0.2)
        h = torch.cat([h, xy], dim=1)                     # dense skip to raw input
        out = self.out(h)
        return out
