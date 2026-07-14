import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import optimizer


class InfoNCE(nn.Module):
    """InfoNCE — contrastive MI lower bound (van den Oord et al. 2018 / CPC).

    Reads MI = log(m) + the in-batch contrastive log-softmax at the positive pair, so the bound
    saturates at log(m) (the contrastive-set size: batch at train time, full data at eval).
    """
    MI_CLAMP = 50.0         # final scalar MI clipped to [-MI_CLAMP, MI_CLAMP]

    def __init__(self, architecture_critic, hyperparams):
        super().__init__()

        # hyperparameters
        self.max_iteration = 4000 if not hasattr(hyperparams, 'max_iteration') else hyperparams.max_iteration
        self.lr = 5e-4 if not hasattr(hyperparams, 'lr') else hyperparams.lr
        self.bs = 500 if not hasattr(hyperparams, 'bs') else hyperparams.bs   # in-batch contrastive-set size; MI ceiling = log(bs)
        self.wd = 1e-5 if not hasattr(hyperparams, 'wd') else hyperparams.wd

        # layers
        self.critic_layer = CriticLayer(architecture_critic, 1)

    def MI(self, x, y):
        self.eval()
        with torch.no_grad():
            v = self.objective_func(x, y).item()
            if not math.isfinite(v):
                return 0.0
            return float(max(-self.MI_CLAMP, min(self.MI_CLAMP, v)))

    def log_ratio(self, x, y):
        zy = torch.cat([x, y], dim=1)
        t = self.critic_layer(zy)
        return t.view(-1)

    def objective_func(self, x, y):
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
