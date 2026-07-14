import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import optimizer


class MINE(nn.Module):
    """MINE — Mutual Information Neural Estimator (Belghazi et al. 2018).

    Estimates MI via the Donsker-Varadhan lower bound with a neural critic (unclipped).
    """

    def __init__(self, architecture_critic, hyperparams):
        super().__init__()

        # hyperparameters
        self.max_iteration = 4000 if not hasattr(hyperparams, 'max_iteration') else hyperparams.max_iteration
        self.lr = 5e-4 if not hasattr(hyperparams, 'lr') else hyperparams.lr
        self.bs = 500 if not hasattr(hyperparams, 'bs') else hyperparams.bs
        self.wd = 1e-5 if not hasattr(hyperparams, 'wd') else hyperparams.wd
        self.n_neg = 2 if not hasattr(hyperparams, 'n_neg') else hyperparams.n_neg

        # layers
        self.critic_layer = CriticLayer(architecture_critic, 1)

    def MI(self, x, y):
        self.eval()
        with torch.no_grad():
            return self.objective_func(x, y).item()

    def log_ratio(self, x, y):
        zy = torch.cat([x, y], dim=1)
        return self.critic_layer(zy).view(-1)

    def objective_func(self, x, y):
        # classifier to classifier (x,y) ~ p(x,y) and (x,y) ~ p(x)p(y)
        m, d = x.size()
        idx_pos = []
        idx_neg = []
        n_neg = self.n_neg if self.training else min(m, 50)
        for i in range(n_neg):
            idx_pos = idx_pos + np.linspace(0, m-1, m).tolist()
            idx_neg = idx_neg + torch.randperm(m).cpu().numpy().tolist()

        zy_pos = torch.cat([x[idx_pos], y[idx_pos]], dim=1)
        zy_neg = torch.cat([x[idx_pos], y[idx_neg]], dim=1)
        f_pos = self.critic_layer(zy_pos)
        f_neg = self.critic_layer(zy_neg)
        mi = f_pos.mean() - (f_neg - np.log(len(f_neg))).logsumexp(dim=0)
        return mi

    def learn(self, x, y):
        return optimizer.NNOptimizer.learn(self, x, y)


class CriticLayer(nn.Module):
    '''
        Dense concat-shortcut MLP critic: maps a concatenated [x, y] to K scalar logits,
        with a skip connection from the raw input to the output layer.
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
