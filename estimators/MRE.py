import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import optimizer


class MRE(nn.Module):
    """MRE — Multi-class (bridge) Density-Ratio Estimator.

    Estimates MI by classifying samples along a sequence of interpolating bridges between the
    joint p(x, y) and the product of marginals p(x)p(y).
    """
    def __init__(self, architecture_critic, hyperparams):
        super().__init__()

        # hyperparameters
        self.max_iteration = 4000 if not hasattr(hyperparams, 'max_iteration') else hyperparams.max_iteration
        self.lr = 5e-4 if not hasattr(hyperparams, 'lr') else hyperparams.lr
        self.bs = 500 if not hasattr(hyperparams, 'bs') else hyperparams.bs
        self.wd = 1e-5 if not hasattr(hyperparams, 'wd') else hyperparams.wd
        self.n_neg = 5 if not hasattr(hyperparams, 'n_neg') else hyperparams.n_neg
        self.K = 5 if not hasattr(hyperparams, 'n_bridges') else hyperparams.n_bridges   # number of bridges
        self.bridge_mode = 'intrapolation' if not hasattr(hyperparams, 'bridge_mode') else hyperparams.bridge_mode
        # 'mlp' (default) or 'resnet' critic; any non-'resnet' value uses the MLP critic
        self.critic = 'mlp' if not hasattr(hyperparams, 'critic') else hyperparams.critic

        # layers
        critic_cls = ResidualCriticLayer if self.critic == 'resnet' else CriticLayer
        self.critic_layer = critic_cls(architecture_critic, self.K+2)

        print('self.K', self.K, 'critic', self.critic)
        
    def encode(self, x):
        # s = s(x), get the summary statistic of x
        return x
    
    def encode2(self, y):
        # theta = h(y), get the representation of y
        return y
    
    def _generate_bridge_samples(self, samples_P, samples_Q, k):   
        # generate samples ~ P', Q', where KL(P', Q') < KL(P, Q)
        a1 = 1.0/(self.K+1)*k                                     # when k=0, P'=P; when k=K+1, P'=Q 
        n_P, n_Q = len(samples_P), len(samples_Q)
        if self.bridge_mode == 'mixture':
            idx_P1, idx_Q1 = torch.randperm(n_P)[:int(n_P*(1-a1))], torch.randperm(n_Q)[:int(n_Q*a1)]
            samples_new_P = torch.cat([samples_P[idx_P1]] + [samples_Q[idx_Q1]], dim=0)    
        if self.bridge_mode == 'intrapolation':
            samples_new_P = samples_P*(1-a1) + samples_Q*a1
        return samples_new_P
    
    def MI(self, x, y, mode='mc'):
        self.eval()
        with torch.no_grad(): 
            return self.log_ratio(x, y).mean().item() 
    
    def log_ratio(self, x, y, idx1=0, idx2=-1):
        z, y = self.encode(x), self.encode2(y)
        zy = torch.cat([z, y], dim=1)
        softmax_score = self.critic_layer(zy)
        lr = softmax_score[:, idx1] - softmax_score[:, idx2]
        return lr
    
    def objective_func(self, x, y):
        # compute representation of z, y
        z, y = self.encode(x), self.encode2(y)
        m, d = x.size()
        
        # construct samples P = p(x,y), Q = p(x)p(y)
        idx_pos = []
        idx_neg1 = []
        idx_neg2 = []
        for i in range(self.n_neg): 
            idx_pos = idx_pos + np.linspace(0, m-1, m).tolist()
            idx_neg1 = idx_neg1 + torch.randperm(m).cpu().numpy().tolist()
            idx_neg2 = idx_neg2 + torch.randperm(m).cpu().numpy().tolist()
        zy_P = torch.cat([z[idx_pos], y[idx_pos]], dim=1)
        zy_Q = torch.cat([z[idx_neg1], y[idx_neg2]], dim=1)

        # generate bridge + normal samples
        xy_array, label_array = [], []
        with torch.no_grad():
            for k in range(self.K+2):
                xy_k = self._generate_bridge_samples(zy_P, zy_Q, k)            # class 0 is P, class K+1 is Q
                label_k = torch.zeros(len(zy_P)).to(zy_P.device).long() + k
                xy_array.append(xy_k)
                label_array.append(label_k)
        xy_array, label_array = torch.cat(xy_array, dim=0), torch.cat(label_array, dim=0)

        # optimize softmax score
        softmax_score = self.critic_layer(xy_array)
        return -F.cross_entropy(softmax_score, label_array)
    
    def learn(self, x, y):
        return optimizer.NNOptimizer.learn(self, x, y)





class CriticLayer(nn.Module): 
    '''
        Critic layer
    '''
    def __init__(self, architecture, K):                  
        super().__init__()       
        in_dims = architecture[0]                                                                           
        dim_hidden = architecture[1]
        self.input = nn.Sequential(
            nn.Linear(in_dims, dim_hidden),
        )
        self.BN = False                                   # for critic used in InfoNCE & MINE, bn does not work well
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
        h = torch.cat([h, xy], dim=1)                     # dense net arch very important!
        out = self.out(h)
        return out


class ResidualBlock(nn.Module):
    '''
        Pre-activation residual MLP block:  x + W2(act(W1(act(x)))).
    '''
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.fc1(F.leaky_relu(x, 0.2))
        h = self.dropout(h)
        h = self.fc2(F.leaky_relu(h, 0.2))
        return x + h                                       # residual (identity) skip


class ResidualCriticLayer(nn.Module):
    '''
        ResNet critic: a stack of pre-activation residual blocks. Same I/O contract
        as CriticLayer -- maps concatenated [x, y] -> K class logits, with a final
        dense skip-concat to the raw input. Number of residual blocks is taken from
        the architecture depth (len(architecture) - 3), matching CriticLayer.
    '''
    def __init__(self, architecture, K, dropout=0.0):
        super().__init__()
        in_dims = architecture[0]
        dim_hidden = architecture[1]
        n_blocks = max(1, len(architecture) - 3)
        self.input = nn.Linear(in_dims, dim_hidden)
        self.blocks = nn.ModuleList([ResidualBlock(dim_hidden, dropout) for _ in range(n_blocks)])
        self.out = nn.Linear(in_dims + dim_hidden, K)

    def forward(self, xy):
        h = self.input(xy)
        for block in self.blocks:
            h = block(h)
        h = F.leaky_relu(h, 0.2)
        h = torch.cat([h, xy], dim=1)                     # dense skip to raw input (as in CriticLayer)
        out = self.out(h)
        return out