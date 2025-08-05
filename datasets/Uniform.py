import numpy as np
import torch
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal
from scipy.stats import norm, t
from torch.utils.data import Dataset

import torch.distributions as distribution


class Uniform(Dataset):
    """"""


    def __init__(self, n_samples=100000, n_dims=80, eps=0.5):
        """ """
        self.n_dims = n_dims
        self.eps = eps
        self.data = self.sample_data(100000)
          
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def sample_data(self, n_samples=10000):
        X, Y = sample_uniform(n_samples, self.n_dims, self.eps)
        return torch.Tensor(X), torch.Tensor(Y)

    def true_mutual_info(self):
        return compute_MI_given_eps_unif(self.eps)*self.n_dims







def sample_uniform(n, d, eps):
    x = np.random.uniform(0, 1, (n, d))
    n = np.random.uniform(-eps, eps, (n, d))
    y = x + n
    return x, y



def compute_MI_given_eps_unif(eps):
    if eps > 0.5:
        return 1/(4*eps)
    else:
        return eps - np.log(2*eps)
