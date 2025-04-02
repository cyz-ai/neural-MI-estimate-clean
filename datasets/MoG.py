import numpy as np
import torch
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from torch.utils.data import Dataset

from datasets.NonlinearGaussian import NonlinearGaussian



class MoG(Dataset):
    """"""

    # def __init__(self, n_samples=100000, n_dims=80, K=5, rhos=np.random.rand(1, 100)):
    #     """ """
    #     self.n_dims = n_dims
    #     self.K = K
    #     self.gaussians = [NonlinearGaussian(n_samples, n_dims, rhos[i]) for i in range(K)]
    #     self.data = self.sample_data(n_samples)

    def __init__(self, n_samples=100000, n_dims=80, K=5, shifts=np.random.rand(1, 100)*0, rhos=np.random.rand(1, 100)):
        """ """
        self.n_dims = n_dims
        self.K = K
        self.gaussians = [NonlinearGaussian(n_samples, n_dims, rhos[i], shifts[i]) for i in range(K)]
        self.gaussians_x = [NonlinearGaussian(n_samples, n_dims//2, 0, shifts[i]) for i in range(K)]
        self.gaussians_y = [NonlinearGaussian(n_samples, n_dims//2, 0, shifts[i]) for i in range(K)]
        self.data = self.sample_data(n_samples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def sample_data(self, n_samples):
        X, Y = [], []
        for i in range(self.K):
            X_i, Y_i = self.gaussians[i].sample_data(n_samples//self.K)
            X.append(X_i)
            Y.append(Y_i)
        X, Y = torch.cat(X, dim=0), torch.cat(Y, dim=0)
        idx = torch.randperm(n_samples)
        return X[idx], Y[idx]
    
    def _numerator_log_prob(self, u):
        # the joint is a MoG, need summing logprob
        log_prob = np.zeros((len(u), self.K))
        for i in range(self.K):
            log_prob[:, i] = self.gaussians[i]._numerator_log_prob(u)
        return logsumexp(log_prob, 1) - np.log(self.K)
        
    def _denominator_log_prob(self, u):
        # compute log p(x) and log p(y) respectively
        log_prob_x = np.zeros((len(u), self.K))
        log_prob_y = np.zeros((len(u), self.K))
        x, y = NonlinearGaussian.u2xy(u)
        for i in range(self. K):
            log_prob_x[:, i] = self.gaussians_x[i]._numerator_log_prob(x)
            log_prob_y[:, i] = self.gaussians_y[i]._numerator_log_prob(y)
        return logsumexp(log_prob_x, 1) - np.log(self.K) + logsumexp(log_prob_y, 1) - np.log(self.K)

    def log_ratio(self, X, Y):                    # this return log p(x, y)/p(x)p(y)  
        samples = np.zeros((len(X), self.n_dims))
        samples[:, ::2], samples[:, 1::2] = X, Y
        return self._numerator_log_prob(samples) - self._denominator_log_prob(samples)
    
    def log_prob(self, X, Y):
        samples = np.zeros((len(X), self.n_dims))
        samples[:, ::2], samples[:, 1::2] = X.cpu().numpy(), Y.cpu().numpy()
        return self._numerator_log_prob(samples)

    def true_mutual_info(self):
        return "not implemented true MI"
       
    def empirical_mutual_info(self):
        X, Y = self.sample_data(100000)
        samples = np.zeros((len(X), self.n_dims))
        samples[:, ::2], samples[:, 1::2] = X, Y
        return np.mean(self._numerator_log_prob(samples) - self._denominator_log_prob(samples))
