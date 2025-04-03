import numpy as np
import torch
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal
from scipy.stats import norm, t
from torch.utils.data import Dataset

import torch.distributions as distribution


class NonlinearGaussian(Dataset):
    """"""

    def __init__(self, n_samples=100000, n_dims=80, rho=0.80, mu=0, case=0):
        """ """
        self.case = case
        self.n_dims = n_dims
        self.mu = np.zeros(self.n_dims)+mu
        self.rho = rho
        self.rhos = np.ones(n_dims // 2) * self.rho
        self.cov_matrix = block_diag(*[[[1, self.rho], [self.rho, 1]] for _ in range(n_dims // 2)])
        self.data = self._sample_gaussian(n_samples, self.cov_matrix).astype(dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _sample_gaussian(self, n_samples, cov_matrix):
        mvn = multivariate_normal(mean=np.zeros(self.n_dims)+self.mu, cov=cov_matrix)
        return mvn.rvs(n_samples)

    @staticmethod
    def _get_rho_from_mi(mi, n_dims):   # analytically calculate correlation coefficient from MI value
        x = (4 * mi) / n_dims
        return (1 - np.exp(-x)) ** 0.5  
    
    @staticmethod
    def _get_mi_from_rho(rho, n_dims):  # analytically calculate mutual information from correlation value
        a = np.log(1 - rho**2)
        return -1/4.0*n_dims*a   
    
    @staticmethod
    def u2xy(u):
        X, Y = u[:, ::2], u[:, 1::2]
        return X, Y
    
    @staticmethod
    def xy2u(X, Y):
        n, d = X.shape
        samples = np.zeros((len(X), d*2))
        samples[:, ::2], samples[:, 1::2] = X, Y
        return samples      
    
    def _numerator_log_prob(self, u):
        mvn = multivariate_normal(mean=np.zeros(self.n_dims)+self.mu, cov=self.cov_matrix)
        return mvn.logpdf(u)
    
    def _denominator_log_prob(self, u):
        mvn = multivariate_normal(mean=np.zeros(self.n_dims)+self.mu, cov=np.diag(np.ones(self.n_dims)))
        return mvn.logpdf(u)

    def sample_data(self, n_samples, mode='joint'):
        cov = self.cov_matrix if mode=='joint' else np.diag(np.ones(self.n_dims))
        data = self._sample_gaussian(n_samples, cov)
        X, Y = torch.Tensor(data[:, ::2]).clone(), torch.Tensor(data[:, 1::2]).clone()
        return X, Y
        
    def log_ratio(self, X, Y):                    # this return log p(x, y)/p(x)p(y)  
        samples = np.zeros((len(X), self.n_dims))
        samples[:, ::2], samples[:, 1::2] = X, Y
        return self._numerator_log_prob(samples) - self._denominator_log_prob(samples)
    
    def true_mutual_info(self):
        return self._get_mi_from_rho(self.rho, self.n_dims)

    def empirical_mutual_info(self):
        samples = self._sample_gaussian(100000, self.cov_matrix)
        return np.mean(self._numerator_log_prob(samples) - self._denominator_log_prob(samples))

    def transformation(self, x, y):       
        '''
            transform the data to make the task more difficult
        '''
        standardize = lambda z: (z-z.mean(dim=0, keepdim=True))/(x.std(dim=0, keepdim=True)+0e-8)
        case = self.case
        if case == '0':                                            # x, y = x, y
            x, y = x, y
        if case == '1a':                                           # x, y = tanh(x), e^y
            x, y = torch.tanh(x), torch.exp(y)      
        if case == '1b':                                           # x, y = x^3, e^y
            x, y = x**3, y**3
        if case == '1c':                                           # x, y = sign(x)x^2, sign(y)y^2
            n, d = x.size()
            x, y = torch.sign(x)*x**2, torch.sign(y)*y**2        
        if case == '1d':                                           # x, y = sign(x)x^2, sign(y)y^2
            n, d = x.size()
            x, y = 3*x, y/2   
        if case == '2':                                            # x, y = Ax, By
            n, d = x.size()
            A = torch.ones(d, d).tril()                           
            B = torch.ones(d, d).tril()
            A, B = A/A.sum(dim=1, keepdim=True), B/B.sum(dim=1, keepdim=True)
            self.A, self.B = A, B
            x, y = torch.matmul(x, A), torch.matmul(y, B)  
        if case == '3a':                                           # x, y = Atanh(x), Be^y
            n, d = x.size()
            x, y = torch.tanh(x), torch.exp(y)
            A = torch.ones(d, d).tril()                           
            B = torch.ones(d, d).tril()
            A, B = A/A.sum(dim=1, keepdim=True), B/B.sum(dim=1, keepdim=True)
            self.A, self.B = A, B
            x, y = torch.matmul(x, A), torch.matmul(y, B)  
        if case == '3b':                                           # x, y = Ax^3, Be^y
            n, d = x.size()
            x, y = x**3, torch.exp(y)
            A = torch.ones(d, d).tril()                           
            B = torch.ones(d, d).tril()
            #A, B = A/A.sum(dim=1, keepdim=True), B/B.sum(dim=1, keepdim=True)
            self.A, self.B = A, B
            x, y = torch.matmul(x, A), torch.matmul(y, B)  
        if case == '3c':                                           # x, y = Asign(x)x^2, Bsign(y)y^2
            n, d = x.size()
            x, y = torch.sign(x)*x**2, torch.sign(y)*y**2
            A = torch.ones(d, d).tril()                             
            B = torch.ones(d, d).tril()
            self.A, self.B = A, B
            x, y = torch.matmul(x, A), torch.matmul(y, B)  
        if case == '3d':                                           # x, y = student-t, student-t
            n, d = x.size()
            v = 3
            gaussian_cdf_x, gaussian_cdf_y = norm.cdf(x.cpu().numpy()), norm.cdf(y.cpu().numpy())
            t_samples_x, t_samples_y = t.ppf(gaussian_cdf_x, df=v), t.ppf(gaussian_cdf_y, df=v)
            x, y = torch.Tensor(t_samples_x).float(), torch.Tensor(t_samples_y).float()
            A = torch.ones(d, d).tril()                             
            B = torch.ones(d, d).tril()
            self.A, self.B = A, B
            x, y = torch.matmul(x, A), torch.matmul(y, B)  
        if case == '3e':                                           # x, y = tanh(x), exp(y^3)
            n, d = x.size()
            n, d = x.size()
            x, y = torch.tanh(x), 0.5*y**3 + 0.5*torch.exp(y)
            A = torch.ones(d, d).tril()                             
            B = torch.ones(d, d).tril()
            self.A, self.B = A, B
            x, y = torch.matmul(x, A), torch.matmul(y, B)  
        self.mu_x, self.mu_y = x.mean(dim=0, keepdim=True), y.mean(dim=0, keepdim=True)
        self.std_x, self.std_y = x.std(dim=0, keepdim=True), y.std(dim=0, keepdim=True)
        return x, y                                                 #return standardize(x), standardize(y)

    def log_prob(self, X, Y):
        case = self.case
        if case == '0':
            eps_x, eps_y = X, Y
            log_dxde, log_dyde = (0*X).sum(dim=1), (0*Y).sum(dim=1)
        if case == '1a':
            eps_x, eps_y = torch.atanh(X), torch.log(Y)
            log_dxde, log_dyde = torch.log(1-torch.tanh(eps_x)**2).sum(dim=1), eps_y.sum(dim=1)
        if case == '1b':
            eps_x, eps_y = torch.sign(X)*X.abs().pow(0.33333), torch.sign(Y)*Y.abs().pow(0.33333)
            log_dxde, log_dyde = torch.log(3*eps_x.abs().pow(2)).sum(dim=1), torch.log(3*eps_y.abs().pow(2)).sum(dim=1)
        if case == '1c':
            eps_x, eps_y = torch.sign(X)*X.abs().sqrt(), torch.sign(Y)*Y.abs().sqrt()
            log_dxde, log_dyde = torch.log(2*eps_x.abs()).sum(dim=1), torch.log(2*eps_y.abs()).sum(dim=1)
        if case == '1d':
            eps_x, eps_y = X/3, Y*2
            log_dxde, log_dyde = (0*X + np.log(3)).sum(dim=1), (0*Y + np.log(0.5)).sum(dim=1) 
        if case == '2':
            eps_x, eps_y = X.cpu()@self.A.inverse(), Y.cpu()@self.B.inverse()
            log_dxde, log_dyde = self.A.logdet().repeat(len(X)), self.B.logdet().repeat(len(X)) 
        if case == '3a':
            log_dxde1, log_dyde1 = self.A.logdet().repeat(len(X)), self.B.logdet().repeat(len(X)) 
            x, y = X.cpu()@self.A.inverse(), Y.cpu()@self.B.inverse()
            eps_x, eps_y = torch.atanh(x), torch.log(y)
            log_dxde2, log_dyde2 = torch.log(1-torch.tanh(eps_x)**2).sum(dim=1), eps_y.sum(dim=1)
            log_dxde, log_dyde = log_dxde1+log_dxde2, log_dyde1+log_dyde2
        if case == '3b':
            x, y = X.cpu()@self.A.inverse(), Y.cpu()@self.B.inverse()
            eps_x, eps_y = torch.sign(x)*x.abs().pow(0.33333), torch.log(y)
            log_dxde, log_dyde = torch.log(3*eps_x.abs().pow(2)).sum(dim=1), eps_y.sum(dim=1)
        if case == '3c':
            x, y = X.cpu()@self.A.inverse(), Y.cpu()@self.B.inverse()
            eps_x, eps_y = torch.sign(x)*x.abs().sqrt(), torch.sign(y)*y.abs().sqrt()
            log_dxde, log_dyde = torch.log(2*eps_x.abs()).sum(dim=1), torch.log(2*eps_y.abs()).sum(dim=1)
        u = self.xy2u(eps_x.cpu().numpy(), eps_y.cpu().numpy())
        log_dxy_du = log_dxde + log_dyde
        log_pu = self._numerator_log_prob(u)
        return log_pu - log_dxy_du.cpu().numpy()
        
