import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.distributions as distribution
import math
import numpy as np
import time
import optimizer
from nde import NAF, MAF, MDN
from copy import deepcopy


class VGC(nn.Sequential):
    """ 
        Vector Gaussian copula
    """
    def __init__(self, n_blocks, n_inputs, n_hidden, n_cond_inputs=2):
        super().__init__()
        self.maf1 = NAF(n_blocks, n_inputs, n_hidden, n_cond_inputs)
        self.maf2 = NAF(n_blocks, n_inputs, n_hidden, n_cond_inputs)
        self.base = MDN(n_in=2, n_hidden=10, n_out=2*n_inputs, K=1)
        self.max_iteration = 200
        self.lr = 1e-3
        self.bs = 250
        self.normal = None

    def forward(self, x, y):
        xx, _ = self.maf1.forward(x)
        yy, _ = self.maf2.forward(y)
        return xx, yy
    
    def sample(self, size):
        with torch.no_grad():
            z = self.normal.rsample(size)
            n, d = z.size()
            z_x, z_y = z[:, 0:d//2], z[:, d//2:]
            x, _ = self.maf1.forward(inputs=z_x, mode='inverse')
            y, _ = self.maf2.forward(inputs=z_y, mode='inverse')
            return torch.cat([x.clone().detach(), y.clone().detach()], dim=1)
    
    def log_prob(self, xy):
        n, d = xy.size()
        x, y = xy[:, 0:d//2], xy[:, d//2:]
        xx, log_jacob_xx = self.maf1.forward(x)
        yy, log_jacob_yy = self.maf2.forward(y)
        xxyy = torch.cat([xx, yy], dim=1)
        log_jacob = (log_jacob_xx + log_jacob_yy).view(n, -1)
        t = torch.ones(n, 2).to(xy.device)
        if self.normal is None:
            log_base_prob = self.base.log_probs(inputs=xxyy, cond_inputs=t).view(n, -1)          
        else:
            log_base_prob = self.normal.log_prob(xxyy).view(n, -1)
        return (log_base_prob + log_jacob).view(-1)
    
    def objective_func(self, x, y):
        log_probs_marginal = 0.5*self.maf1.log_probs(x).mean() + 0.5*self.maf2.log_probs(y).mean()
        xy = torch.cat([x, y], dim=1)
        log_probs_joint = self.log_prob(xy).mean()
        lamda = 1.0 
        return lamda*log_probs_joint + (1-lamda)*log_probs_marginal
    
    def learn(self, x, y):
        n, d = x.size()
        # [A]. pre-train f, g
        if self.maf1.max_iteration > 0:
            self.maf1.learn(x)
        if self.maf2.max_iteration > 0:
            self.maf2.learn(y)
        if self.max_iteration > 0:
            optimizer.NNOptimizer.learn(self, x=x, y=y)
        with torch.no_grad():
            xx, yy = self.forward(x, y)
        # [B]. learn the inner Gaussian 
        self.mu, self.V = self.empirical_params(xx, yy)
        self.mu2, self.V2 = self.mu.clone(), torch.eye(2*d).to(x.device)
        self.Vx, self.mx = self.V[0:d, 0:d], self.mu[0:d]
        self.Vy, self.my = self.V[d:, d:], self.mu[d:]
        self.normal = distribution.multivariate_normal.MultivariateNormal(self.mu, self.V)
        self.normal2 = distribution.multivariate_normal.MultivariateNormal(self.mu2, self.V2)
        self.normal_x = distribution.multivariate_normal.MultivariateNormal(self.mx, self.Vx)
        self.normal_y = distribution.multivariate_normal.MultivariateNormal(self.my, self.Vy)
        return 
    
    def empirical_params(self, x, y):
        z = torch.cat([x, y], dim=1)
        n, d = z.size()
        mu = z.mean(dim=0, keepdim=True)
        V = (z-mu).t() @ (z-mu)/(n+1)
        return mu.view(-1), V

    def MI(self, x, y, inner=True):
        if inner is False:
            x, y = self.forward(x, y)
        xy = torch.cat([x, y], dim=1)
        log_copula_density_xy = self.normal.log_prob(xy)
        log_copula_density_x = self.normal_x.log_prob(x)
        log_copula_density_y = self.normal_y.log_prob(y)       
        mi = log_copula_density_xy - log_copula_density_x - log_copula_density_y
        return mi.mean().item()
    
    def print(self):
        print('mu=', self.mu)
        print('V=',  (self.V*100).int()/100.0)


    #optimizer.NNOptimizer.learn(self, x=x, y=y)