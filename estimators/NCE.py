import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd as autograd
import numpy as np
import scipy
import math
import time
import optimizer
import estimators.layers as layers
from nde.MAF import MAF


class NCE(nn.Module):
    """ 
        Noise contrastive estimation
    """
    def __init__(self, architecture_encoder_x, architecture_encoder_y, architecture_critic, hyperparams):
        super().__init__()
        
        # hyperparameters
        self.estimator = 'NCE' if not hasattr(hyperparams, 'estimator') else hyperparams.estimator  
        self.bs = 250 if not hasattr(hyperparams, 'bs') else hyperparams.bs 
        self.lr = 5e-4 if not hasattr(hyperparams, 'lr') else hyperparams.lr
        self.wd = 0e-5 if not hasattr(hyperparams, 'wd') else hyperparams.wd
        self.n_neg = 4 if not hasattr(hyperparams, 'n_neg') else hyperparams.n_neg
        self.encode_x = False if not hasattr(hyperparams, 'encode_x') else hyperparams.encode_x
        self.encode_y = False if not hasattr(hyperparams, 'encode_y') else hyperparams.encode_y
        self.critic = 'neural' if not hasattr(hyperparams, 'critic') else hyperparams.critic
        self.max_iteration = 1500 if not hasattr(hyperparams, 'max_iteration') else hyperparams.max_iteration
        
        CriticLayer = layers.QuadraticCriticLayer if hyperparams.critic == 'quadratic' else layers.NeuralCriticLayer
        
        # layers
        self.encode_layer = layers.EncodeLayer(architecture_encoder_x, hyperparams)
        self.encode2_layer = layers.EncodeLayer(architecture_encoder_y, hyperparams)
        self.critic_layer = CriticLayer(architecture_critic, 1, hyperparams)
            
    def encode(self, x):
        # s = s(x), get the representation of x
        return self.encode_layer(x) if self.encode_x else x
    
    def encode2(self, y):
        # theta = h(y), get the representation of y
        return self.encode2_layer(y) if self.encode_y else y
    
    def MI(self, x, y):
        self.eval()
        with torch.no_grad():
            return self.log_ratio(x, y).mean()

    def log_ratio(self, x, y):
        z, y = self.encode(x), self.encode2(y)
        zy =  torch.cat([z, y], dim=1)
        t = self.critic_layer(zy)
        return t.view(-1)
    
    def objective_func(self, x, y):
        # classifier to classifier (x,y) ~ p(x,y) and (x,y) ~ p(x)p(y)
        m, d = x.size()
        z, y = self.encode(x), self.encode2(y)
        idx_pos = []
        idx_neg = []
        for i in range(self.n_neg): 
            idx_pos = idx_pos + np.linspace(0, m-1, m).tolist()
            idx_neg = idx_neg + torch.randperm(m).cpu().numpy().tolist()
            
        zy_pos = torch.cat([z[idx_pos], y[idx_pos]], dim=1)
        zy_neg = torch.cat([z[idx_pos], y[idx_neg]], dim=1)
        f_pos = self.critic_layer(zy_pos)
        f_neg = self.critic_layer(zy_neg)
        A, B = -F.softplus(-f_pos), -F.softplus(f_neg)
        mi = A.mean() + B.mean()
        return mi
 
    def learn(self, x, y):
        return optimizer.NNOptimizer.learn(self, x, y)