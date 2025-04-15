import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd as autograd
import numpy as np
import time 
from utils import utils_os


'''
    Estimating I(X; f(Y)) where f(Y) masks (1-p)% of the dimensions

    Ideal value: all should be <= I(X; Y)
'''
def base_test(X, Y, estimator_class, p_array=[0, 0.2, 0.4, 0.6, 0.8], device='cuda:0', name='N/A'):

    # normalize data
    X = _standardize(X)
    Y = _standardize(Y)

    # mask data & estimate MI
    mi_array = []
    for p in p_array:
        print('p is masked', p)

        XX = X.clone()
        YY = _mask(Y.clone(), p)

        estimator = _build_estimator(estimator_class, XX.size()[1]*2, device)
        estimator.learn(XX, YY)
        mi = estimator.MI(XX, YY)
        mi_array.append(mi)
    return mi_array





'''
    Estimating I(X'; Y) where X' is the randomly shuffled version of X

    Ideal value: I(X'; Y) = 0
'''
def shuffle_test(X, Y, estimator_class, device='cuda:0', name='N/A'):

    # normalize data
    X = _standardize(X)
    Y = _standardize(Y)

    # mask data & estimate MI
    print('shuffle test', 'ideal value:', '0')

    n, d = X.size()
    idx = torch.randperm(n)
    XX = X.clone()[idx]
    YY = Y.clone()

    estimator = _build_estimator(estimator_class, XX.size()[1]*2, device)
    estimator.learn(XX, YY)
    mi = estimator.MI(XX, YY)
    return [mi]




'''
    Estimating I([X, X*p]; Y) where X is masked with fraction p 

    Ideal value: all should be = I(X; Y)
'''
def dpi_test(X, Y, estimator_class, p_array=[0, 0.2, 0.4, 0.6, 0.8], device='cuda:0', name='N/A'):


    # normalize data
    X = _standardize(X)
    Y = _standardize(Y)

    # mask data & estimate MI
    mi_array = []
    for p in p_array:
        print('X is masked with ratio:', p, 'ideal value:', 'I(X; Y)')

        XX = torch.cat([X, _mask(X, p)], dim=1)           # XX = [X, X*mask[p]]
        YY = _pad(Y, XX.size()[1])                        # YY = [Y, noises]

        estimator = _build_estimator(estimator_class, XX.size()[1]*2, device)
        estimator.learn(XX, YY)
        mi = estimator.MI(XX, YY)
        mi_array.append(mi)
    return mi_array





'''
    Estimating I([X, X']; [Y, Y']) where X', Y' is shuffled version of X, Y

    Ideal value: all should be = 2*I(X; p*Y)
'''
def additive_test(X, Y, estimator_class, p_array=None, device='cuda:0', name='N/A'):

    # normalize data
    X = _standardize(X)
    Y = _standardize(Y)

    # create dummy data
    idx = torch.randperm(len(X))
    X2 = X.clone()[idx]
    Y2 = Y.clone()[idx]

    # mask data & estimate MI
    for p in [1]:
        print('additive test', 'ideal value:', '2*I(X; Y)')

        XX = torch.cat([X, X2], dim=1)
        YY = torch.cat([Y, Y2], dim=1)

        estimator = _build_estimator(estimator_class, XX.size()[1]*2, device)
        estimator.learn(XX, YY)
        mi = estimator.MI(XX, YY)

    return [mi]






# --------------------------------- Utility functions --------------------------------- #



'''
    mask p% of the data
'''
def _mask(X, p):
    noise = torch.randn_like(X)
    XX = X.clone()
    n, d = X.size()
    d_to_mask = int(d*p)
    XX[:, d-d_to_mask:] = noise[:, d-d_to_mask:]
    return XX

'''
    pad data with noises
'''
def _pad(X, d_target):
    n, dx = X.size()
    noise = torch.randn(n, d_target-dx).to(X.device)
    return torch.cat([X, noise], dim=1)



'''
    standardizing data to have zero mean and unit std
'''
def _standardize(X):
    (X - X.mean(dim=1, keepdim=True))/X.std(dim=1, keepdim=True)
    return X



'''
    build estimator according to the specific estimator class
'''
def _build_estimator(estimator_class, dim, device):

    # Hyperparams of MI estimators
    class Hyperparams(object):
        def __init__(self): 
            self.critic = 'neural'                
            self.lr = 5e-4
            self.bs = 500
            self.wd = 1e-5
            self.dim = 100
            self.n_bridges = 4
            self.early_stop = True
            self.t_patience = 500
            self.importance_sampling = True
            self.max_iteration = 2000
            self.device = device

    hyperparams=Hyperparams()
    hyperparams.dim = dim//2

    architecture_critic = [dim, 500, 500, 500, 1]

    return estimator_class(None, None, architecture_critic, hyperparams).to(device)