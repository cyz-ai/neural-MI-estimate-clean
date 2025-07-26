import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd as autograd
import numpy as np
import scipy
import math
import time
import optimizer
import random 
import estimators.layers as layers


class fDIME(nn.Module):
    """ 
        f-divergence + data dearrangement
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
        main_layer = CriticLayer(architecture_critic, 1, hyperparams)
        self.critic_layer = CombinedNet(main_layer, divergence='GAN')
        
            
    def encode(self, x):
        # s = s(x), get the representation of x
        return self.encode_layer(x) if self.encode_x else x
    
    def encode2(self, y):
        # theta = h(y), get the representation of y
        return self.encode2_layer(y) if self.encode_y else y
    
    def MI(self, x, y):
        self.eval()
        with torch.no_grad():
            return self.log_ratio(x, y).mean().item()

    def log_ratio(self, x, y):
        data_u, data_v = x, y
        data_uv, data_u_v = data_generation_mi(data_u, data_v, device=self.device)
        D_value_1, D_value_2 = self.critic_layer(data_uv, data_u_v)
        loss, R = gan_fdime_deranged(D_value_1, D_value_2, x.device)
        return torch.log(R)
    
    def objective_func(self, x, y):
        # classifier to classifier (x,y) ~ p(x,y) and (x,y) ~ p(x)p(y)
        m, d = x.size()
  
        data_u, data_v = x, y
        data_uv, data_u_v = data_generation_mi(data_u, data_v, device=self.device)
        D_value_1, D_value_2 = self.critic_layer(data_uv, data_u_v)
        loss, R = gan_fdime_deranged(D_value_1, D_value_2, x.device)

        return -loss
 
    def learn(self, x, y):
        return optimizer.NNOptimizer.learn(self, x, y)
    
    
    
    
    
    
    
    
class CombinedNet(nn.Module):
    def __init__(self, single_architecture, divergence):
        super(CombinedNet, self).__init__()
        self.div_to_act_func = {
            "GAN": nn.Sigmoid(),
            "KL": nn.Softplus(),
            "RKL": nn.Softplus(),
            "HD": nn.Softplus(),
            "MINE": nn.Identity(),
            "GAN_DIME": nn.Sigmoid(),
            "SL": nn.Sigmoid(),
            "SMILE": nn.Identity(),
            "NWJ": nn.Identity()
        }
        self.divergence = divergence
        self.single_architecture = single_architecture
        self.final_activation = self.div_to_act_func[divergence]

    def forward(self, input_tensor_1, input_tensor_2):
        intermediate_1 = self.single_architecture(input_tensor_1)
        output_tensor_1 = self.final_activation(intermediate_1)
        intermediate_2 = self.single_architecture(input_tensor_2)
        output_tensor_2 = self.final_activation(intermediate_2)

        return output_tensor_1, output_tensor_2
    
    
    
def data_generation_mi(data_x, data_y, device="cpu"):
    """
    Generates samples of the product of marginal distributions, given the samples from the joint distribution.
    """
    der = True
    data_xy = torch.cat([data_x, data_y], dim=1)
    if der:  # Derangement
        data_y_shuffle = torch.index_select(data_y, 0, derangement(list(range(data_y.shape[0])), device))
        #ordered_derangement = [(idx + 1) % data_y.shape[0] for idx in range(data_y.shape[0])]
        #data_y_shuffle = torch.index_select(data_y, 0, torch.Tensor(ordered_derangement).int().to(device))
    else:  # Permutation
        data_y_shuffle = torch.index_select(data_y, 0, torch.tensor(np.random.permutation(data_y.shape[0])).int().to(device))

    data_x_y = torch.cat([data_x, data_y_shuffle], dim=1)
    return data_xy, data_x_y


def gan_fdime_deranged(D_value_1, D_value_2, device="cpu"):
    """GAN cost function"""
    BCE = nn.BCELoss()
    batch_size_1 = D_value_1.size(0)
    batch_size_2 = D_value_2.size(0)
    valid_2 = torch.ones((batch_size_2, 1), device=device)
    fake_1 = torch.zeros((batch_size_1, 1), device=device)
    loss_1 = BCE(D_value_1, fake_1)
    loss_2 = BCE(D_value_2, valid_2)
    loss = loss_1 + loss_2
    R = (1 - D_value_1) / D_value_1
    return loss, R


def derangement(l, device):
    """Random derangement"""
    o = l[:]
    while any(x == y for x, y in zip(o, l)):
        random.shuffle(l)
    return torch.Tensor(l).long().to(device)