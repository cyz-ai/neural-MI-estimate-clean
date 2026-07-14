import torch
import torch.nn as nn
from copy import deepcopy

from nde import VGC


class MIENF(nn.Module):
    """MIENF — Mutual Information Estimation via Normalizing Flows.

    Fits a vector Gaussian copula (per-side flows + a joint Gaussian base) and reads MI off the
    latent copula density ratio.
    """
    def __init__(self, hyperparams):
        super().__init__()

        # default hyperparameters 
        self.bs = 500 if not hasattr(hyperparams, 'bs') else hyperparams.bs 
        self.lr = 5e-4 if not hasattr(hyperparams, 'lr') else hyperparams.lr
        self.wd = 0e-5 if not hasattr(hyperparams, 'wd') else hyperparams.wd
        self.encode_x = False if not hasattr(hyperparams, 'encode_x') else hyperparams.encode_x
        self.encode_y = False if not hasattr(hyperparams, 'encode_y') else hyperparams.encode_y
        self.max_iteration = 1500 if not hasattr(hyperparams, 'max_iteration') else hyperparams.max_iteration
        self.joint_learning = True if not hasattr(hyperparams, 'joint_learning') else hyperparams.joint_learning
        
        # layers
        print('MIENF (K=1), joint learning', self.joint_learning, '\n')
  
           
    def MI(self, x, y, mode='mc'):
        self.eval()
        with torch.no_grad():
            # ported VGC.MI reads the copula ratio off the joint MoG base;
            # inner=False first pushes (x, y) through the marginal flows.
            return self.gc.MI(x, y, inner=False)

    def learn(self, x, y):
        gc = self.learn_nde(x, y)
        self.gc = gc
        self.gc_state_dict = deepcopy(gc.state_dict())

    def learn_nde(self, x, y):
        n, d = x.size()

        # Neural density estimate. The ported VGC drives its own two learning
        # modes, which map onto MIENF's joint/two-stage options:
        #   joint    -> flows + base end-to-end (joint LL)
        #   separate -> pre-train the marginal flows, freeze them, then fit base
        mode = 'joint' if self.joint_learning else 'separate'
        base_iters = 2000 if self.joint_learning else 1000
        gc = VGC(n_blocks=2, n_inputs=d, n_hidden=250, n_cond_inputs=2,
                 K=1, learning_mode=mode, bs=250, max_iteration=base_iters)
        gc.to(x.device)
        if not self.joint_learning:
            # 'separate' mode already freezes the flows while fitting the base; just set how long
            # to pre-train each marginal flow.
            gc.maf1.max_iteration = 2000
            gc.maf2.max_iteration = 2000
        gc.learn(x, y)
        return gc

    