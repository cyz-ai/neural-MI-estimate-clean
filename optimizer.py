import torch
import torch.nn as nn
import math
from copy import deepcopy



class NNOptimizer(nn.Module):
    
    @staticmethod 
    def divide_train_val(x, y, ratio=0.80):
        n = len(x)
        n_train = int(ratio*n)
        x_train, y_train = x[0:n_train], y[0:n_train]
        x_val, y_val = x[n_train:n], y[n_train:n]
        return  x_train, y_train, x_val, y_val
    
    @staticmethod
    def learn(net, x, y, shuffle=False, early_stop=True):
        # shuffle data
        if shuffle:
            idx = torch.randperm(len(x))
            x, y = x[idx].clone().detach(), y[idx].clone().detach()

        # hyperparams
        T = 2000 if not hasattr(net, 'max_iteration') else net.max_iteration
        bs = 200 if not hasattr(net, 'bs') else net.bs
        lr = 5e-4 if not hasattr(net, 'lr') else net.lr
        wd = 0e-5 if not hasattr(net, 'wd') else net.wd
        PRINTING = True if not hasattr(net, 'trace_learning') else net.trace_learning
        T_NO_IMPROVE_THRESHOLD = 200 if not hasattr(net, 't_patience') else net.t_patience

        # divide train & val (centralized 80/20)
        n = len(x)
        x_train, y_train, x_val, y_val = NNOptimizer.divide_train_val(x, y)
        net.device = x.device
        
        # learn in loops
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=wd)
        n_batch, n_val_batch = (int(len(x_train)/bs) if len(x_train) > bs else 1), int(len(x_val)/1000) if len(x_val) > 1000 else 1
        best_val_loss, best_model_state_dict, best_t, no_improvement = math.inf, None, 0, 0
                
        for t in range(T):
            # shuffle the batch
            idx = torch.randperm(len(x_train)) 
            x_train, y_train = x_train[idx], y_train[idx]
            x_chunks, y_chunks = torch.chunk(x_train, n_batch), torch.chunk(y_train, n_batch)
            x_v_chunks, y_v_chunks = torch.chunk(x_val, n_val_batch), torch.chunk(y_val, n_val_batch)

            # gradient descend
            net.train()
            for i in range(len(x_chunks)):
                optimizer.zero_grad()
                loss = -net.objective_func(x_chunks[i], y_chunks[i])
                if t>0:
                    loss.backward()
                    optimizer.step()
              
            # early stopping if val loss does not improve after some epochs
            net.eval()
            loss_val = torch.zeros(1, device=x.device)
            with torch.no_grad():
                for j in range(len(x_v_chunks)):
                    loss_val += -net.objective_func(x_v_chunks[j], y_v_chunks[j])/len(x_v_chunks)
            if loss_val.item() < best_val_loss:
                no_improvement = 0
                best_val_loss = loss_val.item()
                if early_stop:                       # snapshot only kept for the best-val restore
                    best_model_state_dict = deepcopy(net.state_dict())
                best_t = t
            else:
                no_improvement += 1
            if early_stop and no_improvement >= T_NO_IMPROVE_THRESHOLD: break
            # report
            if PRINTING and t%(T//20+1) == 0:
               print('finished: t=', t, 'loss=', loss.item(), 'loss val=', loss_val.item(), 'best val loss=', best_val_loss, 'best t=', best_t)
        print('\n')

        # restore the best snapshot (early-stopping mode); otherwise keep the fully-trained weights
        if early_stop:
            net.load_state_dict(best_model_state_dict)
        return best_val_loss