import torch
import math
import time
from copy import deepcopy


class NNOptimizer:

    @staticmethod
    def divide_train_val(x, y, ratio=0.80):
        # split into the leading `ratio` fraction (train) and the rest (val), no shuffling
        n = len(x)
        n_train = int(ratio*n)
        x_train, y_train = x[0:n_train], y[0:n_train]
        x_val, y_val = x[n_train:n], y[n_train:n]
        return  x_train, y_train, x_val, y_val

    @staticmethod
    def learn(net, x, y, shuffle=False, early_stop=True, timeout=750):
        # Train `net` by maximizing net.objective_func with Adam, using a val split for
        # early stopping. Returns the best (lowest) validation loss reached.
        # `timeout` (seconds, None to disable) is a wall-clock cap: once exceeded, training
        # stops and the best-so-far weights are kept — bounds cost on slow high-dim cells.

        # optional one-off shuffle before the fixed train/val split (kept off so the split is deterministic)
        if shuffle:
            idx = torch.randperm(len(x))
            x, y = x[idx].clone().detach(), y[idx].clone().detach()

        # hyperparams (fall back to defaults when the net does not set them)
        T = getattr(net, 'max_iteration', 2000)
        bs = getattr(net, 'bs', 200)
        lr = getattr(net, 'lr', 5e-4)
        wd = getattr(net, 'wd', 0.0)
        PRINTING = getattr(net, 'trace_learning', True)
        T_NO_IMPROVE_THRESHOLD = getattr(net, 't_patience', 200)

        # divide train & val (centralized 80/20)
        x_train, y_train, x_val, y_val = NNOptimizer.divide_train_val(x, y)
        net.device = x.device

        # optimizer + batch counts (train batches of size `bs`, val batches of ~1000)
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=wd)
        n_batch, n_val_batch = (int(len(x_train)/bs) if len(x_train) > bs else 1), int(len(x_val)/1000) if len(x_val) > 1000 else 1
        # best-so-far tracking for early stopping / best-val restore
        best_val_loss, best_model_state_dict, best_t, no_improvement = math.inf, None, 0, 0
        t_start = time.time()

        for t in range(T):
            # re-shuffle each epoch, then slice into mini-batches
            idx = torch.randperm(len(x_train))
            x_train, y_train = x_train[idx], y_train[idx]
            x_chunks, y_chunks = torch.chunk(x_train, n_batch), torch.chunk(y_train, n_batch)
            x_v_chunks, y_v_chunks = torch.chunk(x_val, n_val_batch), torch.chunk(y_val, n_val_batch)

            # one epoch of gradient descent (t==0: measure the initial val loss as baseline, no update yet)
            net.train()
            for i in range(len(x_chunks)):
                opt.zero_grad()
                loss = -net.objective_func(x_chunks[i], y_chunks[i])
                if t > 0:
                    loss.backward()
                    opt.step()

            # evaluate on the val split; drives the early-stopping decision below
            net.eval()
            loss_val = torch.zeros(1, device=x.device)
            with torch.no_grad():
                for j in range(len(x_v_chunks)):
                    loss_val += -net.objective_func(x_v_chunks[j], y_v_chunks[j])/len(x_v_chunks)
            # new best: reset the patience counter and snapshot the weights; else count a stale epoch
            if loss_val.item() < best_val_loss:
                no_improvement = 0
                best_val_loss = loss_val.item()
                if early_stop:                       # snapshot only kept for the best-val restore
                    best_model_state_dict = deepcopy(net.state_dict())
                best_t = t
            else:
                no_improvement += 1
            # stop once val loss has stalled for `T_NO_IMPROVE_THRESHOLD` epochs
            if early_stop and no_improvement >= T_NO_IMPROVE_THRESHOLD: break
            # wall-clock timeout: stop early but keep the best-so-far snapshot (restored below)
            if timeout is not None and (time.time() - t_start) > timeout:
                if PRINTING: print(f'timeout {timeout}s hit at t={t} (best_t={best_t}); stopping')
                break
            # periodic progress print (~20 lines over the whole run)
            if PRINTING and t%(T//20+1) == 0:
               print('finished: t=', t, 'loss=', loss.item(), 'loss val=', loss_val.item(), 'best val loss=', best_val_loss, 'best t=', best_t)
        print('\n')

        # restore the best snapshot (early-stopping mode); otherwise keep the fully-trained weights
        if early_stop:
            net.load_state_dict(best_model_state_dict)
        return best_val_loss