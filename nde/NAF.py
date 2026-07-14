import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import optimizer


class NAF(nn.Sequential):
    """Neural Autoregressive Flow.

    Extends MAF with learnable marginal transformations via MarginalLayer blocks.
    ``fixed_norm`` prepends a FixedNorm layer to every block: per-dim standardization
    frozen after a data-dependent init, keeping inter-block activations well-scaled
    (constant diagonal Jacobian, no parameters — an optimization aid, not extra capacity).
    """
    def __init__(self, n_blocks, n_inputs, n_hidden, n_cond_inputs, fixed_norm=False):
        module = []
        self.n_blocks = n_blocks
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_cond_inputs = n_cond_inputs
        for _ in range(n_blocks): module += (
            ([FixedNorm(n_inputs)] if fixed_norm else []) + [
            # MADE carries the conditioning on cond_inputs; MarginalLayer is an unconditional
            # per-dimension warp whose parameter-nets are fed a constant dummy of width 2
            # (see MarginalBlock.forward), so its dim_cond is fixed at 2, not n_cond_inputs.
            MADE(n_inputs, n_hidden, n_cond_inputs), MarginalLayer(n_inputs, 2, 1), Reverse(n_inputs)])
        super().__init__(*module)
        self.max_iteration = 100
        self.lr = 1e-3
        self.bs = 250
        self.trace_learning = True

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        """Run the flow in 'direct' or 'inverse' mode. Returns (outputs, sum_log_det)."""
        self.num_inputs = inputs.size(-1)
        sum_logdet = torch.zeros(inputs.size(0), 1, device=inputs.device)
        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module in self._modules.values():
                inputs, logdet = module(inputs, cond_inputs, mode)
                sum_logdet += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, cond_inputs, mode)
                sum_logdet += logdet
        return inputs, sum_logdet

    def log_prob(self, inputs, cond_inputs=None):
        """Alias for log_probs."""
        return self.log_probs(inputs, cond_inputs=None)

    def log_probs(self, inputs, cond_inputs=None):
        """Compute log p(x) = log N(f(x); 0, I) + log |det df/dx|."""
        u, log_jacob = self.forward(inputs, cond_inputs)
        log_base_prob = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(dim=1, keepdim=True)
        return (log_base_prob + log_jacob).sum(dim=1)

    def objective_func(self, x, y):
        """Training objective: mean log-likelihood."""
        return self.log_probs(x, y).mean() if self.cond else self.log_probs(x).mean()

    def learn(self, inputs, cond_inputs=None):
        """Train the flow via NNOptimizer."""
        if cond_inputs is not None:
            self.cond = True
            return optimizer.NNOptimizer.learn(self, x=inputs, y=cond_inputs)
        else:
            self.cond = False
            return optimizer.NNOptimizer.learn(self, x=inputs, y=inputs)


class FixedNorm(nn.Module):
    """Per-dim affine standardization, FROZEN after a data-dependent init (ActNorm variant).

    On the first direct forward, captures the batch mean/std per dimension into buffers
    and never updates them again: no parameters, no gradients, and the Jacobian is the
    constant diagonal -sum(log std) — exactly accounted in the likelihood, so unbiased.
    """

    def __init__(self, n_inputs):
        super().__init__()
        self.register_buffer('mean', torch.zeros(n_inputs))
        self.register_buffer('log_std', torch.zeros(n_inputs))
        self.register_buffer('initialized', torch.tensor(False))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        """Standardize (direct) / de-standardize (inverse). Returns (outputs, logdet[n,1])."""
        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            if not self.initialized:
                self.mean.copy_(inputs.detach().mean(0))
                self.log_std.copy_(inputs.detach().std(0).clamp_min(1e-6).log())
                self.initialized.fill_(True)
            out = (inputs - self.mean) * torch.exp(-self.log_std)
            logdet = -self.log_std.sum()
        else:
            out = inputs * torch.exp(self.log_std) + self.mean
            logdet = self.log_std.sum()
        return out, logdet.view(1, 1).expand(inputs.size(0), 1)


class MarginalLayer(nn.Sequential):
    """Stack of L MarginalBlock layers for element-wise nonlinear transforms."""

    def __init__(self, dim_x, dim_cond, L):
        module = []
        self.L = L
        for _ in range(L): module += [MarginalBlock(dim_x, dim_cond)]
        super().__init__(*module)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        """Apply stacked marginal blocks in direct or inverse mode."""
        self.num_inputs = inputs.size(-1)
        sum_logdet = torch.zeros(inputs.size(0), 1, device=inputs.device)
        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module in self._modules.values():
                inputs, logdet = module(inputs, cond_inputs, mode)
                sum_logdet += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, cond_inputs, mode)
                sum_logdet += logdet
        return inputs, sum_logdet


class MarginalBlock(nn.Module):
    """Element-wise nonlinear transform using sum-of-tanh parameterization:
    x = sum_s A_s * tanh(B_s * z + C_s) + V * z + E.
    """

    def __init__(self, dim_x, dim_cond, S=5):
        super(MarginalBlock, self).__init__()
        self.dim_x = dim_x
        self.S = S
        self.main_A = nn.Sequential(
            nn.Linear(dim_cond, S*dim_x)
        )
        self.main_B = nn.Sequential(
            nn.Linear(dim_cond, S*dim_x)
        )
        self.main_C = nn.Sequential(
            nn.Linear(dim_cond, S*dim_x)
        )
        self.main_V = nn.Sequential(
            nn.Linear(dim_cond, dim_x)
        )
        self.main_E = nn.Sequential(
            nn.Linear(dim_cond, dim_x)
        )

    def ABC(self, cond):
        """Compute transform parameters (A, B, C, V, E) from conditioning input."""
        A = self.main_A(cond)
        B = self.main_B(cond)
        C = self.main_C(cond)
        V = self.main_V(cond)
        E = self.main_E(cond)
        return F.softplus(A), F.softplus(B), C, F.softplus(V), E

    def _tanh_derivative(self, a):
        """Compute d/da tanh(a) = 1 - tanh(a)^2."""
        return 1-torch.tanh(a)**2

    def forward(self, z, cond, mode='direct'):
        """Apply element-wise transform with analytic log-Jacobian."""
        cond = torch.ones(len(z), 2).to(z.device)
        A, B, C, V, E = self.ABC(cond)
        n, D, S = len(A), self.dim_x, self.S
        A, B, C, V, E = A.reshape(n, D, S), B.reshape(n, D, S), C.reshape(n, D, S), V.reshape(n, D), E.reshape(n, D)
        # x
        z0 = z
        z = z.reshape(n, D, 1)
        v = B*z+C
        x = A*torch.tanh(v)                   # <-- n*D*S
        x = x.sum(dim=2)                      # <-- n*D
        x = x + V*z0.view(n, D) + E
        # dx/dz
        det = A*self._tanh_derivative(v)*B    # <-- n*D*S
        det = det.sum(dim=2)                  # <-- n*D
        det = det + V
        log_det_dxdz = det.abs().log()
        log_det_dzdx = -log_det_dxdz.sum(dim=1, keepdim=True)
        return x, -log_det_dzdx


class MADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509).
    """
    def __init__(self, num_inputs, num_hidden, num_cond_inputs=None):
        super(MADE, self).__init__()
        input_mask = self.get_mask(num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = self.get_mask(num_hidden, num_hidden, num_inputs, mask_type='hidden')
        output_mask = self.get_mask(num_hidden, num_inputs, num_inputs, mask_type='output')
        self.join = MaskedLinear(num_inputs, num_hidden, input_mask, num_cond_inputs)
        self.hiddens = nn.Sequential(nn.Tanh(),
                            MaskedLinear(num_hidden, num_hidden, hidden_mask),
                            nn.Tanh(),
                            )
        self.mu = MaskedLinear(num_hidden, num_inputs, output_mask)
        self.alpha = MaskedLinear(num_hidden, num_inputs, output_mask)

    def get_mask(self, n_in, n_out, d, mask_type):
        """Generate autoregressive mask for input, hidden, or output layers."""
        if mask_type == 'input':
            in_degrees = torch.arange(n_in)
            out_degrees = torch.arange(n_out) % (d-1)
            mask = (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()
        elif mask_type == 'output':
            in_degrees = torch.arange(n_in) % (d-1)
            out_degrees = torch.arange(n_out)
            mask = (out_degrees.unsqueeze(-1) > in_degrees.unsqueeze(0)).float()
        elif mask_type == 'hidden':
            in_degrees = torch.arange(n_in) % (d-1)
            out_degrees = torch.arange(n_out) % (d-1)
            mask = (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()
        return mask

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        """Autoregressive transform: direct (x->u) or inverse (u->x) with log-Jacobian."""
        # x -> u, J
        if mode == 'direct':
            h = self.join(inputs, cond_inputs)
            h = self.hiddens(h)
            m, a = self.mu(h), self.alpha(h)
            u = (inputs - m)*torch.exp(-a)
            return u, -a.sum(dim=1, keepdim=True)
        else:
        # u -> x, J
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.join(x, cond_inputs)
                h = self.hiddens(h)
                m, a = self.mu(h), self.alpha(h)
                x[:, i_col] = inputs[:, i_col]*torch.exp(a[:, i_col])+m[:, i_col]
            return x, -a.sum(dim=1, keepdim=True)


class MaskedLinear(nn.Module):
    """Linear layer with a fixed binary mask for autoregressive connectivity."""

    def __init__(self, in_features, out_features, mask, cond_in_features=None):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        if cond_in_features is not None:
            self.cond_linear = nn.Linear(cond_in_features, out_features)
        self.register_buffer('mask', mask)

    def forward(self, inputs, cond_inputs=None):
        """Masked linear transform, optionally conditioned on external inputs."""
        out = F.linear(inputs, self.linear.weight*self.mask, self.linear.bias)
        if cond_inputs is not None:
            out += self.cond_linear(cond_inputs)
        return out


class Reverse(nn.Module):
    """ An implementation of a reversing layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """
    def __init__(self, num_inputs):
        super(Reverse, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        """Reverse dimension ordering (direct) or restore original ordering (inverse)."""
        if mode == 'direct':
            return inputs[:, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)
