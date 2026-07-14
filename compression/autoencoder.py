import torch
import torch.nn as nn
import optimizer


class Autoencoder(nn.Module):
    """Softplus MLP autoencoder trained by MSE reconstruction, used to compress a
    high-dimensional variable to a ``latent_dim`` code before MI estimation."""

    def __init__(self, x_dim, latent_dim):
        super().__init__()
        self.max_iteration = 1000
        self.lr = 1e-3
        self.bs = 250
        self.wd = 0.0
        self.latent_dim = latent_dim

        Lx = 1024
        self.x_encoder = nn.Sequential(nn.Linear(x_dim, Lx),
                                       nn.Softplus(),
                                       nn.BatchNorm1d(Lx),
                                       nn.Linear(Lx, Lx),
                                       nn.Softplus(),
                                       nn.BatchNorm1d(Lx),
                                       nn.Linear(Lx, Lx),
                                       nn.Softplus(),
                                       nn.BatchNorm1d(Lx),
                                       nn.Linear(Lx, latent_dim),
                                       nn.Softplus())
        self.xx_decoder = nn.Sequential(nn.Linear(latent_dim, Lx),
                                        nn.Softplus(),
                                        nn.BatchNorm1d(Lx),
                                        nn.Linear(Lx, Lx),
                                        nn.Softplus(),
                                        nn.BatchNorm1d(Lx),
                                        nn.Linear(Lx, Lx),
                                        nn.Softplus(),
                                        nn.BatchNorm1d(Lx),
                                        nn.Linear(Lx, x_dim))

    def encode(self, X):
        return self.x_encoder(X)

    def decode(self, Zx):
        return self.xx_decoder(Zx)

    def rec_loss(self, hat, samples):
        """Mean-squared reconstruction error."""
        return torch.nn.functional.mse_loss(hat, samples, reduction='mean')

    @staticmethod
    def _compressibility_metrics(hat, samples, eps_rel=1e-6):
        """Eval-only. Returns (comp_mean, comp_frob) as Python floats.

        comp_mean = 1 - mean_i( ||x_i - x̂_i||^2 / ||x_i||^2 ), per-sample ratios clamped
                    to [0, 1] and rows with near-zero norm filtered.
        comp_frob = 1 - ||X - X̂||_F^2 / ||X||_F^2 (same row filter).
        """
        norms_sq = (samples ** 2).sum(dim=1)
        thr = eps_rel * norms_sq.median()
        keep = norms_sq > thr
        if keep.sum().item() == 0:
            return float('nan'), float('nan')
        x = samples[keep]
        xh = hat[keep]
        per = ((xh - x) ** 2).sum(dim=1) / (x ** 2).sum(dim=1)
        comp_mean = float(1.0 - per.clamp(0.0, 1.0).mean().item())
        comp_frob = float(1.0 - (((xh - x) ** 2).sum() / (x ** 2).sum()).item())
        return comp_mean, comp_frob

    @torch.no_grad()
    def compressibility(self, X):
        """Reconstruction quality of X in [~0, 1] (higher = more compressible)."""
        self.eval()
        return self._compressibility_metrics(self.decode(self.encode(X)), X)

    def objective_func(self, X, Y=None):
        return -self.rec_loss(hat=self.decode(self.encode(X)), samples=X)

    def learn(self, X):
        # early_stop=False: reconstruction wants full convergence over max_iteration, not the
        # best-val snapshot -- all other learners keep the optimizer default early_stop=True.
        optimizer.NNOptimizer.learn(self, X, X, early_stop=False)
