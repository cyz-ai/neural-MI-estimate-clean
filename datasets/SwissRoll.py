import numpy as np
import torch
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch.distributions as distribution


class SwissRoll(Dataset):
    """"""

    def __init__(self, n_samples=100000, n_dims=80, rho=0.80, mu=0):
        """ """
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
        return self.transformation(X, Y)
        
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
            transform the data to make data a swise roll
        '''
        # x, y -> F(x), F(y)
        mu, sigma = x*0, x*0 + 1
        normal = distribution.normal.Normal(mu, sigma)
        ux, uy = normal.cdf(x), normal.cdf(y)
        # u -> (a, b, c), 3d coordinate
        t = 3*np.pi/2*(1+2*ux)
        e1, e2 = t*torch.cos(t)/21, t*torch.sin(t)/21
        eps = torch.randn_like(uy)
        data = [e1, e2, uy, eps]
        # new x, new y
        X = torch.cat([data[0], data[1]], dim=1)
        Y = torch.cat([data[2], data[3]], dim=1)
        return X + 0.05*torch.randn_like(X), Y + 0.05*torch.rand_like(Y)   


    def plot(self, X, Y):
        # Split the data into three arrays for plotting
        x = X[0:1000, 0].cpu().numpy()
        z = X[0:1000, 1].cpu().numpy()
        y = Y[0:1000, 0].cpu().numpy()

        # Create a new figure for plotting
        fig = plt.figure()

        # Add a 3D subplot
        # The '111' means "1x1 grid, first subplot" and 'projection="3d"' makes it 3D
        ax = fig.add_subplot(111, projection='3d')

        # Plot the data
        ax.scatter(x, y, z, s=2)

        # grid and view angle
        ax.grid(False)
        ax.view_init(elev=35, azim=125)

        # Set labels for axes
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        # Display the plot
        plt.show()






