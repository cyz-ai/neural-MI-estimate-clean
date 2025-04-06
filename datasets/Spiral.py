import numpy as np
import torch
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch.distributions as distribution


class Spiral(Dataset):
    """"""

    def __init__(self, rho=0.80, dim=6, v=1.0):
        """ """
        self.n_dims = dim
        self.mu = np.zeros(self.n_dims)
        self.rho = rho
        self.rhos = np.ones(self.n_dims // 2) * self.rho
        self.cov_matrix = block_diag(*[[[1, self.rho], [self.rho, 1]] for _ in range(self.n_dims // 2)])
        self.data = self._sample_gaussian(10000, self.cov_matrix).astype(dtype=np.float32)
        
        self.vx = v*1.0/(self.n_dims//2)
        self.vy = v*1.0/(self.n_dims//2)
        
        
    def sample(self, n):
        
        _, dxdy = self.data.shape
        d = dxdy//2
    
        x, y = torch.Tensor(self.data[:, ::2]).clone(), torch.Tensor(self.data[:, 1::2]).clone()
        x, y = torch.Tensor(x), torch.Tensor(y)
        
        x_norm, y_norm = x.norm(dim=1)**2, y.norm(dim=1)**2
        
        A = so_generator(self.n_dims//2, 0, 1)
        B = so_generator(self.n_dims//2, 1, 2)                # or 0, 1, also works
        
        A, B = torch.Tensor(A), torch.Tensor(B)
        new_x, new_y = torch.zeros(n, d), torch.zeros(n, d)
        
        for i in range(n):
            new_x[i] = x[i]@torch.linalg.matrix_exp(self.vx*A*x_norm[i])
            new_y[i] = y[i]@torch.linalg.matrix_exp(self.vy*B*y_norm[i])
            
        return new_x[0:n], new_y[0:n]
    
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
    
    def MI(self):
        return self._get_mi_from_rho(self.rho, self.n_dims)
        
        
    def plot(self, x, y, dims=[0, 1]):

        plt.scatter(x[0:1000, dims[0]].cpu().numpy(), x[0:1000, dims[1]].cpu().numpy())

        # Add labels and title
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("Scatter plot of Spiral")

        # Show plot
        plt.show()
        
        
        
        
        
        
        
def so_generator(n: int = 3, i: int = 0, j: int = 1):
    """The (i,j)-th canonical generator of the so(n) Lie algebra.

    As so(n) Lie algebra is the vector space of all n x n
    skew-symmetric matrices, we have a canonical basis
    such that its (i,j)th vector is a matrix A such that
          A[i, j] = -1, A[j, i] = 1, i < j
    and all the other entries are 0.

    Note that there exist n(n-1)/2 such matrices.

    Args:
        n: we use the Lie algebra so(n)
        i: index in range {0, 1, ..., j-1}
        j: index in range {i+1, i+2, ..., n-1}

    Returns:
        NumPy array (n, n)

    Note:
        This function is NumPy based and is *not* JITtable.
    """
    if n < 2:
        raise ValueError(f"n needs to be at least 2.")
    if not (0 <= i < j < n):
        raise ValueError(f"Index is wrong: n{n} i{i} j{j}.")

    a = np.zeros((n, n))
    a[i, j] = -1
    a[j, i] = 1
    return a

