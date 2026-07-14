"""Multivariate Student-t benchmark with an exact mutual information.

The joint ``(X, Y)`` is multivariate Student-t with a block dispersion matrix
coupling X and Y. Unlike the Gaussian benchmarks, the heavy tails add a
dependence on top of the linear correlation, so the MI is the sum of a Gaussian
term (from the dispersion, computed via :class:`SplitMultinormal`) and a
tail-correction term that depends only on the dimensions and degrees of freedom
(:meth:`MultivariateStudentT.mi_correction`).
"""

import numpy as np
from scipy.special import digamma, gamma
from scipy.stats import multivariate_normal






def _differential_entropy(k: int, dof: int) -> float:
    """Differential entropy of a :math:`Student-t(0, I_k, dof)`.

    See Eq. (7) of
      R.B. Arellano-Valle, J.E. Contreras-Reyes, M.G. Genton,
      Shannon Entropy and Mutual Information for Multivariate
      Skew-Elliptical Distributions,
      Scandinavian Journal of Statistics, vol. 40, pp. 46-47, 2013
    """
    half_sum = 0.5 * (dof + k)
    digamma_term = half_sum * (digamma(half_sum) - digamma(0.5 * dof))

    log_term = -np.log(gamma(half_sum)) + np.log(gamma(0.5 * dof)) + 0.5 * k * np.log(dof * np.pi)

    return log_term + digamma_term



def multivariate_t(mu, sigma, dof, m):
    """
    Produce m samples of d-dimensional multivariate t distribution
    
    Args:
        mu (numpy.ndarray): mean vector
        sigma (numpy.ndarray): scale matrix (covariance)
        dof (float): degrees of freedom
        m (int): # of samples to produce

    Returns:
        numpy.ndarray
    """
    d = len(sigma)
    g = np.tile(np.random.gamma(dof / 2 , 2 / dof, m), (d, 1)).T
    z = np.random.multivariate_normal(np.zeros(d), sigma, m)
    return mu + z / np.sqrt(g)




class MultivariateStudentT():
    """Multivariate Student-t distribution.

    Sampling is based on
    [Wikipedia](https://en.wikipedia.org/wiki/Multivariate_t-distribution)

    Mutual information is based on:

    R.B. Arellano-Valle, J.E. Contreras-Reyes, M.G. Genton,
    Shannon Entropy and Mutual Information for Multivariate
    Skew-Elliptical Distributions,
    Scandinavian Journal of Statistics, vol. 40, pp. 46-47, 2013

    Note that the final formula for the mutual information is slightly wrong,
    but can be calculated using the expressions involving differential entropies
    above.
    """

    def __init__(
        self,
        *,
        dim_x: int,
        dim_y: int,
        df: int,
        dispersion: np.array,
        mean:  np.array
    ) -> None:
        """

        Args:
            dim_x: dimension of the X variable
            dim_y: dimension of the Y variable
            df: degrees of freedom, strictly positive. Use `np.inf` for a Gaussian
            dispersion: dispersion matrix, shape `(dim_x + dim_y, dim_x + dim_y)`
            mean: mean of the distribution, shape `(dim_x + dim_y,)`. Default: zero vector

        Note:
            Dispersion is *not* the covariance matrix.
            To calculate the covariance matrix, use the `covariance` method.
        """
        
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_total = dim_x + dim_y
  
        if mean is None:
            mean = np.zeros(dim_x + dim_y)
        self._mean = np.asarray(mean)

        # Mutual information of multivariate Student-t contains
        # the term corresponding to the multivariate normal distribution.
        # Note that this will also validate all the dimensions
        # and check whether the dispersion matrix is positive-definite
        self._multinormal = SplitMultinormal(
            dim_x=dim_x, dim_y=dim_y, mean=mean, covariance=dispersion
        )

        if df <= 0:
            raise ValueError("Degrees of freedom must be positive.")
        self._degrees_of_freedom = df

        self._dispersion = np.asarray(dispersion)

    def sample(self, n_points: int):
        """Sampling from multivariate Student distribution.

        Note:
            This function is based on SciPy's sampling.
        """
        xy = multivariate_t(
            mu=self._mean,
            sigma=self._dispersion,
            dof=self._degrees_of_freedom,
            m=n_points,
        )
        
        assert xy.shape == (
            n_points,
            self.dim_total,
        ), f"Wrong shape: {xy.shape} != {(n_points, self.dim_total)}."

        x, y = xy[:, : self.dim_x], xy[:, self.dim_x :]  # noqa: E203 colon spacing conventions
        mu_x, std_x, mu_y, std_y = x.mean(axis=0, keepdims=True), x.std(axis=0, keepdims=True), y.mean(axis=0, keepdims=True), y.std(axis=0, keepdims=True)
        return (x-mu_x)/std_x, (y-mu_y)/std_y
        

    @property
    def df(self) -> int:
        """Degrees of freedom."""
        return self._degrees_of_freedom

    def covariance(self) -> np.ndarray:
        """Calculates the covariance matrix.

        Returns:
            array, shape `(dim_x+dim_y, dim_x+dim_y)`

        Raises:
            ValueError: if covariance is not defined (for `df` $\\le 2$)
        """
        if self.df <= 2:
            raise ValueError(
                f"For degrees of freedom {self.df} <= 2 the covariance is not defined."
            )
        else:
            return self.df * self._dispersion / (self.df - 2.0)

    def mi_normal(self) -> float:
        """Part of the mutual information corresponding to a multivariate
        normal with covariance given by the dispersion matrix.
        """
        return self._multinormal.mutual_information()

    def mi_correction(self) -> float:
        """Correction term in MI calculation.

        This term depends only on the dimensionality of each variable
        and the degrees of freedom.
        (It does not depend on the dispersion matrix or mean vector used).
        """
        df, dim_x, dim_y = self.df, self.dim_x, self.dim_y
        h_x = _differential_entropy(k=dim_x, dof=df)
        h_y = _differential_entropy(k=dim_y, dof=df)
        h_xy = _differential_entropy(k=dim_x + dim_y, dof=df)
        return h_x + h_y - h_xy

    def mutual_information(self) -> float:
        """Expression for MI taken from Arellano-Valle et al., p. 47.

        This expression consists of two terms:
            `mi_normal`, which is the MI of a multivariate normal distribution
              with covariance given by the dispersion
            `mi_correction`, which is a correction term which does not depend
              on the means or the dispersion
        """

        return self.mi_normal() + self.mi_correction()

    
    
    
    
    
    
    
    
    
class _Multinormal:
    """Auxiliary object for representing multivariate normal distributions."""

    def __init__(self, mean, covariance) -> None:
        """
        Args:
            mean: mean vector of the distribution, shape (dim,)
            covariance: covariance matrix of the distribution, shape (dim, dim)
        """
        # Mean and the covariance
        self._mean = np.asarray(mean)
        self._covariance = np.asarray(covariance)

        # The determinant of the covariance, used to calculate entropy
        self._det_covariance: float = np.linalg.det(self._covariance)

        # Dimensionality of the space
        self._dim = self._mean.shape[0]

        # Validate the shape
        if self._covariance.shape != (self._dim, self._dim):
            raise ValueError(
                f"Covariance has shape {self._covariance.shape}, expected "
                f"{(self._dim, self._dim)}."
            )

    def sample(self, n_samples: int) -> np.ndarray:
        """Sample from the distribution.

        Args:
            n_samples: number of samples to generate

        Returns:
            samples, shape (n_samples, dim)
        """
        return multivariate_normal.rvs(mean=self._mean, cov=self._covariance, size=n_samples)


    @property
    def dim(self) -> int:
        """The dimensionality."""
        return self._dim

    def entropy(self) -> float:
        """Entropy in nats."""
        return 0.5 * (np.log(self._det_covariance) + self.dim * (1 + np.log(2 * np.pi)))


class SplitMultinormal():
    """Represents two variables with jointly
    multivariate normal distribution

    Covariance matrix should have the block form:

    $$\\Sigma = \\begin{pmatrix}
            \\Sigma_{XX} & \\Sigma_{XY} \\\\
            \\Sigma_{YX} & \\Sigma_{YY}
    \\end{pmatrix}$$

    where:

    - $\\Sigma_{XX}$ is the covariance matrix of $X$ variable (shape `(dim_x, dim_x)`),
    - $\\Sigma_{YY}$ is the covariance of the $Y$ variable (shape `(dim_y, dim_y)`)
    - $\\Sigma_{XY}$ and $\\Sigma_{YX}$
      (being transposes of each other, as the matrix is symmetric,
      of shapes `(dim_x, dim_y)` or transposed one) describe the covariance between $X$ and $Y$.
    """

    def __init__(
        self, *, dim_x: int, dim_y: int, covariance, mean= None
    ) -> None:
        """

        Args:
            dim_x: dimension of the X space
            dim_y: dimension of the Y space
            mean: mean vector, shape `(n,)` where `n = dim_x + dim_y`.
                Default: zero vector
            covariance: covariance matrix, shape (n, n)
        """
        self.dim_total = dim_x + dim_y
        
        # The default mean vector is zero
        if mean is None:
            mean = np.zeros(dim_x + dim_y)

        # Set mean and covariance
        self._mean = np.array(mean)
        self._covariance = np.array(covariance)
        self._validate_shapes()

        self._joint_distribution = _Multinormal(mean=self._mean, covariance=self._covariance)
        self._x_distribution = _Multinormal(
            mean=self._mean[:dim_x], covariance=self._covariance[:dim_x, :dim_x]
        )
        self._y_distribution = _Multinormal(
            mean=self._mean[dim_x:], covariance=self._covariance[dim_x:, dim_x:]
        )

    def _validate_shapes(self) -> None:
        n = self.dim_total

        if self._mean.shape != (n,):
            raise ValueError(f"Mean vector has shape {self._mean.shape}, expected ({n},).")
        if self._covariance.shape != (n, n):
            raise ValueError(
                f"Covariance matrix has shape {self._covariance.shape}, " f"expected ({n}, {n})."
            )

    def mutual_information(self) -> float:
        """Calculates the mutual information I(X; Y) using an exact formula.
        Returns:
            mutual information, in nats
        Mutual information is given by

        $$I(X; Y) = \\frac 12 \\log \\left(\\frac{\\det(\\Sigma_{XX})
        \\det(\\Sigma_{YY})}{\\det(\\Sigma)}\\right)$$

        which follows from the formula
            $I(X; Y) = H(X) + H(Y) - H(X, Y)$
        and the formula for the differential entropy of the multivariate
        normal distribution.
        """
        h_x = self._x_distribution.entropy()
        h_y = self._y_distribution.entropy()
        h_xy = self._joint_distribution.entropy()
        mi = h_x + h_y - h_xy  # Mutual information estimate
        return max(0.0, mi)  # Mutual information is always non-negative