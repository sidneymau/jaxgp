"""Gaussian process regression using JAX.
"""

import jax
import jax.numpy as np

import kernels

#-------------------------------------------------------------------------------

def mu_post(xs, xs_train, ys_train, kernel, hparams):
    """
    Posterior mean conditioned on xs.
    Note: a numerical jitter term of 1e-9 is added to avoid nans when inverting
    """
    # TODO: use cholesky decomposition for inversion
    cov_train = kernels.cov_map(kernel, hparams, xs_train, xs_train) \
                + np.eye(len(xs_train), len(xs_train)) * 1e-9
    cov_cross = kernels.cov_map(kernel, hparams, xs, xs_train)
    return cov_cross \
           @ np.linalg.inv(cov_train) \
           @ ys_train

def sigma_post(xs, xs_train, kernel, hparams):
    """
    Posterior covariance conditioned on xs.
    Note: a numerical jitter term of 1e-9 is added to avoid nans when inverting
    """
    # TODO: use cholesky decomposition for inversion
    cov_test = kernels.cov_map(kernel, hparams, xs, xs)
    cov_train = kernels.cov_map(kernel, hparams, xs_train, xs_train) \
                + np.eye(len(xs_train), len(xs_train)) * 1e-9
    cov_cross = kernels.cov_map(kernel, hparams, xs, xs_train)
    cov_cross_r = kernels.cov_map(kernel, hparams, xs_train, xs)
    return cov_test \
           - cov_cross \
             @ np.linalg.inv(cov_train) \
             @ cov_cross_r

if __name__ == '__main__':
    # Dev
    import numpy as onp
    import matplotlib.pyplot as plt
    xs_train = np.linspace(0, 9, 10)
    ys_train = np.sin(xs_train) + onp.random.normal(0.0, 0.3, size=len(xs_train))
    xs_train = xs_train[:,np.newaxis]
    ys_train = ys_train[:,np.newaxis]
    hparams = {'sigma' : 0.0,
               's'     : 0.5,
               'l'     : 1.0,
               'invCov'   : np.linalg.inv(np.array([[1.0, 0.0], [0.0, 1.0]]))}
    xs = np.linspace(0, 9, 100)[:,np.newaxis]

    mu = mu_post(xs, xs_train, ys_train, kernels.squared_exponential, hparams)
    sigma = sigma_post(xs, xs_train, kernels.squared_exponential, hparams)
    plt.errorbar(xs, mu, np.sqrt(np.diag(sigma)))
    plt.plot(xs_train, ys_train)
    import pdb;pdb.set_trace()
