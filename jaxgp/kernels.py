"""Kernel methods for Gaussian processes using JAX.
"""

import jax
from jax import vmap
import jax.numpy as np

#-------------------------------------------------------------------------------

def white_noise(hparams, x1, x2):
    """
    White noise kernel
    """
    return hparams['sigma'] * np.eye(len(x1), len(x2))

def rbf(hparams, x1, x2):
    """
    Isotropic squared exponential kernel
    """
    r2 = np.square(x1 / hparams['l'] - x2 / hparams['l'])
    # Don't forget to sum along last axis
    return hparams['s'] * np.exp(-np.sum(r2))

def squared_exponential(hparams, x1, x2):
    """
    Anisotropic squared exponential kernel
    """
    r2 = (x1 - x2) * hparams['invCov'] * (x1 - x2).T
    # Don't forget to sum along last axis
    return hparams['s'] * np.exp(-np.sum(r2))

#def von_karman(hparams, x1, x2):
#    """
#    Anisotropic von karman kernel
#    """
#    r2 = (x1 - x2) * hparams['invCov'] * (x1 - x2).T
#    r = np.sqrt(r2)
#
#    # Don't forget to sum along last axis
#    return hparams['s'] * np.power(r, 5./6.) \
#           * kv(-5./6., 2*np.pi*r)

def cov_map(kernel, hparams, xs1, xs2):
    """Compute a covariance matrix from a covariance function and data points.

    Args:
        kernel: callable function, maps pairs of data points to scalars.
        xs1: array of data points, stacked along the leading dimension.
        xs2: array of data points, stacked along the leading dimension.
    Returns:
        A 2d array `a` such that `a[i, j] = kernel(xs[i], xs[j])`.
    """
    return vmap(lambda x1: vmap(lambda x2: kernel(hparams, x1, x2))(xs2))(xs1)

