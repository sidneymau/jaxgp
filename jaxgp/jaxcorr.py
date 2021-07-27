"""Implementation of 2pcf using JAX.
"""

import jax
import jax.numpy as np
from jax import vmap

#-------------------------------------------------------------------------------

def euclidean_distance(x1, x2):
    """
    Compute the distance between two points.

    Args:
        x1: First array of points.
        x2: Second array of points.

    Returns:
        Euclidean distance between points.
    """
    return np.sqrt(np.sum(np.square(np.subtract(x1, x2))))

def pair_product(x1, x2):
    """
    Compute the pair product of two points.

    Args:
        x1: First array of points.
        x2: Second array of points.

    Returns:
        Pair product of points.
    """
    return np.multiply(x1, x2)

def distance_map(xs1, xs2):
    """
    Compute the distance matrix between two arrays.

    Args:
        xs1: First array of points.
        xs2: Second array of points.

    Returns:
        Euclidean distance matrix between all points.
    """
    return jax.vmap(lambda x1: jax.vmap(lambda x2: euclidean_distance(x1, x2))(xs2))(xs1)

def product_map(xs1, xs2):
    """
    Compute the pair products between two arrays.

    Args:
        xs1: First array of points.
        xs2: Second array of points.

    Returns:
        Pair products matrix of all points.
    """
    return jax.vmap(lambda x1: jax.vmap(lambda x2: pair_product(x1, x2))(xs2))(xs1)

def compute_2pcf_1D(xs, ys, bins=10):
    """
    Compute the 2-point correlation function of the field.

    Args:
        xs: Array of the x- and y-components of the process.
        ys: Array of the x- and y-components of the process.
        bins: The number of distance bins at which to compute
              the 2-point correlation functions.

    Returns:
        dr: ndarray,
            separations at which the 2-point correlation functions were calculated
        xi0: ndarray,
            2-point correlation function of the x-component of the process
        xi1: ndarray,
            2-point correlation function of the y-component of the process
    """
    # calculate the euclidean separation between each point in the process
    seps = distance_map(xs, xs)

    # calculate the pair products of each component of each point of the process
    pps = product_map(ys, ys)

    # ``upper triangle`` indices for an (N, N) array
    ind = np.triu_indices(seps.shape[0])

    # Use histograms to efficiently select pps according to sep
    # Inspired by Gary Bernstein via Pierre-Francois Leget
    counts, dr = np.histogram(seps[ind], bins=bins)
    xi, _ = np.histogram(seps[ind], bins=bins, weights=pps[ind])

    # Normalize quantities
    dr = 0.5*(dr[:-1]+dr[1:])
    xi /= counts

    return dr, xi

def compute_2pcf_2D(xs, ys, bins=10):
    """
    Compute the 2-point correlation function of each component of the field.

    Args:
        xs: Array of the x- and y-components of the field.
        ys: Array of the x- and y-components of the field.
        bins: The number of distance bins at which to compute
              the 2-point correlation functions.

    Returns:
        dr: ndarray,
            separations at which the 2-point correlation functions were calculated
        xi0: ndarray,
            2-point correlation function of the x-component of the field
        xi1: ndarray,
            2-point correlation function of the y-component of the field
    """
    # seps has shape (N, N) up to ind
    # calculate the euclidean separation between each point in the process
    seps = distance_map(xs, xs)

    # pps0, pps1 have shape (N, N) up to ind
    # calculate the pair products of each component of each point of the process
    pps = product_map(ys, ys)
    pps0 = pps[:,:,0]
    pps1 = pps[:,:,1]

    # ``upper triangle`` indices for an (N, N) array
    ind = np.triu_indices(seps.shape[0])

    # Use histograms to efficiently select pps according to sep
    # Inspired by Gary Bernstein via Pierre-Francois Leget
    counts, dr = np.histogram(seps[ind], bins=bins)
    xi0, _ = np.histogram(seps[ind], bins=bins, weights=pps0[ind])
    xi1, _ = np.histogram(seps[ind], bins=bins, weights=pps1[ind])

    # Normalize quantities
    dr = 0.5*(dr[:-1]+dr[1:])
    xi0 /= counts
    xi1 /= counts

    return dr, xi0, xi1

#def compute_cov(xs, ys):
#    """
#    Want a covariance matrix measured from the data...
#    """
#    seps = distance_map(xs, xs)
#
#    pps = product_map(ys, ys)
#    pps0 = pps[:,:,0]
#    pps1 = pps[:,:,1]
#
#    ind = np.triu_indices(seps.shape[0])

