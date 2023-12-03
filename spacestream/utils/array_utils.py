from dataclasses import dataclass
from typing import Union

import numpy as np
from torch import Tensor


@dataclass
class FlatIndices:
    chan_flat: np.ndarray
    x_flat: np.ndarray
    y_flat: np.ndarray


def get_flat_indices(dims):
    """
    dims should be a CHW tuple
    """
    num_channels, num_x, num_y = dims

    # chan flat goes like [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, ...]
    chan_flat = np.repeat(np.arange(num_channels), num_x * num_y)

    # x flat goes like [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, ...]
    x_flat = np.repeat(np.tile(np.arange(num_x), num_channels), num_y)

    # y flat goes like [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, ...]
    y_flat = np.tile(np.arange(num_y), num_x * num_channels)

    return FlatIndices(chan_flat, x_flat, y_flat)


def weighted_circular_mean(samples, weights, wrap=None, axis=0):
    if wrap is not None:
        samples = np.mod(samples, wrap)

    x_components_sum = np.sum(np.cos(samples) * weights, axis=axis)
    y_components_sum = np.sum(np.sin(samples) * weights, axis=axis)
    return np.arctan2(y_components_sum, x_components_sum)


def lower_tri(x):
    """return lower triangle of x, excluding diagonal"""
    assert len(x.shape) == 2
    return x[np.tril_indices_from(x, k=-1)]


def get_flat_lower_tri(x, diagonal=False):
    """
    Returns the flattened lower triangle of a provided matrix
    Inputs
        x (np.ndarray): 2D matrix to get triangle from
        diagonal (bool): if True, keeps the diagonal as part of lower triangle
    """
    k = 0 if diagonal else -1
    lower_idx = np.tril_indices(x.shape[0], k)
    return x[lower_idx]


def sem(x, axis=None):
    """
    Return standard error of the mean for the given array
    over the specified axis. If axis is None, std is taken over

    """
    num_elements = x.shape[axis] if axis else len(x)
    return np.nanstd(x, axis=axis) / np.sqrt(num_elements)


def numel(inp: Union[np.ndarray, Tensor]) -> int:
    """
    Returns the number of elements in a tensor whether it's a numpy
    array or a torch tensor
    """
    if isinstance(inp, np.ndarray):
        return np.prod(inp.shape)
    elif isinstance(inp, Tensor):
        return inp.numel()
    else:
        raise ValueError("inp must be a numpy array or torch tensor")


def dprime(features_on, features_off):
    """
    Compute d-prime for two matrices
    Inputs
        features_on - n_samples x n_features to compute selectivity for
        features_off - n_samples x n_features to compute selectivity against
    """

    m_on = np.nanmean(features_on, axis=0)
    m_off = np.nanmean(features_off, axis=0)
    s_on = np.nanstd(features_on, axis=0)
    s_off = np.nanstd(features_off, axis=0)

    denom = np.sqrt((s_on**2 + s_off**2) / 2)

    # if variance is 0, set d-prime to 0
    return np.where(denom == 0, 0.0, (m_on - m_off) / denom)


def flatten(x: np.ndarray):
    """
    Flatten matrix along all dims but first
    """
    return x.reshape((len(x), -1))
