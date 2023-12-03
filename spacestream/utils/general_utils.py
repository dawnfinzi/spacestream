"""
General utils for project
(credit for many of these to internal lab utils: https://github.com/neuroailab/)
"""

from pathlib import Path
from typing import Any, List, Tuple, Union

import numpy as np
import yaml
from attrdict import AttrDict

Iterable = Union[List, Tuple, np.ndarray]


def norm_image(x):
    return (x - np.min(x)) / np.ptp(x)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    exp = np.exp(x)
    return exp / exp.sum(0)  # sums over axis representing columns


def sem(x, axis=None):
    """
    Return standard error of the mean for the given array
    over the specified axis. If axis is None, std is taken over
    """
    num_elements = x.shape[axis] if axis else len(x)
    return np.nanstd(x, axis=axis) / np.sqrt(num_elements)


def rsquared(predicted, actual):
    """The "rsquared" metric"""
    a_mean = actual.mean()
    num = np.linalg.norm(actual - predicted) ** 2
    denom = np.linalg.norm(actual - a_mean) ** 2
    return 1 - num / denom


def featurewise_norm(data, fmean=None, fvar=None):
    """perform a whitening-like normalization operation on the data, feature-wise
    Assumes data = (K, M) matrix where K = number of stimuli and M = number of features
    """
    if fmean is None:
        fmean = data.mean(0)
    if fvar is None:
        fvar = data.std(0)
    data = data - fmean  # subtract the feature-wise mean of the data
    data = data / np.maximum(fvar, 1e-5)  # divide by the feature-wise std of the data
    return data, fmean, fvar


def make_iterable(x) -> Iterable:
    """
    If x is not already array-like, turn it into a list or np.array
    Inputs
        x: either array_like (in which case nothing happens) or non-iterable,
            in which case it gets wrapped by a list
    """

    if not isinstance(x, (list, tuple, np.ndarray)):
        return [x]
    return x


def log(to_print: str):
    print(f"\nLOG: {to_print}")


def fast_pearson(x, y):
    # faster, vectorized version
    xz = x - x.mean(axis=0)
    yz = y - y.mean(axis=0)
    xzss = (xz * xz).sum(axis=0)
    yzss = (yz * yz).sum(axis=0)
    r = np.matmul(xz.transpose(), yz) / (
        np.sqrt(np.outer(xzss, yzss)) + np.finfo(float).eps
    )  # add machine prec. to avoid any divide by 0
    return np.maximum(np.minimum(r, 1.0), -1.0)  # for precision issues


def load_config_from_yaml(path: Union[str, Path]) -> AttrDict:
    """
    Load a yaml config as an AttrDict
    """
    # force path to a Path object
    path = Path(path)

    with path.open("r") as stream:
        raw_dict = yaml.safe_load(stream)

    return AttrDict(raw_dict)
