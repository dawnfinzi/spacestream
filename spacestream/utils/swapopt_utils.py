import logging
from typing import List

import numpy as np
import scipy.io

# set up logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)-15s %(levelname)s:%(message)s"
)
logger = logging.getLogger(__name__)

def collapse_and_trim_neighborhoods(
    nb_list, keep_fraction=0.95, keep_limit=None, target_shape=None
):
    """
    The neighborhood generation process leads to a different number
    of units in each neighborhood. This function trims all neighborhoods
    to a common value so that the tensor of indices can have static shape.
    Inputs:
        nb_list(list): a list of vectors of neighborhood indices
        keep_fraction (float: [0, 1]): the fraction of neighborhoods to keep
        keep_limit (int) maximum number of units to keep
        target_shape (n_neighborhoods, n_units): the final shape of the neighborhood array
    """

    assert keep_fraction > 0.0 and keep_fraction <= 1.0

    neighborhood_sizes = np.array([len(n) for n in nb_list])
    if target_shape is not None:
        # sort neighborhoods in order from smallest to largest
        sort_ind = np.argsort(neighborhood_sizes)
        target_nbs = target_shape[0]
        target_units = target_shape[1]

        # take the biggest neighborhoods (to maximize chances of having target_units inside)
        neighborhoods_to_keep = sort_ind[-target_nbs:]
        nb_list = np.array(nb_list)[neighborhoods_to_keep]

        n_list = list()
        for indices in nb_list:
            if len(indices) < target_units:
                continue
            indices_to_keep = np.random.choice(
                indices, size=(target_units,), replace=False
            )
            n_list.append(indices_to_keep)
    else:
        logger.info(
            "Neighborhood sizes range from %d to %d, with a median of %d"
            % (
                np.min(neighborhood_sizes),
                np.max(neighborhood_sizes),
                np.median(neighborhood_sizes),
            )
        )

        # convert keep fraction to percentile cutoff
        percentile = (1 - keep_fraction) * 100
        n_size = max(
            int(scipy.stats.scoreatpercentile(neighborhood_sizes, percentile)), 10
        )
        failing = np.less(neighborhood_sizes, n_size)
        logger.info(
            "Minimum of %d excludes %.1f%% of neighborhoods"
            % (n_size, np.mean(failing) * 100.0)
        )

        # take final unit limit as minimum of n_size and keep_limit
        if keep_limit is None:
            final_limit = n_size
        else:
            final_limit = np.min([n_size, keep_limit])

        # do the trimming
        n_list = list()
        for indices in nb_list:
            if len(indices) < final_limit:
                continue
            indices_to_keep = np.random.choice(
                indices, size=(final_limit,), replace=False
            )
            n_list.append(indices_to_keep)

    logger.info("Kept %d neighborhoods." % len(n_list))
    neighborhoods = np.stack(n_list)
    return neighborhoods


def precompute_neighborhoods(
    positions: np.ndarray, radius: float = 0.5, n_neighborhoods: int = 10
):
    """
    Inputs:
        positions: N x 2 position matrix
        radius: radius for a neighborhood (width will be 2 * radius)
        n_neighborhoods: how many neighborhoods to generate
    """
    start_xs = np.random.uniform(
        low=np.min(positions[:, 0]) + radius,
        high=np.max(positions[:, 0]) - radius,
        size=(n_neighborhoods,),
    )

    start_ys = np.random.uniform(
        low=np.min(positions[:, 1]) + radius,
        high=np.max(positions[:, 1]) - radius,
        size=(n_neighborhoods,),
    )

    neighborhoods = [
        indices_within_limits(
            positions,
            [[xstart - radius, xstart + radius], [ystart - radius, ystart + radius]],
        )
        for (xstart, ystart) in zip(start_xs, start_ys)
    ]
    return neighborhoods


def indices_within_limits(positions, limits: List[List[float]], unit_limit=None):
    """
    Inputs
        positions: N x 2 matrix of positions, where N is the number of units
        limits:  The ith list indexes the position limits for the ith column of the
            `positions` matrix
        unit_limit: maximum number of units to retain
    Returns
        indices: a 1-D array of indices where positions are within the limits
    """

    indices = np.where(
        (positions[:, 0] >= limits[0][0])
        & (positions[:, 0] <= limits[0][1])
        & (positions[:, 1] >= limits[1][0])
        & (positions[:, 1] <= limits[1][1])
    )[0]

    if isinstance(unit_limit, int) and len(indices) > unit_limit:
        indices = np.random.choice(indices, size=(unit_limit,), replace=False)

    return indices
