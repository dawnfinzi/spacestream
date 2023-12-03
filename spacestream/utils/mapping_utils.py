"""
Functions specific to mappings
"""
from typing import Callable, List, Optional, Tuple

import numpy as np
import scipy.io
from scipy.optimize import linear_sum_assignment
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture

from spacestream.utils.general_utils import log, sem
from spacestream.utils.regression_utils import train_and_test_scikit_regressor


# Utils for linear regression mapping
def traditional_mapping(features, Y, splits, layer_keys, mapping_func, CV, return_weights=False):

    map_args = None
    gridcv_params = None

    # map from features to voxel responses
    if mapping_func == "PLS":
        map_class = PLSRegression
        if CV == 0:
            map_args = {"n_components": 25, "scale": False}
        elif CV == 1:
            gridcv_params = {
                "n_components": [2, 5, 10, 25, 50, 100],
                "scale": [False],
            }
    elif mapping_func == "Ridge":
        map_class = Ridge
        if CV == 0:
            map_args = {"alpha": 100000}
        elif CV == 1:
            gridcv_params = {
                "alpha": list(np.linspace(0, 100000, 10)),
                "fit_intercept": [True],  # all past CVs have returned True - 02/10/22
            }
    else:  # not instantiated yet
        raise ValueError(f"Mapping function: {mapping_func} not recognized")

    all_resdict = {}
    for l in layer_keys:  # for each layer ...
        print("evaluating %s" % l)
        feats = features[l]

        res = train_and_test_scikit_regressor(
            features=feats,
            labels=Y,
            splits=splits,
            model_class=map_class,
            model_args=map_args,
            gridcv_params=gridcv_params,
            feature_norm=False,
            return_models=True,
        )
        if CV == 1:
            print([_m.best_params_ for _m in res["models"]])  # list winners

        all_resdict[l] = res

    rsquared_array = {}
    for l in layer_keys:
        rsquared_array[l] = all_resdict[l]["test"]["mean_rsquared_array"]

    if return_weights:
        print("yes")
        all_weights = {}
        for l in layer_keys:
            if CV == 1:  # return weights from best model with best params
                all_weights[l] = [
                    all_resdict[l]["models"][s].best_estimator_.coef_
                    for s in range(len(splits))
                ]
            else:
                all_weights[l] = [
                    all_resdict[l]["models"][s].coef_ for s in range(len(splits))
                ]
        return rsquared_array, all_weights

    return rsquared_array

# Utils for one-to-one mapping
def mahalanobis(x, mean, cov):
    """Compute the Mahalanobis Distance between each row of x and the fitted Gaussian (given by mean and covariance)
    x    : vector or matrix of data with p columns - shape (p,2)
    mean : mu of the 2D Gaussian fit to the data - shape (2,)
    cov  : sigma of the 2D Gaussian - shape(2,2)
    """
    x_minus_mu = x - mean
    inv_covmat = scipy.linalg.inv(cov)

    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)

    return mahal.diagonal()


def test_for_convergence(mean_movement, var_range, num_iters):

    if len(mean_movement) > num_iters:
        m = mean_movement[-1]  # movement since last iteration

        upper_bound = m * (1 + (var_range / 2))
        lower_bound = m * (1 - (var_range / 2))
        if np.all(
            [
                (mean_movement[i] > lower_bound and mean_movement[i] < upper_bound)
                for i in range(-num_iters, 0)
            ]
        ):
            return True
        else:
            return False
    else:
        return False


def agg_by_distance(
    distances: np.ndarray,
    values: np.ndarray,
    bin_edges: Optional[np.ndarray] = None,
    num_bins: int = 10,
    agg_fn: Callable[[np.ndarray], float] = np.nanmean,
    spread_fn: Callable[[np.ndarray], float] = sem,
    p_cutoff: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the mean and spread of `values` by binning `distances`, using the
    specified agg_fn and spread_fn
    Inputs
        distances (N,): a flat vector indicating pairwise distance between points.
            Typically this is the flattened lower triangle of a pairwise square matrix,
            excluding the diagonal
        values (N,): a flat vector indicating some pairwise value between units, e.g.,
            the pairwise correlation, or difference in tuning curve peaks
        bin_edges: if provided, `num_bins` is ignored, and bin_edges is not recomputed
        num_bins: how many evenly-spaced bins the distances should be clumped into
        agg_fn: function for aggregating `values` in each bin, defaults to mean
        spread_fn: function for computing spread of `values` in each bin, defaults to
            SEM
        p_cutoff: percentile to cut the distances at for a speed up with super skewed data
    """

    # sort distances and values in order of increasing distance
    dist_sort_ind = np.argsort(distances)
    sorted_distances = distances[dist_sort_ind]
    sorted_values = values[dist_sort_ind]

    if p_cutoff:
        # only calculate using distances up to the 99 percentile for speedup
        cutoff = np.percentile(sorted_distances, p_cutoff)
        sorted_distances = sorted_distances[sorted_distances < cutoff]
        sorted_values = sorted_values[sorted_distances < cutoff]

    # compute evenly-spaced bins for distances
    if bin_edges is None:
        bin_edges = np.histogram_bin_edges(sorted_distances, bins=num_bins)
    else:
        num_bins = len(bin_edges) - 1

    # iterate over bins, computing "means" (can be any aggregated value too) and spreads
    means = np.zeros((num_bins,))
    spreads = np.zeros((num_bins,))

    for i in range(1, num_bins + 1):
        bin_start = bin_edges[i - 1]
        bin_end = bin_edges[i]
        valid_values = sorted_values[
            (sorted_distances >= bin_start) & (sorted_distances < bin_end)
        ]

        if valid_values.shape[0] == 0:
            means[i - 1] = np.nan
            spreads[i - 1] = np.nan
        else:
            means[i - 1] = agg_fn(valid_values)
            spreads[i - 1] = spread_fn(valid_values)

    # return values for each bin along with the computed bin_edges
    return means, spreads, bin_edges

def neighbors_choose_neighbors(
    ite,
    cost_matrix,
    row_ind,
    col_ind,
    source_distances,
    source_radius,
    target_distances,
    target_radius,
    gradient_info,
    distance_cutoff,
):
    for i in range(len(row_ind)):
        actual_row = row_ind[i]
        i_assignment = col_ind[i]
        i_dists = target_distances[actual_row, :]
        i_neighbors = np.where(
            (i_dists < target_radius) & (i_dists > 1e-8)
        )  # within radius mm, ignoring self
        if ~np.any(i_neighbors):  # edge unit, try again with twice the radius
            i_neighbors = np.where((i_dists < target_radius * 2) & (i_dists > 1e-8))
        _, i_neigh_row_inds, _ = np.intersect1d(
            row_ind, i_neighbors, return_indices=True
        )  # not all model units are chosen
        i_neighbor_assignments = col_ind[i_neigh_row_inds]

        valid_source_vertices = i_neighbor_assignments
        for _, j in enumerate(i_neighbor_assignments):
            j_dists = source_distances[j, :]
            j_neighbors = np.where(
                (j_dists < source_radius) & (j_dists > 1e-8)
            )  # radius
            valid_source_vertices = np.append(valid_source_vertices, j_neighbors[0])

        mask = np.ones(cost_matrix.shape[1], bool)
        if ite == 0:  # don't allow repeat assignments on the first iteration
            valid = np.unique(
                np.delete(
                    valid_source_vertices,
                    np.where(valid_source_vertices == i_assignment),
                )
            )
        else:
            valid = np.unique(valid_source_vertices)

        x = -gradient_info["x"][valid]
        y = -gradient_info["y"][valid]
        pts = np.vstack((x, y)).T

        if np.any(pts):
            # fit 2D Gaussian
            gm = GaussianMixture().fit(pts)
            # calculate mahalanobis distance for each point and use a strict cutoff to subselect valid points
            mahal_dists = mahalanobis(pts, gm.means_[0], gm.covariances_[0])
            is_ok = mahal_dists < distance_cutoff
            valid = valid[is_ok]

            mask[valid] = False
            cost_matrix[actual_row, mask] = 100
        else:  # still an empty set of valid source vertices -> don't do any masking
            log("Row " + str(i) + " had no valid source vertices")

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    return row_ind, col_ind
