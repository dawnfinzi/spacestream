import argparse
from datetime import datetime
from pathlib import Path

import h5py
import nibabel.freesurfer.mghformat as mgh
import numpy as np
import scipy.io
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture

from spacestream.core.paths import DATA_PATH, RESULTS_PATH
from spacestream.utils.general_utils import fast_pearson, log
from spacestream.utils.get_utils import get_betas
from spacestream.utils.mapping_utils import mahalanobis, test_for_convergence


def write_checkpoint(
    str_path,
    row_ind,
    col_ind,
    ROI_idx,
    streams_trim,
    all_correlations,
    mean_movement,
    accuracy,
    CV,
    test_correlations=None,
    ite=None,
):
    if ite is None:
        full_str_path = str_path + "_final_voxel2voxel_correlation_info.hdf5"
    else:
        full_str_path = (
            str_path + "_checkpoint" + str(ite) + "_voxel2voxel_correlation_info.hdf5"
        )

    winning_val = np.zeros(len(all_correlations))
    winning_roi = np.zeros(len(all_correlations))
    winning_idx = np.zeros(len(all_correlations))

    winning_val[row_ind] = all_correlations[row_ind, col_ind]
    winning_roi[row_ind] = streams_trim[ROI_idx][col_ind]
    winning_idx[row_ind] = ROI_idx[col_ind]  # winning idx wrt streams_trim

    if CV:
        winning_test_val = np.zeros(len(all_correlations))
        winning_test_val[row_ind] = test_correlations[row_ind, col_ind]

    Path(full_str_path).parent.mkdir(parents=True, exist_ok=True)
    h5f = h5py.File(full_str_path, "w")

    voxel2voxel_info = {}
    voxel2voxel_info["winning_roi"] = winning_roi
    voxel2voxel_info["winning_corr"] = winning_val
    voxel2voxel_info["winning_idx"] = winning_idx
    voxel2voxel_info["movement"] = mean_movement
    voxel2voxel_info[
        "accuracy"
    ] = accuracy  # in fsaverage space with subject 2 subject, we can actually track the accuracy of the assignments by computing mean distance from true index
    if CV:
        voxel2voxel_info["winning_test_corr"] = winning_test_val

    for k, v in voxel2voxel_info.items():
        print(str(k))
        h5f.create_dataset(str(k), data=v)
    h5f.close()


def main(
    target_subj,
    source_subj,
    hemi,
    roi,
    radius,
    max_iter,
    distance_cutoff,
    CV,
    seed,
):
    # setup
    base_str_path = (
        RESULTS_PATH
        + "mappings/one_to_one/voxel2voxel/"
        + "/"
        + "target_subj"
        + target_subj
        + "/source_subj"
        + source_subj
        + "/"
        + hemi
        + "_"
        + roi
        + "_HVA_only_radius"
        + str(radius)
        + "_max_iters"
        + str(max_iter)
        + "_constant_radius_"
        + str(distance_cutoff)
        + "dist_cutoff_constant_dist_cutoff_spherical"
        + (("_CV_seed" + str(seed)) if CV else "")
    )

    # get distance matrix
    # get voxel distance matrix
    dfile = DATA_PATH + "brains/ministreams_" + hemi + "_distances.mat"
    with h5py.File(dfile, "r") as f:
        distances = np.array(f["fullspheredists"])

    # get ROI info
    mgh_file = mgh.load(DATA_PATH + "brains/" + hemi + "." + roi + ".mgz")
    streams = mgh_file.get_fdata()[:, 0, 0].astype(int)
    streams_trim = streams[streams != 0]

    # get target and source betas
    target_sorted_validation_betas = get_betas(target_subj, hemi, streams)
    source_sorted_validation_betas = get_betas(source_subj, hemi, streams)

    ROI_idx = np.where((streams_trim == 5) | (streams_trim == 6) | (streams_trim == 7))[
        0
    ]
    target_sorted_validation_betas = target_sorted_validation_betas[:, ROI_idx]
    source_sorted_validation_betas = source_sorted_validation_betas[:, ROI_idx]
    distances = distances[np.ix_(ROI_idx, ROI_idx)]

    # split into train and test if CV'ing
    if CV:
        rng = np.random.RandomState(seed=seed)
        n_imgs = target_sorted_validation_betas.shape[0]
        perm_inds = rng.permutation(n_imgs)
        cutoff = int(0.8 * len(perm_inds))  # 80/20 split

        train_target_betas = target_sorted_validation_betas[perm_inds[0:cutoff], :]
        train_source_betas = source_sorted_validation_betas[perm_inds[0:cutoff], :]

        test_target_betas = target_sorted_validation_betas[perm_inds[cutoff:], :]
        test_source_betas = source_sorted_validation_betas[perm_inds[cutoff:], :]
    else:
        train_target_betas = target_sorted_validation_betas
        train_source_betas = source_sorted_validation_betas

    # get physical location info
    gradient_info = {}
    axes = ["x", "y"]
    for _, ax in enumerate(axes):
        gradient_path = (
            DATA_PATH + "/brains/fsaverage_" + roi + "_" + ax + "gradient.mat"
        )
        if hemi == "lh":
            gradient_info[ax] = np.squeeze(
                scipy.io.loadmat(gradient_path)["left_grad"]
            )[streams != 0]
        else:
            gradient_info[ax] = np.squeeze(
                scipy.io.loadmat(gradient_path)["right_grad"]
            )[streams != 0]

        gradient_info[ax] = gradient_info[ax][ROI_idx]

    # calculate correlations and optimal initial assignment
    all_correlations = fast_pearson(train_target_betas, train_source_betas)
    cost = 1 - all_correlations
    row_ind, col_ind = linear_sum_assignment(cost)
    # NOTE: Here the scipy implementation is used though the lapjv implementation is actually faster
    # The lapjv implementation is not included in this release as it can be a finicky package to install
    # However, if using lapjv instead replace line 218 with:
    # ```
    # col_ind, _, _ = lapjv(cost)
    # row_ind = np.arange(0, len(cost), 1)
    # ```
    if CV:
        test_correlations = fast_pearson(test_target_betas, test_source_betas)
    else:
        test_correlations = None

    prev_col_ind = np.zeros_like(col_ind)
    ite = 0
    mean_movement = []
    accuracy = []
    true_distances = distances[row_ind, col_ind]
    accuracy.append(np.mean(true_distances[~np.isnan(true_distances)]))
    write_checkpoint(  # initial positions
        base_str_path,
        row_ind,
        col_ind,
        ROI_idx,
        streams_trim,
        all_correlations,
        mean_movement,
        accuracy,
        CV,
        test_correlations,
        ite,
    )

    ite_since_last_adjust = 1
    # parameters to check for convergence
    var_range = 0.05  # stable within 5% range
    num_iters = 25  # number of iterations it needs to be stable for
    while (prev_col_ind != col_ind).any() and ite < max_iter:
        log("Iteration no. " + str(ite))
        log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        prev_col_ind = col_ind
        new_cost = cost.copy()

        for i in range(len(row_ind)):
            i_assignment = col_ind[i]
            i_dists = distances[i, :]
            i_neighbors = np.where(
                (i_dists < radius) & (i_dists > 1e-8)
            )  # within radius mm, ignoring self
            i_neighbor_assignments = col_ind[i_neighbors]

            valid_source_vertices = i_neighbor_assignments
            for _, j in enumerate(i_neighbor_assignments):
                j_dists = distances[j, :]
                j_neighbors = np.where((j_dists < radius) & (j_dists > 1e-8))  # radius
                valid_source_vertices = np.append(valid_source_vertices, j_neighbors[0])

            mask = np.ones(len(new_cost), bool)
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

            # fit 2D Gaussian
            gm = GaussianMixture().fit(pts)
            # calculate mahalanobis distance for each point and use a strict cutoff to subselect valid points
            mahal_dists = mahalanobis(pts, gm.means_[0], gm.covariances_[0])
            is_ok = mahal_dists < distance_cutoff
            valid = valid[is_ok]

            mask[valid] = False

            new_cost[i, mask] = 100

        _, col_ind = linear_sum_assignment(new_cost)
        # If using lapjv instead replace line 295 with:
        # ```
        # col_ind, _, _ = lapjv(new_cost)
        # ```

        del new_cost
        ite += 1

        # calculate how far assignments have moved from their previous positions
        movement = distances[prev_col_ind, col_ind]
        movement = movement[~np.isnan(movement)]
        m = np.mean(movement)  # distance from previous assignments
        log(str(m) + " mean distance from previous iter")
        mean_movement.append(m)

        # calculate how far the assignments are from their "true" positions (i.e. position in fsaverage space)
        true_distances = distances[row_ind, col_ind]
        accuracy.append(np.mean(true_distances[~np.isnan(true_distances)]))

        if test_for_convergence(mean_movement, var_range, num_iters):  # end early!
            write_checkpoint(
                base_str_path,
                row_ind,
                col_ind,
                ROI_idx,
                streams_trim,
                all_correlations,
                mean_movement,
                CV,
                test_correlations,
                accuracy,
            )
            log("All done!")
            log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            return

        if ite % 10 == 0:  # every 10 epochs save out a checkpoint
            write_checkpoint(
                base_str_path,
                row_ind,
                col_ind,
                ROI_idx,
                streams_trim,
                all_correlations,
                mean_movement,
                accuracy,
                CV,
                test_correlations,
                ite,
            )

    # final checkpoint and finish up
    write_checkpoint(
        base_str_path,
        row_ind,
        col_ind,
        ROI_idx,
        streams_trim,
        all_correlations,
        mean_movement,
        accuracy,
        CV,
        test_correlations,
    )
    log("All done!")
    log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_subj", type=str)
    parser.add_argument("--source_subj", type=str)
    parser.add_argument("--hemi", type=str, default="rh")
    parser.add_argument("--roi", type=str, default="ministreams")
    parser.add_argument(
        "--radius", type=int, default=5
    )  # radius (in mm) for local neighborhood
    parser.add_argument("--max_iter", type=int, default=100)  # maximum iterations
    parser.add_argument(
        "--distance_cutoff", type=float, default=2.0
    )  # cutoff for mahalanobis neighbor distances
    parser.add_argument(
        "--CV", type=int, default=0
    )  # If 1, cross-validate correlations with 80/20 split
    parser.add_argument(
        "--seed", type=int, default=0
    )  # seed for the splits (ideally, average correlations over 10+ seeds since such a small dataset!)
    ARGS, _ = parser.parse_known_args()

    main(
        ARGS.target_subj,
        ARGS.source_subj,
        ARGS.hemi,
        ARGS.roi,
        ARGS.radius,
        ARGS.max_iter,
        ARGS.distance_cutoff,
        ARGS.CV,
        ARGS.seed,
    )
