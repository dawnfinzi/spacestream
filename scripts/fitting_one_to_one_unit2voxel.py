"""

"""
import argparse
from datetime import datetime
from pathlib import Path
import gc

import h5py
import nibabel.freesurfer.mghformat as mgh
import numpy as np
import scipy.io
import torch
from numpy.random import default_rng
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix

from spacestream.core.constants import SW_PATH_STR_MAPPING
from spacestream.core.feature_extractor import get_features_from_layer
from spacestream.core.paths import DATA_PATH, RESULTS_PATH
from spacestream.datasets.nsd import nsd_dataloader
from spacestream.utils.general_utils import fast_pearson, log
from spacestream.utils.get_utils import get_betas, get_indices, get_model
from spacestream.utils.mapping_utils import (fast_neighbors_choose_neighbors,
                                             test_for_convergence)

rng = default_rng()

TASKS = {"MBs": ["categorization", "action", "detection"], "TDANNs": ["none"]}
MODEL_INFO = {
    "MBs": {
        "18": {
            "layer_name": {
                "categorization": "layer4.1",
                "action": "slow.layer4.1",
                "detection": "backbone.feature_provider.feature_provider.7.1.conv1",
            },
            "model_name": {
                "categorization": "resnet18",
                "action": "slowfast18",
                "detection": "ssd",
            },
            "slowfast_alpha": 8,
            "tasks": ["categorization", "action", "detection"],
        },
        "50": {
            "layer_name": {
                "categorization": "layer4.1",
                "action": "blocks.4.multipathway_blocks.0.res_blocks.1",
                "detection": "backbone.body.layer4.1",
            },
            "model_name": {
                "categorization": "resnet50",
                "action": "slowfast",
                "detection": "faster_rcnn",
            },
            "slowfast_alpha": 4,
            "tasks": ["categorization", "action", "detection"]
        },
        "50_v2": {
             "layer_name": {
                "categorization": "features.7.1",
                "clip": "layer4.1",
                "detection": "model.backbone.conv_encoder.model.encoder.stages.3.layers.1" ,
            },
            "model_name": {
                "categorization": "convnext_tiny",
                "clip": "open_clip_rn50", 
                "detection": "detr_rn50",
            },
            "slowfast_alpha": 4,  # placeholder
            "tasks": ["categorization", "clip", "detection"]
        },
    },
    "TDANNs": {
        "self-supervised": {
            "layer_name": {"none": "base_model.layer4.1"},
            "model_name": {"none": "spacetorch"},
            "slowfast_alpha": 4,  # placeholder
            "tasks": ["none"],
        },
        "supervised": {
            "layer_name": {"none": "base_model.layer4.1"},
            "model_name": {"none": "spacetorch_supervised"},
            "slowfast_alpha": 4,  # placeholder
            "tasks": ["none"],
        },
    },
}
RADIUS = 5.0  # radius (in mm) for local neighborhood
DISTANCE_CUTOFF = 2.0  # cutoff for mahalanobis neighbor distances


def write_checkpoint(
    str_path,
    row_ind,
    col_ind,
    ROI_idx,
    streams_trim,
    all_correlations,
    mean_movement,
    CV,
    test_correlations=None,
    ite=None,
    task_idx=None,
):
    if ite is None:
        full_str_path = (
            str_path
            + "_final_"
            + ("functional_" if task_idx is not None else "")
            + "unit2voxel_correlation_info.hdf5"
        )
    else:
        full_str_path = (
            str_path
            + "_checkpoint"
            + str(ite)
            + ("_functional" if task_idx is not None else "")
            + "_unit2voxel_correlation_info.hdf5"
        )

    winning_val = np.zeros(len(all_correlations))
    winning_roi = np.zeros(len(all_correlations))
    winning_idx = np.zeros(len(all_correlations))

    winning_val[row_ind] = all_correlations[row_ind, col_ind]
    winning_roi[row_ind] = streams_trim[ROI_idx][col_ind]
    winning_idx[row_ind] = ROI_idx[col_ind]  # winning idx wrt streams_trim

    if task_idx is not None:
        winning_task = np.zeros(len(all_correlations))
        winning_task[row_ind] = task_idx[row_ind]

    if CV:
        winning_test_val = np.zeros(len(all_correlations))
        winning_test_val[row_ind] = test_correlations[row_ind, col_ind]

    Path(full_str_path).parent.mkdir(parents=True, exist_ok=True)
    h5f = h5py.File(full_str_path, "w")

    unit2voxel_info = {}
    unit2voxel_info["winning_roi"] = winning_roi
    unit2voxel_info["winning_corr"] = winning_val
    unit2voxel_info["winning_idx"] = winning_idx
    if CV:
        unit2voxel_info["winning_test_corr"] = winning_test_val
    if task_idx is not None:
        unit2voxel_info["winning_task"] = winning_task

    unit2voxel_info["movement"] = mean_movement

    for k, v in unit2voxel_info.items():
        print(str(k))
        h5f.create_dataset(str(k), data=v)
    h5f.close()


def main(
    subj,
    hemi,
    roi,
    model_type,
    base,
    spatial_weight,
    supervised,
    location_type,
    max_iter,
    CV,
    model_seed,
):
    # setup (ugly but backwards compatible)
    if model_type == "TDANNs":
        sub_folder = (
            "/supervised"
            if supervised
            else "/self-supervised"
            + "/spatial_weight"
            + str(spatial_weight)
            + ("_seed" + str(model_seed) if model_seed > 0 else "")
        )
        suffix = (
            "radius5.0_max_iters"
            + str(max_iter)
            + "_constant_radius_2.0dist_cutoff_constant_dist_cutoff_spherical_target_radius_factor1.0"
        )
        assert base == "18", "ResNet50 base not available for TDANNs"
        base = (
            "supervised" if supervised else "self-supervised"
        )  # reassign to match MB structure
    else:
        sub_folder = "/RN" + base
        suffix = "matched_random_subsample_max_iters" + str(max_iter)

    base_str_path = (
        RESULTS_PATH
        + "mappings/one_to_one/unit2voxel/"
        + model_type
        + sub_folder
        + "/subj"
        + subj
        + "/"
        + ("SWAPOPT_" if model_type == "MBs" and location_type == 0 else "RANDOM_" if model_type == "MBs" and location_type == 1 else "")
        + hemi
        + "_"
        + roi
        + ("_CV" if CV else "")
        + "_HVA_only_"
        + suffix
    )

    print(base_str_path)

    # Indexing
    beta_order, _, validation_mask = get_indices(subj)

    model_info = MODEL_INFO[model_type][base]
    task_info = model_info["tasks"]
    if base == "50_v2":
        nsd_batches = 146
        imgs_per_batch = 500
    else:
        nsd_batches = 73
        imgs_per_batch = 1000

    model = {}
    for task in task_info:
        model[task] = get_model(
            model_info["model_name"][task].lower(),
            pretrained=True,
            spatial_weight=spatial_weight,
            model_seed=model_seed,
        )

    # Get indices for subsampling
    chosen_indices = {}
    if model_type == "MBs":
        chosen_save_path = (
            DATA_PATH + "/models/MBs/RN" + base + "/chosen_indices.npz"
        )
        for task in task_info:
            chosen_indices[task] = np.load(chosen_save_path)[task]
   
    # Get model features
    log("Getting features")
    final_features = {}
    subj_stim_idx = np.sort(beta_order)
    prev_batch_end = dict(zip(task_info, [0, 0, 0]))
    for _, task in enumerate(task_info):
        print(task)
        device = "cuda"
        for b in range(nsd_batches):
            log(b)
            subj_batch_idx = subj_stim_idx[
                (subj_stim_idx >= imgs_per_batch * (b))
                & (subj_stim_idx < imgs_per_batch * (b + 1))
            ]
            batch_end = len(subj_batch_idx)

            video, two_pathway, reduction_list = False, False, None
            if task == "action":
                video, two_pathway = True, True
                if base == "50":
                    reduction_list = [2]

            batch = nsd_dataloader(
                list(subj_batch_idx),
                video=video,
                batch_size=len(list(subj_batch_idx)),
                model_name=model_info["model_name"][task],
                slowfast_alpha=model_info["slowfast_alpha"],
            )

            batch_feats = get_features_from_layer(
                model[task],
                batch,
                model_info["layer_name"][task],
                two_pathway=two_pathway,
                reduction_list=reduction_list,
                batch_size=len(list(subj_batch_idx)),
                vectorize=True,
                device=device,
            )
            del batch

            if b == 0:
                if model_type == "MBs":
                    feat_length = len(chosen_indices[task])
                else:
                    feat_length = batch_feats[model_info["layer_name"][task]].shape[
                        1
                    ]  # num feats

                # preallocate array
                final_features[task] = np.zeros(
                    (subj_stim_idx.shape[0], feat_length), dtype=np.float32
                )

            if model_type == "TDANNS":
                chosen_indices[task] = np.arange(feat_length)
            final_features[task][
                prev_batch_end[task] : prev_batch_end[task] + batch_end, :
            ] = batch_feats[model_info["layer_name"][task]][:, chosen_indices[task]]

            prev_batch_end[task] += batch_end
            del batch_feats
            torch.cuda.empty_cache()

        print("end of batch loop for")
        print(task)
        del model[task]
        torch.cuda.empty_cache()
    
    print("get brain info") #DEBUG

    # Get brain data
    # get ROI info
    mgh_file = mgh.load(DATA_PATH + "brains/" + hemi + "." + roi + ".mgz")
    streams = mgh_file.get_fdata()[:, 0, 0].astype(int)
    streams_trim = streams[streams != 0]
    # get source betas
    sorted_betas = get_betas(subj, hemi, streams, chunk_size=100)
    ROI_idx = np.where((streams_trim == 5) | (streams_trim == 6) | (streams_trim == 7))[
        0
    ]
    sorted_betas = sorted_betas[:, ROI_idx]
    # get voxel distance matrix
    dfile = DATA_PATH + "brains/ministreams_" + hemi + "_distances.mat"
    with h5py.File(dfile, "r") as f:
        distances = np.array(f["fullspheredists"])
    radius_as_percent = RADIUS / np.nanmax(
        distances
    )  # get radius as percent of max distance to do conversion for model distances
    distances = distances[np.ix_(ROI_idx, ROI_idx)]
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

    # Get model position and distance data
    if model_type == "TDANNs":
        if supervised:
            coord_path = (
                DATA_PATH
                + "models/TDANNs/spacenet_layer4.1_coordinates_supervised_lw0.npz"
            )
        else:
            coord_path = (
                DATA_PATH + "models/TDANNs/spacenet_layer4.1_coordinates_isoswap_3.npz"
            )
        task_idx = None
    else:
        if location_type == 0:
            # Default is the original, swapopt determined positions
            loc_stem = "swapopt_swappedon_sine_gratings"
        elif location_type == 1:
            # Run a control where the positions are instead randomly determined
            loc_stem = "random" # randomly shuffled version of the above
        coord_path = (
            DATA_PATH
            + "/models/MBs/RN"
            + base
            + "/"
            + loc_stem
            + ".npz"
        )

        task_idx = np.concatenate(
            [
                np.repeat(t_idx, final_features[task].shape[1])
                for t_idx, task in enumerate(task_info)
            ]
        ).ravel()

    total_feats = np.hstack(final_features.values())
    del final_features
    log(str(total_feats.shape[1]))

    coordinates = np.load(coord_path)["coordinates"]
    model_distances = distance_matrix(coordinates, coordinates, p=2)  # euclidean
    model_radius = radius_as_percent * np.max(model_distances)

    # sort into train and test sets if CV'ed
    if CV:
        train_betas = sorted_betas[~validation_mask, :]
        test_betas = sorted_betas[validation_mask, :]

        train_feats = total_feats[~validation_mask, :]
        test_feats = total_feats[validation_mask, :]
    else:
        train_betas = sorted_betas
        train_feats = total_feats
    del sorted_betas, total_feats

    # calculate correlations and optimal initial assignment
    all_correlations = fast_pearson(train_feats, train_betas)
    cost = 1 - all_correlations
    row_ind, col_ind = linear_sum_assignment(cost)
    if CV:
        test_correlations = fast_pearson(test_feats, test_betas)
    else:
        test_correlations = None

    prev_col_ind = np.zeros_like(col_ind)
    ite = 0
    mean_movement = []
    write_checkpoint(  # initial positions
        base_str_path,
        row_ind,
        col_ind,
        ROI_idx,
        streams_trim,
        all_correlations,
        mean_movement,
        CV,
        test_correlations,
        ite,
        task_idx=task_idx,
    )

    # build dict to store starting unit assignments
    unit_keys = range(0, len(model_distances))
    prev_assignments = dict(zip(unit_keys, [None] * len(unit_keys)))
    for r in range(len(row_ind)):
        prev_assignments[row_ind[r]] = col_ind[r]

    # parameters to check for convergence
    var_range = 0.05  # stable within 5% range
    num_iters = 25  # number of iterations it needs to be stable for
    while (prev_col_ind != col_ind).any() and ite < max_iter:
        log("Iteration no. " + str(ite))
        log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        prev_col_ind = col_ind
        new_cost = cost.copy()

        row_ind, col_ind = fast_neighbors_choose_neighbors(
            ite,
            new_cost,
            row_ind,
            col_ind,
            distances,
            RADIUS,
            model_distances,
            model_radius,
            gradient_info,
            DISTANCE_CUTOFF,
        )

        del new_cost
        ite += 1

        # calculate how far assignments have moved from their previous positions
        new_assignments = dict(zip(unit_keys, [None] * len(unit_keys)))
        for r in range(len(row_ind)):  # create a dict for unit assignments on this ite
            new_assignments[row_ind[r]] = col_ind[r]
        movement = []
        for u in range(len(prev_assignments)):
            if (
                prev_assignments[u] and new_assignments[u]
            ):  # unit assigned in both iterations
                movement.append(distances[prev_assignments[u], new_assignments[u]])
        m = np.nanmean(movement)  # distance from previous assignments
        log(str(m) + " mean distance from previous iter")
        mean_movement.append(m)
        # replace previous assignment dict
        prev_assignments = new_assignments

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
                task_idx=task_idx,
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
                CV,
                test_correlations,
                ite,
                task_idx=task_idx,
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
        CV,
        test_correlations,
        task_idx=task_idx,
    )
    log("All done!")
    log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", type=str, default="01")
    parser.add_argument("--hemi", type=str, default="rh")
    parser.add_argument("--roi", type=str, default="ministreams")
    parser.add_argument("--model_type", type=str, default="MBs") #"TDANNs")  # 'TDANNs' or 'MBs'
    parser.add_argument(
        "--base", type=str, default="18",
    )  # base architecture (resnet18 or resnet50 or resnet50_v2 for 2nd set of functional models)
    parser.add_argument("--spatial_weight", type=float, default=1.25) # (only relevant for TDANNs)
    parser.add_argument(
        "--supervised", type=int, default=0
    )  # supervised (1) or simCLR (0) objective (only relevant for TDANNs)
    parser.add_argument(
        "--location_type", type=int, default=0
    )  # random control (1) or swapopt (0) determined position locations (only relevant for MBs)
    parser.add_argument("--max_iter", type=int, default=100)  # maximum iterations
    parser.add_argument(
        "--CV", type=int, default=1
    )  # If 1, cross-validate correlations with 515 shared images as test set
    parser.add_argument("--model_seed", type=int, default=0)  # Seed for model
    ARGS, _ = parser.parse_known_args()

    main(
        ARGS.subj,
        ARGS.hemi,
        ARGS.roi,
        ARGS.model_type,
        ARGS.base,
        ARGS.spatial_weight,
        ARGS.supervised,
        ARGS.location_type,
        ARGS.max_iter,
        ARGS.CV,
        ARGS.model_seed,
    )
