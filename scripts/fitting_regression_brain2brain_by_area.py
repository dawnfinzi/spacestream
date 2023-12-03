import argparse

import h5py
import nibabel.freesurfer.mghformat as mgh
import numpy as np

from spacestream.core.constants import SUBJECTS
from spacestream.core.paths import BETA_PATH, DATA_PATH, RESULTS_PATH
from spacestream.utils.get_utils import get_indices
from spacestream.utils.mapping_utils import traditional_mapping
from spacestream.utils.regression_utils import get_splits


def log(to_print: str):
    print(f"\nLOG: {to_print}")


def main(
    target_subj,
    hemi,
    target_area,
    target_roi,
    source_roi,
    n_source_voxels,
    num_splits,
    subsamp_type,
    pooled,
):
    ROIS = [
        "Ventral",
        "Lateral",
        "Parietal",
    ]  # ROI_NAMES[1:]  # all ROIs except 'Unknown'
    skip = 5  # ROIs skipped
    source_subj = [s for s in SUBJECTS if s != target_subj]
    num_images = 515  # number of shared val images
    rng = np.random.RandomState(seed=0)

    # Get target betas
    log("Getting target betas")
    betas = h5py.File(
        DATA_PATH
        + "brains/organized_betas_by_area/subj"
        + target_subj
        + "_"
        + hemi
        + "_"
        + target_roi
        + "_"
        + target_area
        + ".hdf5",
        "r",
    )
    beta_order, beta_mask, validation_mask = get_indices(target_subj, shared=True)
    # sort betas
    indx = beta_order.argsort(axis=0)
    betas = np.array(betas[target_area])[
        beta_mask, :
    ]  # returns only images with all 3 trials
    sorted_betas = betas[indx, :]
    target_sorted_validation_betas = sorted_betas[validation_mask, :]
    del sorted_betas, betas

    # Get source betas (i.e. "features")
    log("Getting source betas")
    features = {}
    stacked_nc = {}
    stacked_vox = [{} for i in range(len(source_subj))]
    all_nc = [[[] for j in range(len(ROIS))] for i in range(len(source_subj))]

    for sidx, subj in enumerate(source_subj):
        if subsamp_type == 1:
            # Get ROI data
            mgh_file = mgh.load(
                DATA_PATH + "brains/" + hemi + "." + source_roi + ".mgz"
            )
            streams = mgh_file.get_fdata()[:, 0, 0].astype(int)
            streams_trim = streams[streams != 0]
            # Get noise ceiling estimates
            mgh_file = mgh.load(
                DATA_PATH + "brains/NC/subj" + subj + "/" + hemi + ".nc_3trials.mgh"
            )
            NC = mgh_file.get_fdata()[:, 0, 0]
            NC_trim = NC[streams != 0]
            for r in range(len(ROIS)):
                all_nc[sidx][r] = NC_trim[streams_trim == r + skip] / 100

        for r, area in enumerate(ROIS):
            betas = h5py.File(
                DATA_PATH
                + "brains/organized_betas_by_area/subj"
                + subj
                + "_"
                + hemi
                + "_"
                + source_roi
                + "_"
                + area
                + ".hdf5",
                "r",
            )
            # indexing
            beta_order, beta_mask, validation_mask = get_indices(subj, shared=True)
            # sort betas
            indx = beta_order.argsort(axis=0)
            betas = np.array(betas[area])[
                beta_mask, :
            ]  # returns only images with all 3 trials
            sorted_betas = betas[indx, :]
            source_sorted_validation_betas = sorted_betas[validation_mask, :]
            stacked_vox[sidx][area] = source_sorted_validation_betas

            if sidx == 0:
                features[area] = source_sorted_validation_betas
                stacked_nc[area] = all_nc[sidx][r]
            else:
                features[area] = np.hstack(
                    (features[area], source_sorted_validation_betas)
                )
                stacked_nc[area] = np.hstack((stacked_nc[area], all_nc[sidx][r]))
            del (
                beta_order,
                validation_mask,
                sorted_betas,
                source_sorted_validation_betas,
            )

    # all splits for regression training
    num_train = int(0.8 * num_images)
    num_test = num_images - num_train
    all_splits = get_splits(
        data=target_sorted_validation_betas,
        split_index=0,
        num_splits=num_splits,
        num_per_class_test=num_test,
        num_per_class_train=num_train,
    )

    if pooled:
        subsamp_feats = {}
        if n_source_voxels is None:
            subsamp_feats = features  # use all voxels
        else:
            keep_inds = {}
            for l in ROIS:
                if subsamp_type == 0:  # select a random subset of voxels per ROI
                    n = features[l].shape[1]
                    perm = rng.permutation(n)
                    keep_inds[l] = perm[:n_source_voxels]
                elif subsamp_type == 1:
                    sorted_by_nc = np.argsort(stacked_nc[l])[::-1]
                    keep_inds[l] = sorted_by_nc[:n_source_voxels]
                subsamp_feats[l] = features[l][:, keep_inds[l]]
        log("Doing mapping")
        rsquared_array = traditional_mapping(
            subsamp_feats, target_sorted_validation_betas, all_splits, ROIS, "PLS", 0
        )
    else:
        del features, stacked_nc
        log("Doing mapping")
        by_source_rsquared_array = {}
        rsquared_array = {}
        for sidx, s in enumerate(source_subj):
            by_source_rsquared_array[s] = traditional_mapping(
                stacked_vox[sidx],
                target_sorted_validation_betas,
                all_splits,
                ROIS,
                "PLS",
                0,
            )
        for r, area in enumerate(ROIS):
            rsquared_array[area] = np.mean(
                np.vstack(
                    [
                        by_source_rsquared_array[s][area]
                        for s in by_source_rsquared_array
                    ]
                ),
                axis=0,
            )  # average across source subjects

    # save to local data folder
    log("Saving results")
    h5f = h5py.File(
        RESULTS_PATH
        + "mappings/regression/brain2brain/subj"
        + target_subj
        + "_"
        + hemi
        + "_"
        + target_area
        + "_"
        + source_roi
        + "_to_"
        + target_roi
        + "_"
        + str(num_splits)
        + "splits_"
        + "_subsample_"
        + str(n_source_voxels if n_source_voxels is not None else 0)
        + "voxels_subsamptype"
        + str(subsamp_type)
        + "_pooled"
        + str(pooled)
        + ".hdf5",
        "w",
    )

    for k, v in rsquared_array.items():
        print(str(k))
        h5f.create_dataset(str(k), data=v)
    h5f.close()


if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_subj", type=str)
    parser.add_argument("--hemi", type=str, default="rh")
    parser.add_argument("--target_area", type=str, default="Ventral")
    parser.add_argument("--target_roi", type=str, default="streams_shrink10")
    parser.add_argument("--source_roi", type=str, default="streams_shrink20")
    parser.add_argument("--n_source_voxels", type=int, default=None)
    parser.add_argument(
        "--num_splits", type=int, default=10
    )  # number of splits to average over
    parser.add_argument(
        "--subsamp_type", type=int, default=0
    )  # 0 is random, 1 is highest N voxels per region
    parser.add_argument(
        "--pooled", type=int, default=1
    )  # 0 is fit using each other subject as a source individually, 1 is fit using voxels pooled across all other subjects
    ARGS, _ = parser.parse_known_args()

    main(
        ARGS.target_subj,
        ARGS.hemi,
        ARGS.target_area,
        ARGS.target_roi,
        ARGS.source_roi,
        ARGS.n_source_voxels,
        ARGS.num_splits,
        ARGS.subsamp_type,
        ARGS.pooled,
    )
