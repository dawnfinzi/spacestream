import argparse
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from spacestream.analyses.hvm_fitter import HvmFitter
from spacestream.core.paths import HVM_PATH
from spacestream.utils.general_utils import log

STREAM_IDX = {"Ventral": 5, "Lateral": 6, "Parietal": 7}


def main(
    task,
    stream,
    model,
    trained,
    spatial_weight,
    model_seed,
    subj,
    hemi,
    roi,
    checkpoint,
    pca,
    sampling,
    var_splits,
):

    log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if model == "spacetorch_supervised":
        supervised = True
    else:
        supervised = False

    trained = True if trained == 1 else False
    pca = True if pca == 1 else False

    layer_name = "base_model.layer4.1"
    subj_name = "subj" + subj

    ## Retrieve mapping
    # current defaults
    space = "fsaverage"
    algorithm = "hungarian"
    if supervised:
        corr_dir = (
            Path(
                "/oak/stanford/groups/kalanit/biac2/kgs/projects/Dawn/NSD/results/spacetorch/"
            )
            / space
            / "neighborhood/supervised"
            / algorithm
        )
    else:
        corr_dir = (
            Path(
                "/oak/stanford/groups/kalanit/biac2/kgs/projects/Dawn/NSD/results/spacetorch/"
            )
            / space
            / "neighborhood"
            / algorithm
        )
    mapping_path = Path(
        corr_dir
        / (
            "spatial_weight"
            + str(spatial_weight)
            + (("_seed" + str(model_seed)) if model_seed > 0 else "")
        )
        / subj_name
        / (
            hemi
            + "_"
            + roi
            + "_CV_HVA_only_radius5.0_max_iters100_constant_radius_2.0dist_cutoff_constant_dist_cutoff_"
            + "spherical_target_radius_factor1.0_"
            + checkpoint
            + "_unit2voxel_correlation_info.hdf5"
        )
    )
    mapping = {}
    with h5py.File(mapping_path, "r") as f:
        keys = f.keys()
        for k in keys:
            mapping[k] = f[k][:]

    ## Get units for stream
    unit_idx = np.where(mapping["winning_roi"] == STREAM_IDX[stream])[0]

    if sampling > 0:
        n = 1000 * sampling
        unit_idx = unit_idx[
            np.argsort(mapping["winning_test_corr"][unit_idx])[::-1][0:n]
        ]

    ## Test transfer
    if var_splits == 0:
        train_test_splits_file = HVM_PATH + "test_hvm_splits_v3v6.npz"
    else:
        train_test_splits_file = HVM_PATH + "test_hvm_splits.npz"
    fitter = HvmFitter(
        train_test_splits_file, model, trained, spatial_weight, model_seed
    )
    log("Starting " + task + " estimation")
    fitter.fit(
        task,
        layer_name,
        spatial_weight,
        model_seed,
        hemi,
        subj_name,
        stream,
        unit_idx,
        pca,
        sampling,
        var_splits,
    )

    log("All done!")
    log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="size")
    parser.add_argument(
        "--stream", type=str, default="Ventral"
    )  # "Ventral", "Lateral", or "Parietal"
    parser.add_argument(
        "--model", type=str, default="spacetorch"
    )  # spacetorch or spacetorch_supervised
    parser.add_argument("--trained", type=int, default=1)  # 1 is trained, 0 is random
    parser.add_argument("--spatial_weight", type=float, default=0.5)
    parser.add_argument("--model_seed", type=int, default=0)

    parser.add_argument("--subj", type=str, default="01")
    parser.add_argument("--hemi", type=str, default="rh")
    parser.add_argument("--roi", type=str, default="ministreams")
    parser.add_argument("--checkpoint", type=str, default="final")

    parser.add_argument("--pca", type=int, default=1)  # 1 is do_pca, 0 is don't do
    parser.add_argument(
        "--sampling", type=int, default=0
    )  # unit sampling type: 0 is all, 1 is top 1k by test_corr values, 2 is top 2k etc
    parser.add_argument(
        "--var_splits", type=int, default=1
    )  # 0 is base v3v6 only version, 1 is using splits where trained on v0 and v3 and tested on v6

    ARGS, _ = parser.parse_known_args()

    main(
        ARGS.task,
        ARGS.stream,
        ARGS.model,
        ARGS.trained,
        ARGS.spatial_weight,
        ARGS.model_seed,
        ARGS.subj,
        ARGS.hemi,
        ARGS.roi,
        ARGS.checkpoint,
        ARGS.pca,
        ARGS.sampling,
        ARGS.var_splits,
    )
