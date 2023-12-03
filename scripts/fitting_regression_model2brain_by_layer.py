import argparse
import os
from datetime import datetime
from typing import List, Optional

import h5py
import nibabel.freesurfer.mghformat as mgh
import numpy as np

from spacestream.core.fit import extract_and_fit
from spacestream.core.paths import BETA_PATH, DATA_PATH, RESULTS_PATH
from spacestream.utils.get_utils import get_indices, get_model_layers
from spacestream.utils.regression_utils import get_splits


def log(to_print: str):
    print(f"\nLOG: {to_print}")


def resolve_layer(layers: List[str], arg_layer: Optional[str]):
    """
    Prefers sherlock SLURM_ARRAY_TASK_ID but will return command line arg if not available
    """
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is not None:
        return layers[int(task_id)]

    return arg_layer


def main(
    subj,
    hemi,
    roi,
    model_name,
    model_layer_string,
    subsample,
    mapping_func,
    CV,
    pretrained,
    reduce_temporal_dims,
    return_weights,
    spatial_weight,
):
    betas = h5py.File(
        BETA_PATH + "datab3fsaverage_subj" + subj + "_" + hemi + "_betas.hdf5",
        "r",
    )

    # indexing
    beta_order, beta_mask, validation_mask = get_indices(subj)
    num_train = int((8 / 9) * (beta_order.shape[0] - np.sum(validation_mask)))
    num_test = int((1 / 9) * (beta_order.shape[0] - np.sum(validation_mask)))

    # get ROI data to pick which voxels to fit
    # get ROI info
    mgh_file = mgh.load(DATA_PATH + "brains/" + hemi + "." + roi + ".mgz")
    streams = mgh_file.get_fdata()[:, 0, 0].astype(int)
    # trim betas
    stream_betas = betas["betas"][:, np.nonzero(streams != 0)[0]]

    # sort betas
    stream_betas = stream_betas[
        np.nonzero(beta_mask)[0], :
    ]  # returns only images with all 3 trials
    indx = beta_order.argsort(axis=0)
    sorted_betas = stream_betas[indx, :]

    del streams, stream_betas

    # all splits for regression training
    num_splits = 10  # average over ten random splits
    all_splits = get_splits(
        data=sorted_betas,
        split_index=0,
        num_splits=num_splits,
        num_per_class_test=num_test,
        num_per_class_train=num_train,
        exclude=validation_mask,
    )

    candidate_layer_strings = get_model_layers(model_name, model_layer_string)
    layer = resolve_layer(candidate_layer_strings, model_layer_string)
    layer_for_filename = layer
    if (
        layer is None
    ):  # edge case where no SLURM task ids but also no layer provided at command line (defaults to full set of layers)
        layer = candidate_layer_strings
        layer_for_filename = "all"

    if return_weights == 1:
        return_weights = True
    else:
        return_weights = False

    rsquared_array = extract_and_fit(
        subj,
        model_name,
        layer,
        subsample,
        mapping_func,
        CV,
        sorted_betas,
        beta_order,
        all_splits,
        pretrained,
        reduce_temporal_dims,
        return_weights,
        spatial_weight,
    )

    # save to local data folder
    stem = (
        RESULTS_PATH
        + "mappings/regression/model2brain/"
        + "subj"
        + subj
        + "_"
        + hemi
        + "_"
        + roi
        + "_"
        + model_name
        + (
            str(spatial_weight)
            if model_name == "spacetorch" or model_name == "spacetorch_supervised"
            else ""
        )
        + (
            str(reduce_temporal_dims)
            if model_name == "slowfast" or model_name == "slowfast_full"
            else ""
        )
        + "_"
        + layer_for_filename
        + "_"
        + mapping_func
        + "_subsample_"
        + str(subsample)
        + "_"
        + str(CV)
        + "CV_"
        + str(pretrained)
        + "pretraining"
    )

    # Save weights
    if return_weights:
        log("Saving weights")
        (rsquared_array, weights) = rsquared_array  # unpack
        weights_h5f = h5py.File(
            stem + "_weights.hdf5",
            "w",
        )
        for k, v in weights.items():
            print(str(k))
            weights_h5f.create_dataset(str(k), data=v)
        weights_h5f.close()
        del weights

    # Save fits
    log("Saving fits")
    h5f = h5py.File(
        stem + "_fits.hdf5",
        "w",
    )
    for k, v in rsquared_array.items():
        print(str(k))
        h5f.create_dataset(str(k), data=v)
    h5f.close()
    del rsquared_array

    log(["All done with layer %s!" % layer])
    log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", type=str)
    parser.add_argument("--hemi", type=str, default="rh")
    parser.add_argument("--roi", type=str, default="streams_shrink10")
    parser.add_argument("--model_name", type=str, default="alexnet")
    parser.add_argument(
        "--model_layer_string", default=None  # type=Optional[str],
    )  # particular layer to evaluate, or None if using SLURM task ids
    parser.add_argument(
        "--subsample", type=int, default=1
    )  # subsample the features randomly (1) or using PCA (2)
    parser.add_argument(
        "--mapping_func", type=str, default="PLS"
    )  # which mapping function to use
    parser.add_argument("--CV", type=int, default=0)  # cross-validated (1) or not (0)
    parser.add_argument(
        "--pretrained", type=int, default=1
    )  # pretrained network (1) or not (0)
    parser.add_argument(
        "--reduce_temporal_dims", type=int, default=0
    )  # if video network, reduce along temporal dimension (1) or not (0)
    parser.add_argument(
        "--return_weights", type=int, default=0
    )  # return regression weights (1) or not(0)
    parser.add_argument(
        "--spatial_weight", type=float, default=1.25
    )  # spatial weight for the spacetorch model
    ARGS, _ = parser.parse_known_args()

    main(
        ARGS.subj,
        ARGS.hemi,
        ARGS.roi,
        ARGS.model_name,
        ARGS.model_layer_string,
        ARGS.subsample,
        ARGS.mapping_func,
        ARGS.CV,
        ARGS.pretrained,
        ARGS.reduce_temporal_dims,
        ARGS.return_weights,
        ARGS.spatial_weight,
    )
