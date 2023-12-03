import argparse
import os
import pickle
from pathlib import Path

import h5py
import numpy as np

from spacestream.analyses.effective_dimensionality import (compute_eigvals,
                                                           effective_dim)
from spacestream.core.feature_extractor import get_features_from_layer
from spacestream.core.paths import RESULTS_PATH
from spacestream.datasets.nsd import nsd_dataloader
from spacestream.utils.get_utils import get_indices, get_model

STREAM_IDX = {"Ventral": 5, "Lateral": 6, "Parietal": 7}


def main(
    supervised,
    hemi,
    roi,
    layer_name,
):
    # all weights, subjects, seeds & streams
    sws = ["0.0", "0.1", "0.25", "0.5", "1.25", "2.5", "25.0"]
    subjects = ["01", "02", "03", "04", "05", "06", "07", "08"]
    seeds = [0, 1, 2, 3, 4]
    streams = ["Ventral", "Lateral", "Parietal"]
    trained = True  # using trained networks only so far

    if supervised:
        stem, model_name = "supervised", "spacetorch_supervised"
    else:
        stem, model_name = "self-supervised", "spacetorch"
    corr_dir = RESULTS_PATH + "mappings/one_to_one/unit2voxel/TDANNs/" + stem
    ED_model_results = {
        "Spatial Weight": [],
        "Seed": [],
        "Stream": [],
        "Subject": [],
        "ED": [],
    }
    ED = np.zeros((len(sws), len(seeds), len(subjects), len(streams)))

    for swx, spatial_weight in enumerate(sws):
        for midx, model_seed in enumerate(seeds):
            model = get_model(model_name, trained, spatial_weight, model_seed)

            for sidx, subj in enumerate(subjects):
                subj_name = "subj" + subj
                print(subj_name)

                # Indexing
                beta_order, _, _ = get_indices(subj)
                subj_stim_idx = np.sort(beta_order)

                # Subject specific images/features
                dataloader = nsd_dataloader(
                    list(subj_stim_idx), video=False, batch_size=128
                )
                features = get_features_from_layer(
                    model,
                    dataloader,
                    layer_name,
                    two_pathway=False,
                    reduction_list=None,
                    batch_size=128,
                    vectorize=True,
                )
                print(features[layer_name].shape)

                # Mapping for stream unit idx
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
                        + "spherical_target_radius_factor1.0_final_unit2voxel_correlation_info.hdf5"
                    )
                )
                mapping = {}
                with h5py.File(mapping_path, "r") as f:
                    keys = f.keys()
                    for k in keys:
                        mapping[k] = f[k][:]

                for streamx, stream in enumerate(streams):
                    ## Get units for stream
                    unit_idx = np.where(mapping["winning_roi"] == STREAM_IDX[stream])[0]

                    eigvals = compute_eigvals(features[layer_name][:, unit_idx])

                    ED[swx, midx, sidx, streamx] = effective_dim(eigvals)

                    ED_model_results["Spatial Weight"].append(spatial_weight)
                    ED_model_results["Seed"].append(model_seed)
                    ED_model_results["Subject"].append(subj)
                    ED_model_results["Stream"].append(stream)
                    ED_model_results["ED"].append(ED[swx, midx, sidx, streamx])

        save_dir = os.path.join(
            RESULTS_PATH,
            "analyses/effective_dim/",
        )
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        context = f"{model_name}_ED_by_stream_{hemi}.pkl"
        fname = os.path.join(save_dir, context)
        pickle.dump(ED, open(fname, "wb"))
        print(f"Saved results to {fname}.")


if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--supervised", type=int, default=0
    )  # 1 is supervised, 0 is self-sup
    parser.add_argument("--hemi", type=str, default="rh")
    parser.add_argument("--roi", type=str, default="ministreams")
    parser.add_argument(
        "--layer_name", type=str, default="base_model.layer4.1"
    )  # model layer

    ARGS, _ = parser.parse_known_args()

    main(
        ARGS.supervised,
        ARGS.hemi,
        ARGS.roi,
        ARGS.layer_name,
    )
