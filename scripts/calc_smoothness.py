import argparse

import deepdish as dd
import h5py
import nibabel.freesurfer.mghformat as mgh
import numpy as np
from scipy import stats
from scipy.spatial import distance_matrix

from spacestream.analyses.smoothness import prep_smoothness
from spacestream.core.constants import ROI_NAMES
from spacestream.core.paths import DATA_PATH, RESULTS_PATH
from spacestream.utils.general_utils import sem
from spacestream.utils.get_utils import get_mapping
from spacestream.utils.mapping_utils import agg_by_distance

STREAM_NAMES = ROI_NAMES[-3:]


def individual_calc(
    source: str,
    target: str,
    combo_type: str,  # options are unit2voxel or voxel2voxel
    hemi: str = "rh",
    roi: str = "ministreams",
    checkpoint: str = "final",
    supervised: int = 0,
    seed: int = 0,
    by_stream: int = 0,
    model_type: str = "TDANN",
):
    # get distance matrix
    dfile = DATA_PATH + "brains/ministreams_" + hemi + "_distances.mat"
    with h5py.File(dfile, "r") as f:
        brain_distances = np.array(f["fullspheredists"])
    brain_distances[np.isnan(brain_distances)] = 0
    scaled_brain_distances = brain_distances / (np.max(brain_distances))
    del brain_distances

    # set up source and target distances
    if combo_type == "voxel2voxel":
        scaled_source_distances = scaled_brain_distances
        scaled_target_distances = scaled_brain_distances
        del scaled_brain_distances

        # get idx for HVA_only
        mgh_file = mgh.load(DATA_PATH + "brains/" + hemi + "." + roi + ".mgz")
        streams = mgh_file.get_fdata()[:, 0, 0].astype(int)
        streams_trim = streams[streams != 0]
        HVA_idx = np.where(
            (streams_trim == 5) | (streams_trim == 6) | (streams_trim == 7)
        )[0]
        # dummy spatial weight
        weight = 0.0
    elif combo_type == "unit2voxel":
        scaled_source_distances = scaled_brain_distances

        # load and scale model distances
        if model_type == "TDANN":
            if supervised:
                coord_path = (
                    DATA_PATH
                    + "models/TDANNs/spacenet_layer4.1_coordinates_supervised_lw0.npz"
                )
            else:
                coord_path = (
                    DATA_PATH + "models/TDANNs/spacenet_layer4.1_coordinates_isoswap_3.npz"
                )
        else:
            coord_path = (
                DATA_PATH
                + "models/MBs/RN"
                + target
                + "/swapopt_swappedon_sine_gratings.npz"
            )
            model_type = "MB" + target

        coordinates = np.load(coord_path)["coordinates"]
        model_distances = distance_matrix(coordinates, coordinates, p=2)  # euclidean
        scaled_model_distances = model_distances / (np.max(model_distances))
        del coordinates, model_distances
        scaled_target_distances = scaled_model_distances
        del scaled_model_distances, scaled_brain_distances

        HVA_idx = None

    if checkpoint == "final":
        ckpt_stem = "final"
    else:
        ckpt_stem = "checkpoint" + checkpoint

    if combo_type == "voxel2voxel":
        subj_name = "subj" + target
    elif combo_type == "unit2voxel":
        weight = target
        subj_name = "subj" + source

    print("Getting mapping for " + model_type)
    mapping = get_mapping(
        subj_name,
        combo_type,
        weight,
        seed,
        supervised,
        hemi,
        ckpt_stem,
        source,
        "ministreams",
        model_type,
    )
    matched_t, matched_s = prep_smoothness(
        mapping, scaled_target_distances, scaled_source_distances, HVA_idx, by_stream
    )
    return matched_t, matched_s


def main(
    combo_type, checkpoint, seed, model_type, supervised, hemi, by_stream
):

    if combo_type == "voxel2voxel":
        target_list = ["01", "02", "03", "04", "05", "06", "07", "08"]
        full_source_list = ["01", "02", "03", "04", "05", "06", "07", "08"]
    elif combo_type == "unit2voxel":
        if model_type == "TDANN":
            target_list = [
                "0.0",
                "0.1",
                "0.25",
                "0.5",
                "1.25",
                "2.5",
                "25.0",
            ]
        else:
            target_list = [
                "18",
                "50",
                "50_v2",
            ]
        full_source_list = ["01", "02", "03", "04", "05", "06", "07", "08"]

    if combo_type == "voxel2voxel":
        slen = 7
    elif combo_type == "unit2voxel":
        slen = 8

    if by_stream:
        rs = np.zeros((len(target_list), slen, 3))
        ps = np.zeros((len(target_list), slen, 3))
        thirddist_rs = np.zeros((len(target_list), slen, 3))
        thirddist_ps = np.zeros((len(target_list), slen, 3))
    else:
        rs = np.zeros((len(target_list), slen))
        ps = np.zeros((len(target_list), slen))
        thirddist_rs = np.zeros((len(target_list), slen))
        thirddist_ps = np.zeros((len(target_list), slen))

    for t_sidx, target in enumerate(target_list):
        print(target)

        if combo_type == "voxel2voxel":
            source_list = full_source_list.copy()
            source_list.remove(target)
        elif combo_type == "unit2voxel":
            source_list = full_source_list

        for s_sidx, source in enumerate(source_list):
            print(source)

            d1, d2 = individual_calc(
                source=source,
                target=target,
                combo_type=combo_type,
                hemi=hemi,
                checkpoint=checkpoint,
                supervised=supervised,
                seed=seed,
                by_stream=by_stream,
                model_type=model_type,
            )

            if by_stream:
                for streamx, stream in enumerate(STREAM_NAMES):
                    num_units = len(d1[stream])
                    corrs = np.empty(num_units)
                    corrs[:] = np.nan
                    third_corrs = np.empty(num_units)
                    third_corrs[:] = np.nan
                    for i in range(num_units):
                        dists = d1[stream][i, :]
                        vals = d2[stream][i, :]
                        poss_idx = np.where((dists > 0.33) & (dists < 0.34))[0]
                        if len(poss_idx) > 0:
                            upto33 = poss_idx[0]
                            # calc correlations
                            tr, tp = stats.pearsonr(
                                dists[0:upto33], vals[0:upto33]
                            )
                            r, p = stats.pearsonr(dists, vals)
                            corrs[i] = r
                            third_corrs[i] = tr
                    rs[t_sidx, s_sidx, streamx] = np.nanmean(corrs)
                    thirddist_rs[t_sidx, s_sidx, streamx] = np.nanmean(
                        third_corrs
                    )
                    print(thirddist_rs[t_sidx, s_sidx, streamx])
            else:
                num_units = len(d1)
                corrs = np.zeros(num_units)
                third_corrs = np.zeros(num_units)
                for i in range(num_units):
                    dists = d1[i, :]
                    vals = d2[i, :]
                    upto33 = np.where((dists > 0.33) & (dists < 0.34))[0][0]
                    # calc correlations
                    tr, tp = stats.pearsonr(dists[0:upto33], vals[0:upto33])
                    r, p = stats.pearsonr(dists, vals)
                    corrs[i] = r
                    third_corrs[i] = tr
                rs[t_sidx, s_sidx] = np.mean(corrs)
                thirddist_rs[t_sidx, s_sidx] = np.mean(third_corrs)

    save_path = (
        RESULTS_PATH
        + "analyses/spatial/"
        + ("brains" if combo_type == "voxel2voxel" else model_type)
        + "/smoothness_calc_"
        + ("by_stream_" if by_stream else "")
        + ("lh_" if hemi == "lh" else "")
        + combo_type
        + ("_supervised" if supervised else "")
        + (("_seed" + str(seed)) if seed > 0 else "")
        + "_correlations_by_unit_" 
        + "ckpt"
        + checkpoint
        + "VAL.hdf"
    )
    smoothness = {}
    smoothness["r"] = rs
    smoothness["thirddist_r"] = thirddist_rs

    #monkey path because of version issue
    np.object = object    
    
    dd.io.save(save_path, smoothness)


if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--combo_type", type=str, default="unit2voxel")
    parser.add_argument("--checkpoint", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_type", type=str, default="TDANN")  # 'TDANN' or 'MB'
    parser.add_argument(
        "--supervised", type=int, default=0
    )  # supervised (1) or simCLR (0) objective
    parser.add_argument("--hemi", type=str, default="rh")
    parser.add_argument(
        "--by_stream", type=int, default=1
    )  # calculate divided up by stream assignment (1) or not (0)
    ARGS, _ = parser.parse_known_args()

    main(
        ARGS.combo_type,
        ARGS.checkpoint,
        ARGS.seed,
        ARGS.model_type,
        ARGS.supervised,
        ARGS.hemi,
        ARGS.by_stream,
    )
