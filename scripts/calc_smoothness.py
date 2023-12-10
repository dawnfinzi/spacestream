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
    num_bins: int = 100,
    supervised: bool = False,
    aggregate: bool = True,
    corr_type: int = 1,
    seed: int = 0,
    by_stream: int = 0,
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
    elif combo_type == "unit2voxel":
        scaled_source_distances = scaled_brain_distances

        # load and scale model distances
        if supervised:
            coord_path = (
                DATA_PATH
                + "models/TDANNs/spacenet_layer4.1_coordinates_supervised_lw0.npz"
            )
        else:
            coord_path = (
                DATA_PATH + "models/TDANNs/spacenet_layer4.1_coordinates_isoswap_3.npz"
            )
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

    mapping = get_mapping(
        subj_name=subj_name,
        mapping_type=combo_type,
        spatial_weight=weight,
        model_seed=seed,
        supervised=supervised,
        hemi=hemi,
        checkpoint=ckpt_stem,
        source_subj=source,
    )
    matched_t, matched_s = prep_smoothness(
        mapping, scaled_target_distances, scaled_source_distances, HVA_idx, by_stream
    )
    if corr_type == 2:
        return matched_t, matched_s
    distances = np.ravel(matched_t[:, 1:])
    del matched_t
    values = np.ravel(matched_s[:, 1:])
    del matched_s
    if aggregate:
        means, _, bin_edges = agg_by_distance(distances, values, num_bins=num_bins)
        return means, bin_edges
    else:  # return all raw distances and values instead
        if corr_type == 1:
            # sort distances and values in order of increasing distance
            dist_sort_ind = np.argsort(distances)
            sorted_distances = distances[dist_sort_ind]
            sorted_values = values[dist_sort_ind]
            return sorted_distances, sorted_values


def main(
    combo_type, checkpoint, num_bins, calc_type, seed, supervised, hemi, by_stream
):
    aggregate = False if calc_type else True

    if by_stream:
        assert calc_type == 2, "combo not implemented yet!"

    if combo_type == "voxel2voxel":
        target_list = ["01", "02", "03", "04", "05", "06", "07", "08"]
        full_source_list = ["01", "02", "03", "04", "05", "06", "07", "08"]
    elif combo_type == "unit2voxel":
        target_list = [
            "0.0",
            "0.1",
            "0.25",
            "0.5",
            "1.25",
            "2.5",
            "25.0",
        ]
        full_source_list = ["01", "02", "03", "04", "05", "06", "07", "08"]

    if aggregate:
        by_target_means = np.zeros((num_bins, len(target_list)))
        by_target_spread = np.zeros((num_bins, len(target_list)))
        by_target_bins = np.zeros((num_bins + 1, len(target_list)))
    else:
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

        full_means = np.zeros((num_bins, len(source_list)))
        full_bin_edges = np.zeros((num_bins + 1, len(source_list)))

        for s_sidx, source in enumerate(source_list):
            print(source)

            d1, d2 = individual_calc(
                source=source,
                target=target,
                combo_type=combo_type,
                hemi=hemi,
                checkpoint=checkpoint,
                num_bins=num_bins,
                supervised=(True if supervised else False),
                aggregate=aggregate,
                corr_type=calc_type,
                seed=seed,
                by_stream=by_stream,
            )

            if aggregate:
                full_means[:, s_sidx] = d1
                full_bin_edges[:, s_sidx] = d2
            else:
                if calc_type == 1:
                    # cutoffs
                    upto33 = np.where((d1 > 0.33) & (d1 < 0.34))[0][0]
                    # calc correlations
                    tr, tp = stats.pearsonr(d1[0:upto33], d2[0:upto33])
                    r, p = stats.pearsonr(d1, d2)
                    rs[t_sidx, s_sidx] = r
                    ps[t_sidx, s_sidx] = p
                    thirddist_rs[t_sidx, s_sidx] = tr
                    thirddist_ps[t_sidx, s_sidx] = tp
                elif calc_type == 2:
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

        if aggregate:
            by_target_means[:, t_sidx] = np.mean(full_means, axis=1)
            by_target_spread[:, t_sidx] = sem(full_means, axis=1)
            by_target_bins[:, t_sidx] = np.mean(full_bin_edges, axis=1)

            del full_means, full_bin_edges

    if calc_type == 0:
        save_path = (
            RESULTS_PATH
            + "analyses/spatial/"
            + ("brains" if combo_type == "voxel2voxel" else "TDANNs")
            + "/smoothness_calc_"
            + ("lh_" if hemi == "lh" else "")
            + combo_type
            + ("_supervised" if supervised else "")
            + (("_seed" + str(seed)) if seed > 0 else "")
            + "_means_for_plotting_"
            + str(num_bins)
            + "bins_ckpt"
            + checkpoint
            + ".hdf"
        )

        smoothness = {}
        smoothness["means"] = by_target_means
        smoothness["spread"] = by_target_spread
        smoothness["bin_edges"] = by_target_bins
    else:  # correlations
        save_path = (
            RESULTS_PATH
            + "analyses/spatial/"
            + ("brains" if combo_type == "voxel2voxel" else "TDANNs")
            + "/smoothness_calc_"
            + ("by_stream_" if by_stream else "")
            + ("lh_" if hemi == "lh" else "")
            + combo_type
            + ("_supervised" if supervised else "")
            + (("_seed" + str(seed)) if seed > 0 else "")
            + "_correlations_"
            + ("by_unit_" if calc_type == 2 else "")
            + "ckpt"
            + checkpoint
            + ".hdf"
        )
        smoothness = {}
        smoothness["r"] = rs
        smoothness["thirddist_r"] = thirddist_rs
        if calc_type == 1:
            smoothness["p"] = ps
            smoothness["thirddist_p"] = thirddist_ps

    dd.io.save(save_path, smoothness)


if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--combo_type", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--num_bins", type=int, default=100)
    parser.add_argument(
        "--calc_type", type=int, default=0
    )  # 0 is agg_by_dist, 1 is correlation on raw vals, 2 is also correlation but for each individual unit and then avg'ed
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--supervised", type=int, default=0
    )  # supervised (1) or simCLR (0) objective
    parser.add_argument("--hemi", type=str, default="rh")
    parser.add_argument(
        "--by_stream", type=int, default=0
    )  # calculate divided up by stream assignment (1) or not (0)
    ARGS, _ = parser.parse_known_args()

    main(
        ARGS.combo_type,
        ARGS.checkpoint,
        ARGS.num_bins,
        ARGS.calc_type,
        ARGS.seed,
        ARGS.supervised,
        ARGS.hemi,
        ARGS.by_stream,
    )
