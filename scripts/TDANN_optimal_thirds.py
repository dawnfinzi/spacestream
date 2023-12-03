import argparse
from itertools import permutations

import deepdish as dd
import numpy as np
from geopandas import GeoSeries
from shapely.affinity import rotate
from shapely.geometry import Point, Polygon

from spacestream.core.paths import DATA_PATH, RESULTS_PATH
from spacestream.utils.get_utils import get_mapping


def main(supervised, hemi, topXpercent, checkpoint):
    # calculate a proxy of "percent correct assignment by stream" to mirror the calculations for the task models
    # allows for division of simulated cortical sheet into 3 (any order/rotation), pinned at center
    subj_list = ["01", "02", "03", "04", "05", "06", "07", "08"]
    seeds = [0, 1, 2, 3, 4]
    sw_list = ["0.0", "0.1", "0.25", "0.5", "1.25", "2.5", "25.0"]

    # get model distance matrix
    if supervised:
        coord_path = (
            DATA_PATH + "models/TDANNs/spacenet_layer4.1_coordinates_supervised_lw0.npz"
        )
    else:
        coord_path = (
            DATA_PATH + "models/TDANNs/spacenet_layer4.1_coordinates_isoswap_3.npz"
        )
    coordinates = np.load(coord_path)["coordinates"]
    shift = coordinates - 5  # center at 0
    coord_points = GeoSeries(map(Point, zip(shift[:, 0], shift[:, 1])))

    # set up the candidate geoms to rotate
    center = (0, 0)
    A = (5, 2.5)
    B = (0, -5)
    C = (-5, 2.5)
    coords_A = [center, A, (5, 5), (-5, 5), C]
    poly_A = Polygon(coords_A)
    coords_B = [center, A, (5, -5), B]
    poly_B = Polygon(coords_B)
    coords_C = [center, B, (-5, -5), C]
    poly_C = Polygon(coords_C)

    data = np.zeros((len(seeds), len(subj_list), len(sw_list), 3))
    total_max = np.zeros((len(seeds), len(subj_list), len(sw_list), 3))
    rotation = np.zeros((len(seeds), len(subj_list), len(sw_list)))

    perm = permutations([5, 6, 7])
    perm_list = list(perm).copy()
    for seedx, model_seed in enumerate(seeds):
        for sidx, subj in enumerate(subj_list):
            subj_name = "subj" + subj

            for swidx, spatial_weight in enumerate(sw_list):
                mapping = get_mapping(
                    subj_name=subj_name,
                    spatial_weight=spatial_weight,
                    model_seed=model_seed,
                    supervised=supervised,
                    hemi=hemi,
                    checkpoint=checkpoint,
                )
                by_fives = np.percentile(
                    mapping["winning_test_corr"][mapping["winning_test_corr"] != 0],
                    np.arange(0, 100, 5),
                )
                # determine top X% only
                topX = int((100 - topXpercent) / 5)
                mapping["winning_roi"][
                    np.where(mapping["winning_test_corr"] < by_fives[topX])[0]
                ] = 0

                full_acc = np.zeros((36 * 2, 3))
                for rot in range(0, 360, 5):
                    new_poly_A = rotate(poly_A, rot)
                    new_poly_B = rotate(poly_B, rot)
                    new_poly_C = rotate(poly_C, rot)

                    poly_class_opts = np.zeros((6, len(coord_points)))
                    for pidx, point in enumerate(coord_points):
                        for permidx, perm_case in enumerate(perm_list):
                            if point.within(new_poly_A):
                                poly_class_opts[permidx, pidx] = perm_case[0]
                            elif point.within(new_poly_B):
                                poly_class_opts[permidx, pidx] = perm_case[1]
                            elif point.within(new_poly_C):
                                poly_class_opts[permidx, pidx] = perm_case[2]

                    idx = {}
                    for i in range(6):
                        idx[str(i + 1)] = np.where(
                            poly_class_opts[i] == mapping["winning_roi"]
                        )[0]

                    acc = np.zeros((6, 3))

                    for i in range(6):
                        key = idx[str(i + 1)]
                        acc[i, 0] = sum(mapping["winning_roi"][key] == 5) / sum(
                            mapping["winning_roi"] == 5
                        )
                        acc[i, 1] = sum(mapping["winning_roi"][key] == 6) / sum(
                            mapping["winning_roi"] == 6
                        )
                        acc[i, 2] = sum(mapping["winning_roi"][key] == 7) / sum(
                            mapping["winning_roi"] == 7
                        )

                    best_of_six = np.argmax(np.mean(acc, axis=1))
                    full_acc[rot // 5, :] = acc[best_of_six, :]

                best = np.argmax(np.mean(full_acc, axis=1))
                rotation[seedx, sidx, swidx] = best

                data[seedx, sidx, swidx, :] = full_acc[best, :]
                total_max[seedx, sidx, swidx, :] = np.max(full_acc, axis=0)

    save_path = (
        RESULTS_PATH
        + "analyses/spatial/optimal_rgb_percentages_top"
        + str(topXpercent)
        + "_"
        + hemi
        + ("_supervised" if supervised else "")
        + ("_" + str(checkpoint) if checkpoint != "final" else "")
        + ".hdf"
    )
    TDANN_on_brain = {}
    TDANN_on_brain["max_by_avg"] = data
    TDANN_on_brain["max_each_stream"] = total_max
    TDANN_on_brain["rotation"] = rotation

    dd.io.save(save_path, TDANN_on_brain)


if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--supervised", type=int, default=0
    )  # 1 is supervised, 0 is self-sup
    parser.add_argument("--hemi", type=str, default="rh")
    parser.add_argument(
        "--topXpercent", type=int, default=100
    )  # what percent of units to use (by test correlation)
    parser.add_argument("--checkpoint", type=str, default="final")

    ARGS, _ = parser.parse_known_args()

    main(
        ARGS.supervised,
        ARGS.hemi,
        ARGS.topXpercent,
        ARGS.checkpoint,
    )
