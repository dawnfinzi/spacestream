
import numpy as np

from spacestream.core.constants import ROI_NAMES

STREAM_NAMES = ROI_NAMES[-3:]


def prep_smoothness(
    mapping, scaled_target_distances, scaled_source_distances, HVA_idx, by_stream
):
    if HVA_idx is None:
        unit_idx = np.arange(0, len(mapping["winning_idx"]))
    else:  # voxel2voxel HVA_only requires different indexing
        unit_idx = HVA_idx

    if by_stream:
        trimmed_unit_idx_by_stream = {}
        trimmed_winning_idx_by_stream = {}
        for streamx, stream in enumerate(STREAM_NAMES):
            trimmed_unit_idx_by_stream[stream] = unit_idx[
                mapping["winning_roi"] == streamx + 5
            ]  # units actually assigned
            trimmed_winning_idx_by_stream[stream] = mapping["winning_idx"][
                mapping["winning_roi"] == streamx + 5
            ].astype(int)

        matched_target = {}
        matched_source = {}
        for stream in STREAM_NAMES:
            targidx = trimmed_unit_idx_by_stream[stream]
            souridx = trimmed_winning_idx_by_stream[stream]

            temp_selected_target_distances = scaled_target_distances[:, targidx][
                targidx, :
            ]
            temp_selected_source_distances = scaled_source_distances[:, souridx][
                souridx, :
            ]

            # sort by distance on cortical sheet
            ind = np.argsort(temp_selected_target_distances, axis=1)
            matched_target[stream] = np.take_along_axis(
                temp_selected_target_distances, ind, 1
            )  # model unit x dists from other units (ascending order)
            matched_source[stream] = np.take_along_axis(
                temp_selected_source_distances, ind, 1
            )  # corresponding chosen brain unit x dists from other brain units (in order of dist on cortical sheet)
    else:
        trimmed_unit_idx = unit_idx[
            mapping["winning_idx"] != 0
        ]  # units actually assigned
        trimmed_winning_idx = mapping["winning_idx"][
            mapping["winning_idx"] != 0
        ].astype(
            int
        )  # corresponding assignments
        selected_target_distances = scaled_target_distances[:, trimmed_unit_idx][
            trimmed_unit_idx, :
        ]
        selected_source_distances = scaled_source_distances[:, trimmed_winning_idx][
            trimmed_winning_idx, :
        ]

        # sort by distance on cortical sheet
        ind = np.argsort(selected_target_distances, axis=1)
        matched_target = np.take_along_axis(
            selected_target_distances, ind, 1
        )  # model unit x dists from other units (ascending order)
        matched_source = np.take_along_axis(
            selected_source_distances, ind, 1
        )  # corresponding chosen brain unit x dists from other brain units (in order of dist on cortical sheet)

    return matched_target, matched_source