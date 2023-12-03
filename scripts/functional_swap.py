import argparse
from datetime import date
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from numpy.random import default_rng
from torch.utils.data import DataLoader

from spacestream.core.feature_extractor import get_features_from_layer
from spacestream.core.paths import DATA_PATH
from spacestream.core.swapopt import Swapper
from spacestream.datasets.sine_gratings import SineResponses, sine_dataloader
from spacestream.utils.array_utils import flatten
from spacestream.utils.general_utils import load_config_from_yaml
from spacestream.utils.get_utils import get_model

rng = default_rng()

TASKS = ["categorization", "action", "detection"]
MODEL_INFO = {
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
        "device": {
            "categorization": "cuda",
            "action": "cuda",
            "detection": "cuda",
        },
        "video": {
            "categorization": False,
            "action": True,
            "detection": False,
        },
        "slowfast_alpha": 8,
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
        "device": {
            "categorization": "cuda",
            "action": "cuda",
            "detection": "cpu",
        },
        "video": {
            "categorization": False,
            "action": True,
            "detection": False,
        },
        "slowfast_alpha": 4,
    },
}


def get_sine_responses(
    model: nn.Module,
    dataloader: DataLoader,
    batch_size: int = 32,
    layer: str = "layer4.1",
    device: str = "cpu",
    video: bool = False,
) -> SineResponses:
    model = model.to(device)
    if video:  # can't return inputs and labels, features only
        features = get_features_from_layer(
            model,
            dataloader,
            layer,
            batch_size=batch_size,
            two_pathway=True,
            reduction_list=[2],
        )
        return flatten(features[layer])
    else:
        features, _, labels = get_features_from_layer(
            model,
            dataloader,
            layer,
            batch_size=batch_size,
            return_inputs_and_labels=True,
        )
        return SineResponses(features[layer], labels)


def save_sine_grating_features_and_positions(base):
    today = date.today().strftime("%d-%m-%Y")
    num_units_per = 6388

    model, final_features, chosen_indices = {}, {}, {}
    model_info = MODEL_INFO[str(base)]
    for task in TASKS:
        model[task] = get_model(model_info["model_name"][task].lower())

        responses = get_sine_responses(
            model[task],
            sine_dataloader(
                None,
                model_info["video"][task],
                32,
                model_info["model_name"][task],
                model_info["slowfast_alpha"],
            ),
            32,
            model_info["layer_name"][task],
            model_info["device"][task],
            model_info["video"][task],
        )
        if task != "action":
            responses = responses._data.data

        non_zero = np.where(responses.any(axis=0))[
            0
        ]  # only pick from units that are actually responding
        subset = rng.choice(len(non_zero), size=num_units_per, replace=False)

        chosen_indices[task] = non_zero[subset]
        final_features[task] = responses[:, chosen_indices[task]]
    total_feats = np.hstack(final_features.values())
    del final_features

    chosen_save_path = Path(
        DATA_PATH
        + "models/MBs/RN"
        + str(base)
        + "/swapopt/chosen_indices-"
        + today
        + ".npz"
    )
    np.savez(
        chosen_save_path,
        categorization=chosen_indices["categorization"],
        action=chosen_indices["action"],
        detection=chosen_indices["detection"],
    )

    # Assign a subset of positions
    coord_path = DATA_PATH + "models/TDANNs/spacenet_layer4.1_coordinates_isoswap_3.npz"
    coords = np.load(coord_path)["coordinates"]
    positions_used = rng.choice(
        coords.shape[0],
        size=num_units_per * 3,
        replace=False,
    )
    positions = coords[positions_used]
    position_save_path = Path(
        DATA_PATH
        + "models/MBs/RN"
        + str(base)
        + "/swapopt/random_initial_positions"
        + "-"
        + today
        + ".npz"
    )
    np.savez(position_save_path, coordinates=positions)

    # save out responses to sine gratings just in case
    feature_path = Path(
        DATA_PATH
        + "models/MBs/RN"
        + str(base)
        + "/swapopt/testing_random_subset_random_positions"
        + "-"
        + today
        + ".hdf5"
    )
    save_feats = {}
    save_feats["random_pos"] = total_feats
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(feature_path, "w") as f:
        for k, v in save_feats.items():
            f.create_dataset(k, data=v)
    return today


def plot_metrics(swapper: Swapper, save_path: Path):
    line_kwargs = {"lw": 1.5, "c": "k"}

    fig, axes = plt.subplots(figsize=(10, 5), ncols=2)
    axes[0].plot(swapper.metrics.num_swaps, **line_kwargs)
    axes[1].plot(swapper.metrics.losses, **line_kwargs)

    axes[0].set_ylabel("Number of Swaps")
    axes[1].set_ylabel("Neighborhood Loss")

    fig.savefig(save_path, dpi=100, bbox_inches="tight", facecolor="white")


def main(
    generate_features,
    data_stem,
    date,
    layer,
    dataset_name,
    neighborhood_width,
    base,
):
    if generate_features:
        date = save_sine_grating_features_and_positions(base)

    if date:
        data_stem = data_stem + "-" + date

    stem = DATA_PATH + "models/MBs/RN" + str(base) + "/swapopt/" + data_stem
    config = load_config_from_yaml(Path(stem + ".yaml"))
    feature_path = Path(stem + ".hdf5")

    swapper = Swapper(
        config,
        feature_path=feature_path,
        layer=layer,
        dataset_name=dataset_name,
        neighborhood_width=neighborhood_width,
    )

    # swapper can get blocked if running it would overwrite existing position files
    if not swapper.blocked:
        print("Swapping")
        swapper.swap()

        print("Saving positions back")
        swapper.save_positions()

        print("Saving metrics plot")
        save_path = Path(f"{stem}_metrics.png")
        plot_metrics(swapper, save_path)

        print("All done!")


if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate_features", type=int, default=1
    )  # generate new features and positions (1) or not (0)
    parser.add_argument(
        "--data_stem", type=str, default="testing_random_subset_random_positions"
    )
    parser.add_argument(
        "--date",
        type=str,
        required=False,
    )
    parser.add_argument("--layer", type=str, default="random_pos")
    parser.add_argument("--dataset_name", type=str, default="sine_gratings")
    parser.add_argument("--neighborhood_width", type=int, default=4.545454)
    parser.add_argument(
        "--base", type=int, default=18
    )  # base architecture (resnet18 or resnet50)
    ARGS, _ = parser.parse_known_args()

    main(
        ARGS.generate_features,
        ARGS.data_stem,
        ARGS.date,
        ARGS.layer,
        ARGS.dataset_name,
        ARGS.neighborhood_width,
        ARGS.base,
    )
