import logging
import os
from pathlib import Path
from typing import List, Union
from tqdm import tqdm

import h5py
import numpy as np
import open_clip
import pandas as pd
import scipy.io
import torch
import torchvision.models as models
from transformers import DetrForObjectDetection
from ssd.config import cfg
from ssd.modeling.detector import build_detection_model
from ssd.utils.checkpoint import CheckPointer

from spacestream.core.constants import (MATCHING_SLOWFAST_LAYERS, N_REPEATS,
                                        RESNET18_LAYERS, RESNET50_LAYERS,
                                        RESNET101_LAYERS, SLOWFAST_LAYERS,
                                        SPACETORCH_LAYERS, SW_PATH_STR_MAPPING,
                                        X3D_LAYERS)
from spacestream.core.paths import BETA_PATH, DATA_PATH, RESULTS_PATH
from spacestream.models.spatial_resnet import SpatialResNet18
from spacestream.utils.slowfast_utils import load_slowfast_model

# set up logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)-15s %(levelname)s:%(message)s"
)
logger = logging.getLogger(__name__)


def get_ckpts(spatial_weight: float, seed: int, supervised: bool):
    sw = SW_PATH_STR_MAPPING[spatial_weight]

    if supervised:
        if spatial_weight == 0.0:
            task_stem = "vissl_checkpoints/supervised_resnet18_"
        else:
            task_stem = "vissl_checkpoints/supervised_spatial_resnet18_swappedon_SineGrating2019_"
    else:
        task_stem = (
            "relu_rescue/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_"
        )

        if (
            spatial_weight == 0.1
        ):  # seed order is swapped (just based on when it finished running)
            if seed == 0:
                seed = 1
            elif seed == 1:
                seed = 0

    seed_str = "seed_" + str(seed) + "_" if seed > 0 else ""

    ckpt_path = (
        f"{DATA_PATH}"
        "models/TDANNs/model_checkpoints/"
        f"{task_stem}{sw}{seed_str}checkpoints/"
        "model_final_checkpoint_phase199.torch"
    )

    return ckpt_path


def get_eval_ckpts(spatial_weight: float, seed: int, supervised: bool):
    # this is different than traditional get_ckpts in get_utils because we need the linear eval weights for the simCLR models

    sw = SW_PATH_STR_MAPPING[spatial_weight]

    if supervised:
        if spatial_weight == 0.0:
            task_stem = "vissl_checkpoints/supervised_resnet18_"
        else:
            task_stem = "vissl_checkpoints/supervised_spatial_resnet18_swappedon_SineGrating2019_"
    else:
        task_stem = "relu_rescue/linear_eval/relu_rescue__simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_"

        if (
            spatial_weight == 0.1
        ):  # seed order is swapped (just based on when it finished running)
            if seed == 0:
                seed = 1
            elif seed == 1:
                seed = 0

    seed_str = "seed_" + str(seed) + "_" if seed > 0 else ""

    if supervised:
        ckpt_path = (
            f"{DATA_PATH}"
            "models/TDANNs/model_checkpoints/"
            f"{task_stem}{sw}{seed_str}checkpoints/"
            "model_final_checkpoint_phase199.torch"
        )
    else:
        ckpt_path = (
            f"{DATA_PATH}"
            "models/TDANNs/model_checkpoints/"
            f"{task_stem}{sw}{seed_str}linear_eval_checkpoints/"
            "model_final_checkpoint_phase27.torch"
        )

    return ckpt_path


def get_indices(subj: str, shared: bool = False):
    order = scipy.io.loadmat(BETA_PATH + "datab3nativesurface_subj" + subj)
    data = pd.read_csv(
        DATA_PATH + "brains/ppdata/subj" + subj + "/behav/responses.tsv", sep="\t"
    )
    expdesign = scipy.io.loadmat(DATA_PATH + "brains/nsd_expdesign.mat")

    # 73KIDs
    all_ids = np.array(data["73KID"])
    vals, _, count = np.unique(all_ids, return_counts=True, return_index=True)
    which_reps = vals[count == N_REPEATS]
    mask_3reps = np.isin(all_ids, which_reps)
    id_nums_3reps = np.array(data["73KID"])[mask_3reps]
    rep_vals = np.unique(id_nums_3reps)  # sorted version of beta order

    # how the betas are ordered (using COCO 73K id numbers)
    beta_order_in_73Kids = all_ids[order["allixs"][0] - 1]
    beta_mask = np.isin(
        beta_order_in_73Kids, id_nums_3reps
    )  # mask just those images with 3 repeats in the cases where subjects did not finish the full experiment
    beta_order = (
        beta_order_in_73Kids[beta_mask] - 1
    )  # -1 to convert from matlab to python indexing

    if shared:  # use shared 515 across all subjects for val mask
        val_ids = h5py.File(
            DATA_PATH + "brains/shared_73Kids.h5",
            "r",
        )
        sharedix = np.array(val_ids["shared_515"])
    else:
        # shared (i.e. validation) IDS (but include all potential shared reps for the subj, not min across subjs)
        sharedix = expdesign["sharedix"][0]
    validation_mask = np.isin(rep_vals, sharedix)

    return beta_order, beta_mask, validation_mask

def get_betas(subj, hemi, roi_info, chunk_size=1000):
    """
    Get betas (efficiently) for a given ROI, subject and hemi.
    """
    beta_order, beta_mask, _ = get_indices(subj, shared=True)
    indx = beta_order.argsort(axis=0)

    roi_cols = np.nonzero(roi_info != 0)[0]
    mask_rows = np.nonzero(beta_mask)[0]

    with h5py.File(
        BETA_PATH + f"datab3fsaverage_subj{subj}_{hemi}_betas.hdf5", "r"
    ) as f:
        betas_data = f["betas"]  # shape: [all_images, all_voxels]

        # Allocate output array
        n_rows = len(mask_rows)
        n_cols = len(roi_cols)
        stream_betas = np.empty((n_rows, n_cols), dtype=np.float32)

        for i in tqdm(range(0, n_rows, chunk_size),
                      desc="Loading betas in chunks",
                      total=(n_rows // chunk_size + int(n_rows % chunk_size != 0))):
            batch_idx = mask_rows[i : i + chunk_size]
            # shape of betas_data[batch_idx] => (chunk_size, all_voxels)
            # index columns by roi_cols => (chunk_size, n_cols)
            stream_betas[i : i + chunk_size, :] = betas_data[batch_idx][:, roi_cols]

    sorted_betas = stream_betas[indx, :]
    return sorted_betas


def get_model(
    model_name: str,
    pretrained: bool = True,
    spatial_weight: float = 1.25,
    model_seed: int = 0,
):
    if model_name.lower() == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif model_name.lower() == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    elif model_name.lower() == "resnet101":
        model = models.resnet101(pretrained=pretrained)
    elif model_name.lower() == "slowfast" or model_name.lower() == "slowfast_full":
        model = torch.hub.load(
            "facebookresearch/pytorchvideo", "slowfast_r50", pretrained=pretrained
        )
    elif model_name.lower() == "slowfast18":
        fname = os.path.join(
            DATA_PATH, "models/MBs/RN18/slowfast/slow_fast_resnet18_kinetics.pt"
        )
        model = load_slowfast_model(
            "slow_fast_resnet18",
            trained=pretrained,
            model_path=fname,
            alpha=8,
            f2s_mul=2,
            fusion_stride=[8, 1, 1],
        )
    elif model_name.lower() == "x3d_xs" or model_name.lower() == "x3d_s" or model_name.lower() == "x3d_m":
        model = torch.hub.load(
            "facebookresearch/pytorchvideo", model_name, pretrained=pretrained
        )
    elif model_name.lower() == "spacetorch" or model_name.lower() == "spacetorch_supervised":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if model_name == "spacetorch":
            supervised = False
        else:
            supervised = True
        weight_path = get_ckpts(float(spatial_weight), model_seed, supervised)
        ckpt = torch.load(weight_path, map_location=torch.device(device))
        model_params = ckpt["classy_state_dict"]["base_model"]["model"]["trunk"]
        model = SpatialResNet18()
        if pretrained:
            model.load_state_dict(model_params, strict=False)
    elif model_name.lower() == "faster_rcnn":
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    elif model_name.lower() == "ssd":
        config_file = (
            DATA_PATH
            + "models/MBs/RN18/SSD/configs/resnet18_ssd300_voc0712_nopre_300k_seed"
            + str(model_seed)
            + ".yaml"
        )
        cfg.merge_from_file(config_file)
        ckpt = DATA_PATH + "models/MBs/RN18/SSD/checkpoints/model_final.pth"
        device = torch.device(cfg.MODEL.DEVICE)
        model = build_detection_model(cfg)
        model = model.to(device)
        checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
        checkpointer.load(ckpt, use_latest=ckpt is None)
    elif model_name.lower() == "open_clip_rn50":
        # Load OpenCLIP model (RN50 backbone)
        model, _, _ = open_clip.create_model_and_transforms('RN50', pretrained='openai')
        model = model.visual # vision encoder only
    elif model_name.lower() == "open_clip_vit_b_32":
        # Load OpenCLIP model (ViT-B-32 backbone)
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        model = model.visual # vision encoder only
    elif model_name.lower() == "convnext_tiny": # RN50 equivalent
        model = models.convnext_tiny(weights='DEFAULT') # Imagenet1k_V1 weights
    elif model_name.lower() == "detr_rn50": # DETR trained with RN50 backbone
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    return model


def get_model_layers(model_name: str, model_layer_strings: Union[str, List[str]]):
    # This utility function which gets the layer names for a given model is 
    # currently only used for traditional linear mapping analysis (fit.py in core).
    if model_layer_strings is not None:
        if isinstance(model_layer_strings, str):
            if "," in model_layer_strings:
                model_layer_strings = model_layer_strings.split(",")
            else:
                model_layer_strings = [model_layer_strings]
    else:
        if model_name == "resnet18":
            model_layer_strings = RESNET18_LAYERS
        elif model_name == "resnet50":
            model_layer_strings = RESNET50_LAYERS
        elif model_name == "resnet101":
            model_layer_strings = RESNET101_LAYERS
        elif model_name == "slowfast":
            model_layer_strings = SLOWFAST_LAYERS
        elif model_name == "slowfast_full":
            model_layer_strings = MATCHING_SLOWFAST_LAYERS
        elif model_name == "x3d_xs" or model_name == "x3d_s" or model_name == "x3d_m":
            model_layer_strings = X3D_LAYERS
        elif model_name == "spacetorch" or model_name == "spacetorch_supervised":
            model_layer_strings = SPACETORCH_LAYERS
        elif model_name == "faster_RCNN":
            model_layer_strings = ["backbone.body.layer4.1"]  # one layer so far

    model_layer_strings = [
        item
        for sublist in [
            [item] if type(item) is not list else item for item in model_layer_strings
        ]
        for item in sublist
    ]

    return model_layer_strings


def get_mapping(
    subj_name,
    mapping_type="unit2voxel",
    spatial_weight="0.25",
    model_seed=0,
    supervised=0,
    hemi="rh",
    checkpoint="final",
    source_subj="01",  # if voxel2voxel mapping
    roi="ministreams",
    model_type="TDANN",  # options are "TDANN", "MB18", "MB50", and "MB50_v2"
    vit_control=False,  # if using vit control model
    random_pos_control=False,  # if using random position control
):
    # setup (ugly but backwards compatible)
    if mapping_type == "unit2voxel":
        if model_type == "TDANN":
            sub_folder = (
                ("/supervised"
                if supervised
                else "/self-supervised")
                + "/spatial_weight"
                + str(spatial_weight)
                + ("_seed" + str(model_seed) if model_seed > 0 else "")
            )
            mapping_stem = "_CV_HVA_only_radius5.0_max_iters100_constant_radius_2.0dist_cutoff_constant_dist_cutoff_spherical_target_radius_factor1.0"
            stem = (
                "supervised" if supervised else "self-supervised"
            )  # reassign to match MB structure
        else:
            sub_folder = "/RN" + ("18" if "18" in model_type else "50_v2" if "50_v2" in model_type else "50")
            model_type = "MB"
            mapping_stem = "_CV_HVA_only_matched_random_subsample_max_iters100"

        corr_dir = (
            RESULTS_PATH + "mappings/one_to_one/unit2voxel/" + model_type + "s" + sub_folder
        )
    elif mapping_type == "voxel2voxel":
        corr_dir = RESULTS_PATH + "mappings/one_to_one/voxel2voxel/target_" + subj_name
        mapping_stem = (
            "_HVA_only_radius5_max_iters100_constant_radius_2.0dist_cutoff_constant_dist_cutoff_spherical"
            + "_CV_seed" + str(model_seed)
        )

    mapping_path = (
        corr_dir
        + "/"
        + ("vit_control/" if vit_control else "")
        + (
            (subj_name)
            if mapping_type == "unit2voxel"
            else ("source_subj" + source_subj)
        )
        + (
            "/"
            + ("RANDOM_" if random_pos_control else "SWAPOPT_" if "MB" in model_type else "")
            + hemi
            + "_"
            + roi
            + mapping_stem
            + "_"
            + checkpoint
            + "_"
            + ("functional_" if "MB" in model_type else "")
            + mapping_type
            + "_correlation_info.hdf5"
        )
    )
    
    mapping = {}
    with h5py.File(mapping_path, "r") as f:
        keys = f.keys()
        for k in keys:
            mapping[k] = f[k][:]

    return mapping


def get_winning_roi(
    subj_name="subj01",
    hemi="rh",
    roi="ministreams",
    checkpoint="final",
    spatial_weight=0.25,
    model_seed=0,
    corr_thresh=None,
):
    mapping = get_mapping(
        subj_name=subj_name,
        spatial_weight=spatial_weight,
        model_seed=model_seed,
        supervised=0,
        hemi=hemi,
        checkpoint=checkpoint,
        roi=roi,
    )

    # threshold based on assignment correlation values if specified
    if corr_thresh:
        mapping["winning_roi"][mapping["winning_corr"] < corr_thresh] = 0

    return mapping["winning_roi"]
