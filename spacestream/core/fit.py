from datetime import datetime

import numpy as np
from sklearn.decomposition import PCA

from spacestream.core.feature_extractor import get_features_from_layer
from spacestream.datasets.imagenet import imagenet_validation_dataloader
from spacestream.datasets.nsd import nsd_dataloader
from spacestream.utils.get_utils import get_model, get_model_layers
from spacestream.utils.mapping_utils import traditional_mapping


def log(to_print: str):
    print(f"\nLOG: {to_print}")


def extract_and_fit(
    model_name,
    model_layer_strings,
    subsample,
    mapping_func,
    CV,
    sorted_betas,
    beta_order,
    all_splits,
    pretrained=1,
    reduce_temporal_dims=0,
    return_weights=False,
    spatial_weight=1.25,
):
    rng = np.random.RandomState(seed=0)

    log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log("Setting/getting model")

    # load model (and/or get features)
    pt = True if pretrained == 1 else False

    model_name = model_name.lower()
    model = get_model(model_name, pretrained=pt, spatial_weight=spatial_weight)
    model_layer_strings = get_model_layers(model_name, model_layer_strings)

    two_pathway = False
    video = False
    reduction_list = None
    if model_name == "slowfast" or model_name == "slowfast_full":
        model_name = "slowfast"  # generic model name for dataloader
        two_pathway = True
        video = True
    elif model_name == "x3d_xs" or model_name == "x3d_s" or model_name == "x3d_m":
        video = True
    device = None
    if model_name == "faster_rcnn":
        device = "cpu"  # too big for CUDA mems

    if video == True:  # temporal dims not applicable otherwise
        if reduce_temporal_dims:
            reduction_list = np.tile(2, len(model_layer_strings))
            for lidx, l in enumerate(model_layer_strings):
                if (
                    l == "blocks.5.proj" or l == "blocks.5" or l == "blocks.6.proj"
                ):  # no temporal dim to reduce
                    reduction_list[lidx] = -1

    n_val_images = 1000
    if subsample == 2:  # PCA
        imagenet_batch = imagenet_validation_dataloader(
            list(range(0, n_val_images)),
            video=video,
            model_name=model_name,
        )
        imagenet_feats = get_features_from_layer(
            model,
            imagenet_batch,
            model_layer_strings,
            two_pathway=two_pathway,
            reduction_list=reduction_list,
            vectorize=True,
        )
        del imagenet_batch

    final_features = {}
    n_feats_per_layer = {}
    N = n_val_images  # num features to keep if subsampling (used to be 5k for subsamp = 1, and 1k for subsamp = 2)

    # compute features on the fly

    log("Computing features")
    nsd_batches = 146  # num batches to use (low memory load)
    imgs_per_batch = 500
    subj_stim_idx = np.sort(beta_order)
    keep_inds = {}
    pca_model = {}
    prev_batch_end = 0

    for b in range(nsd_batches):
        log(b)
        subj_batch_idx = subj_stim_idx[
            (subj_stim_idx >= imgs_per_batch * (b))
            & (subj_stim_idx < imgs_per_batch * (b + 1))
        ]
        batch_end = len(subj_batch_idx)

        batch = nsd_dataloader(
            list(subj_batch_idx),
            video=video,
            batch_size=len(list(subj_batch_idx)),
            model_name=model_name,
        )

        batch_feats = get_features_from_layer(
            model,
            batch,
            model_layer_strings,
            two_pathway=two_pathway,
            reduction_list=reduction_list,
            batch_size=len(list(subj_batch_idx)),
            vectorize=True,
            device=device,
        )

        for l in model_layer_strings:
            if b == 0:
                n_feats_per_layer[l] = batch_feats[l].shape[1]  # num feats pre subsamp
                print(n_feats_per_layer[l])
                perm = rng.permutation(
                    n_feats_per_layer[l]
                )  # pick a permutation of the set [0, ... n-1]
                keep_inds[l] = perm[:N]
                feat_length = (
                    n_feats_per_layer[l]
                    if subsample != 1
                    else min(N, n_feats_per_layer[l])
                )

                if subsample == 2:  # on first batch, PCA imagenet feats and keep models
                    feat_length = np.min([n_feats_per_layer[l], n_val_images])
                    pca_model[l] = PCA(n_components=feat_length)
                    pca_model[l].fit(imagenet_feats[l])
                    del imagenet_feats[l]

                # preallocate array
                final_features[l] = np.zeros(
                    (subj_stim_idx.shape[0], feat_length), dtype=np.float32
                )

            if subsample == 0:  # no subsampling
                final_features[l][
                    prev_batch_end : prev_batch_end + batch_end, :
                ] = batch_feats[l]
            elif subsample == 1:  # random subsampling
                final_features[l][
                    prev_batch_end : prev_batch_end + batch_end, :
                ] = batch_feats[l][:, keep_inds[l]]
            elif subsample == 2:  # PCA
                final_features[l][
                    prev_batch_end : prev_batch_end + batch_end, :
                ] = pca_model[l].transform(batch_feats[l])

        prev_batch_end += batch_end
        del batch, batch_feats

    # map from features to voxel responses
    log("Mapping to voxel responses")
    rsquared_array = traditional_mapping(
        final_features,
        sorted_betas,
        all_splits,
        model_layer_strings,
        mapping_func,
        CV,
        return_weights,
    )

    return rsquared_array
