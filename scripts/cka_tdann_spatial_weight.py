"""
Compute linear CKA between two TDANN (spacetorch) models with different spatial
weights on NSD stimuli. Supports subsampling units to keep covariance sizes
manageable.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from spacestream.core.paths import RESULTS_PATH
from spacestream.datasets.nsd import nsd_dataloader
from spacestream.core.feature_extractor import get_features_from_layer
from spacestream.utils.get_utils import get_model, get_stimulus_indices
from spacestream.utils.metric_utils import (
    accumulate_centered_covariances,
    compute_streaming_means,
    iter_batches,
    linear_cka_from_covariances,
)


def parse_layers(layer_names, all_layers):
    if all_layers:
        return [
            "base_model.maxpool",
            "base_model.layer1.0",
            "base_model.layer1.1",
            "base_model.layer2.0",
            "base_model.layer2.1",
            "base_model.layer3.0",
            "base_model.layer3.1",
            "base_model.layer4.0",
            "base_model.layer4.1",
        ]
    return [name.strip() for name in layer_names.split(",") if name.strip()]


def select_indices(feature_len, max_units, rng):
    if max_units is None or feature_len <= max_units:
        return np.arange(feature_len, dtype=np.int64)
    return rng.choice(feature_len, size=max_units, replace=False)


def chunked(seq, chunk_size):
    for i in range(0, len(seq), chunk_size):
        yield seq[i : i + chunk_size]


def estimate_cov_memory_gb(feature_indices, layer_subset, cov_dtype):
    bytes_per = np.dtype(cov_dtype).itemsize
    total_bytes = 0
    for layer in layer_subset:
        d = len(feature_indices[layer])
        total_bytes += 3 * d * d * bytes_per
    return total_bytes / (1024 ** 3)


def extract_batch_features(
    model,
    layer_names,
    batch_idx,
    device,
):
    dataloader = nsd_dataloader(
        list(batch_idx),
        video=False,
        batch_size=len(batch_idx),
        model_name="resnet18",
    )
    feats = get_features_from_layer(
        model,
        dataloader,
        layer_names,
        batch_size=len(batch_idx),
        vectorize=True,
        device=device,
    )
    return feats


def compute_cka(
    model_a,
    model_b,
    layer_names,
    stim_indices,
    batch_size,
    device,
    feature_indices,
    layer_chunk_size,
    cov_dtype,
    max_cov_gb,
):
    cov_dtype_np = np.float16 if cov_dtype == "float16" else np.float32
    cka = {}

    for layer_chunk in chunked(layer_names, layer_chunk_size):
        est_gb = estimate_cov_memory_gb(feature_indices, layer_chunk, cov_dtype_np)
        if est_gb > max_cov_gb:
            raise ValueError(
                f"Estimated covariance memory for layer chunk {layer_chunk} is {est_gb:.2f} GB "
                f"(limit {max_cov_gb:.2f} GB). Reduce --layer_chunk_size or --max_units, "
                f"or use --cov_dtype float16."
            )

        key_a = {layer: f"a::{layer}" for layer in layer_chunk}
        key_b = {layer: f"b::{layer}" for layer in layer_chunk}
        all_keys = [key_a[layer] for layer in layer_chunk] + [key_b[layer] for layer in layer_chunk]
        expected_dims = {}
        for layer in layer_chunk:
            feat_len = len(feature_indices[layer])
            expected_dims[key_a[layer]] = feat_len
            expected_dims[key_b[layer]] = feat_len

        def extract_joint_batch(batch_idx):
            feats_a = extract_batch_features(model_a, layer_chunk, batch_idx, device)
            feats_b = extract_batch_features(model_b, layer_chunk, batch_idx, device)
            out = {}
            for layer in layer_chunk:
                out[key_a[layer]] = feats_a[layer][:, feature_indices[layer]]
                out[key_b[layer]] = feats_b[layer][:, feature_indices[layer]]
            return out

        means = compute_streaming_means(
            keys=all_keys,
            stim_indices=stim_indices,
            batch_size=batch_size,
            extract_batch_fn=extract_joint_batch,
            expected_dims=expected_dims,
        )
        cross_pairs = [(key_a[layer], key_b[layer]) for layer in layer_chunk]
        xx, xy = accumulate_centered_covariances(
            keys=all_keys,
            cross_pairs=cross_pairs,
            stim_indices=stim_indices,
            batch_size=batch_size,
            extract_batch_fn=extract_joint_batch,
            means=means,
            expected_dims=expected_dims,
            cov_dtype=cov_dtype_np,
        )
        cka_pairs = linear_cka_from_covariances(xx, xy)
        for layer in layer_chunk:
            cka[layer] = cka_pairs[(key_a[layer], key_b[layer])]

    return cka


def main(args):
    if args.max_units is not None and args.max_units <= 0:
        raise ValueError("--max_units must be positive or None.")

    layer_names = parse_layers(args.layer_names, args.all_layers)
    if not layer_names:
        raise ValueError("No layer names provided.")

    spatial_weight_a = args.spatial_weight_a
    spatial_weight_b = args.spatial_weight_b

    stim_indices = get_stimulus_indices(args.subset)
    n_stimuli = len(stim_indices)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    rng = np.random.default_rng(0)

    model_name = "spacetorch_supervised" if args.supervised else "spacetorch"
    seed_list = [0, 1, 2, 3, 4]
    # Initialize feature indices based on the first model pass
    probe_model = get_model(
        model_name,
        pretrained=True,
        spatial_weight=spatial_weight_a,
        model_seed=seed_list[0],
    ).eval()
    feature_indices = {}
    feature_dims = {}
    for batch_idx in iter_batches(stim_indices, args.batch_size):
        feats = extract_batch_features(probe_model, layer_names, batch_idx, device)
        for layer in layer_names:
            if layer not in feature_indices:
                feature_dims[layer] = feats[layer].shape[1]
                feature_indices[layer] = select_indices(
                    feature_dims[layer], args.max_units, rng
                )
        break
    del probe_model
    if device == "cuda":
        torch.cuda.empty_cache()

    pair_results = {}
    if spatial_weight_b is None:
        for i, seed_i in enumerate(seed_list):
            for seed_j in seed_list[i + 1 :]:
                model_a = get_model(
                    model_name,
                    pretrained=True,
                    spatial_weight=spatial_weight_a,
                    model_seed=seed_i,
                ).eval()
                model_b = get_model(
                    model_name,
                    pretrained=True,
                    spatial_weight=spatial_weight_a,
                    model_seed=seed_j,
                ).eval()
                cka_pair = compute_cka(
                    model_a,
                    model_b,
                    layer_names,
                    stim_indices,
                    args.batch_size,
                    device,
                    feature_indices,
                    args.layer_chunk_size,
                    args.cov_dtype,
                    args.max_cov_gb,
                )
                pair_results[f"{seed_i}_vs_{seed_j}"] = cka_pair
                del model_a, model_b
                if device == "cuda":
                    torch.cuda.empty_cache()
    else:
        for seed_i in seed_list:
            for seed_j in seed_list:
                model_a = get_model(
                    model_name,
                    pretrained=True,
                    spatial_weight=spatial_weight_a,
                    model_seed=seed_i,
                ).eval()
                model_b = get_model(
                    model_name,
                    pretrained=True,
                    spatial_weight=spatial_weight_b,
                    model_seed=seed_j,
                ).eval()
                cka_pair = compute_cka(
                    model_a,
                    model_b,
                    layer_names,
                    stim_indices,
                    args.batch_size,
                    device,
                    feature_indices,
                    args.layer_chunk_size,
                    args.cov_dtype,
                    args.max_cov_gb,
                )
                pair_results[f"{seed_i}_vs_{seed_j}"] = cka_pair
                del model_a, model_b
                if device == "cuda":
                    torch.cuda.empty_cache()
    cka = None

    out_dir = Path(RESULTS_PATH) / "analyses" / "cka" / "TDANNs"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"seedpairs_sw{spatial_weight_a}"
        if spatial_weight_b is None
        else f"seedpairs_sw{spatial_weight_a}_vs_sw{spatial_weight_b}"
    )
    out_path = out_dir / (
        f"cka_tdann_{tag}_{args.subset}.npz"
    )

    np.savez(
        out_path,
        cka=cka,
        layer_names=np.array(layer_names),
        n_stimuli=n_stimuli,
        batch_size=args.batch_size,
        spatial_weight_a=spatial_weight_a,
        spatial_weight_b=spatial_weight_b,
        seed_list=seed_list,
        supervised=args.supervised,
        device=device,
        subset=args.subset,
        max_units=args.max_units,
        layer_chunk_size=args.layer_chunk_size,
        cov_dtype=args.cov_dtype,
        max_cov_gb=args.max_cov_gb,
        feature_dims=feature_dims,
        feature_indices=feature_indices,
        pair_results=pair_results,
    )
    print(f"Saved CKA results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial_weight_a", type=float, default=0.0)
    parser.add_argument("--spatial_weight_b", type=float, default=None)
    parser.add_argument("--layer_names", type=str, default="base_model.layer4.1")
    parser.add_argument("--all_layers", action="store_true")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--supervised", type=int, default=0)
    parser.add_argument("--subset", type=str, default="shared_1000",
                        choices=["shared_1000", "shared_515", "full"])
    parser.add_argument("--max_units", type=int, default=19164) # use size of stream ROIs to fit in memory if not doing channel avg
    parser.add_argument("--layer_chunk_size", type=int, default=1)
    parser.add_argument("--cov_dtype", type=str, default="float32", choices=["float32", "float16"])
    parser.add_argument("--max_cov_gb", type=float, default=8.0)
    args = parser.parse_args()
    if args.layer_chunk_size <= 0:
        raise ValueError("--layer_chunk_size must be >= 1.")
    main(args)
