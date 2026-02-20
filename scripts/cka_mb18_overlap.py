"""
Compute linear CKA between MB18 task representations on NSD stimuli.
Uses the MB18 selected units (chosen_indices) and streams through images to
avoid storing full feature matrices.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from spacestream.core.feature_extractor import get_features_from_layer
from spacestream.core.paths import DATA_PATH, RESULTS_PATH
from spacestream.datasets.nsd import nsd_dataloader
from spacestream.utils.get_utils import get_model, get_stimulus_indices
from spacestream.utils.metric_utils import (
    accumulate_centered_covariances,
    compute_streaming_means,
    linear_cka_from_covariances,
)


TASK_CONFIG = {
    "categorization": {
        "layer_name": "layer4.1",
        "model_name": "resnet18",
        "video": False,
        "two_pathway": False,
        "reduction_list": None,
    },
    "action": {
        "layer_name": "slow.layer4.1",
        "model_name": "slowfast18",
        "video": True,
        "two_pathway": True,
        "reduction_list": None,
    },
    "detection": {
        "layer_name": "backbone.feature_provider.feature_provider.7.1.conv1",
        "model_name": "ssd",
        "video": False,
        "two_pathway": False,
        "reduction_list": None,
    },
}


def pool_channel_only(feats):
    if feats.ndim == 5:
        return feats.mean(axis=(2, 3, 4))
    if feats.ndim == 4:
        return feats.mean(axis=(2, 3))
    return feats


def extract_batch_features(
    model,
    task_name,
    batch_idx,
    chosen_indices,
    slowfast_alpha,
    device,
    channel_only,
    log_dims,
):
    cfg = TASK_CONFIG[task_name]
    dataloader = nsd_dataloader(
        list(batch_idx),
        video=cfg["video"],
        batch_size=len(batch_idx),
        model_name=cfg["model_name"],
        slowfast_alpha=slowfast_alpha,
    )
    feats = get_features_from_layer(
        model,
        dataloader,
        cfg["layer_name"],
        two_pathway=cfg["two_pathway"],
        reduction_list=cfg["reduction_list"],
        batch_size=len(batch_idx),
        vectorize=not channel_only,
        device=device,
    )[cfg["layer_name"]]
    if log_dims:
        print(
            f"{task_name} raw feats shape: {feats.shape} (channel_only={channel_only})"
        )
    if channel_only:
        feats = pool_channel_only(feats)
        if log_dims:
            print(f"{task_name} channel-only shape: {feats.shape}")
    if chosen_indices is None:
        return feats
    return feats[:, chosen_indices]


def main(args):
    if args.base != "18":
        raise ValueError("Only MB18 is supported by this script for now.")

    tasks = ["categorization", "action", "detection"]
    if args.channel_only:
        chosen_path = None
        chosen_npz = {task: None for task in tasks}
    else:
        chosen_path = (
            args.chosen_indices_path
            or (DATA_PATH + "/models/MBs/RN18/chosen_indices.npz")
        )
        chosen_npz = np.load(chosen_path)

    stim_indices = get_stimulus_indices(args.subset)
    n_stimuli = len(stim_indices)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    models = {
        task: get_model(
            TASK_CONFIG[task]["model_name"],
            pretrained=True,
            model_seed=args.model_seed,
        ).eval()
        for task in tasks
    }

    feat_lens = {
        task: (len(chosen_npz[task]) if chosen_npz[task] is not None else None)
        for task in tasks
    }

    if not args.channel_only:
        assert feat_lens["categorization"] == feat_lens["action"] == feat_lens["detection"], (
            "MB18 chosen_indices must yield identical feature dimensions across tasks "
            f"for feature-space CKA, got {feat_lens}"
        )

    pass1_logged = {task: False for task in tasks}

    def extract_task_batch(batch_idx, log_first=False):
        feats = {}
        for task in tasks:
            task_feats = extract_batch_features(
                models[task],
                task,
                batch_idx,
                chosen_npz[task],
                args.slowfast_alpha,
                device,
                args.channel_only,
                log_first and (not pass1_logged[task]),
            )
            feats[task] = task_feats
            if log_first:
                pass1_logged[task] = True
        return feats

    means = compute_streaming_means(
        keys=tasks,
        stim_indices=stim_indices,
        batch_size=args.batch_size,
        extract_batch_fn=lambda batch_idx: extract_task_batch(batch_idx, log_first=True),
        expected_dims=feat_lens,
    )

    feat_lens = {task: len(means[task]) for task in tasks}
    task_pairs = [
        ("categorization", "action"),
        ("categorization", "detection"),
        ("action", "detection"),
    ]
    xx, xy = accumulate_centered_covariances(
        keys=tasks,
        cross_pairs=task_pairs,
        stim_indices=stim_indices,
        batch_size=args.batch_size,
        extract_batch_fn=lambda batch_idx: extract_task_batch(batch_idx, log_first=False),
        means=means,
        expected_dims=feat_lens,
    )
    cka_pairs = linear_cka_from_covariances(xx, xy)
    cka = {f"{left}_vs_{right}": score for (left, right), score in cka_pairs.items()}

    out_dir = Path(RESULTS_PATH) / "analyses" / "cka" / "MBs" / "RN18"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix_parts = []
    if args.channel_only:
        suffix_parts.append("channel_only")
    suffix_parts.append(args.subset)
    out_path = out_dir / f"cka_{'_'.join(suffix_parts)}.npz"

    np.savez(
        out_path,
        cka=cka,
        tasks=np.array(tasks),
        layer_names=np.array([TASK_CONFIG[t]["layer_name"] for t in tasks]),
        chosen_indices_path=str(chosen_path) if chosen_path is not None else "",
        n_stimuli=n_stimuli,
        batch_size=args.batch_size,
        model_seed=args.model_seed,
        device=device,
        slowfast_alpha=args.slowfast_alpha,
        channel_only=args.channel_only,
    )
    print(f"Saved CKA results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="18")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument(
        "--subset",
        type=str,
        default="shared_1000",
        choices=["shared_1000", "shared_515", "full"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_seed", type=int, default=0)
    parser.add_argument("--slowfast_alpha", type=int, default=8)
    parser.add_argument("--chosen_indices_path", type=str, default=None)
    parser.add_argument("--channel_only", action="store_true")
    args = parser.parse_args()
    main(args)
