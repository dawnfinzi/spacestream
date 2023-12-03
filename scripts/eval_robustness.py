# inspired by https://github.com/nathankong/robustness_primary_visual_cortex/blob/master/robust_spectrum/robustness_eval/imagenet_robustness.py
import argparse
import os
import pickle

import eagerpy as ep
import torch
import torch.nn as nn
import torchvision
from foolbox import PyTorchModel, accuracy
from foolbox.attacks import L1PGD, L2PGD, LinfPGD
from vissl.models.heads import LinearEvalMLP
from vissl.models.heads.mlp import MLP
from vissl.utils.hydra_config import AttrDict

from spacestream.core.paths import RESULTS_PATH
from spacestream.datasets.imagenet import NORM_CFG
from spacestream.models.spatial_resnet import SpatialResNet18
from spacestream.utils.get_utils import get_eval_ckpts

VAL_DIR = "/oak/stanford/groups/kalanit/biac2/kgs/ml_datasets/imagenet/validation/"
# DON'T NORMALIZE!
VAL_TRANSFORMS = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
    ]
)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class CombinedModel(nn.Module):
    def __init__(self, trunk: nn.Module, head: nn.Module):
        super(CombinedModel, self).__init__()
        self.trunk = trunk
        self.head = head

    def forward(self, x, **trunk_kwargs):
        x = self.trunk(x, **trunk_kwargs)
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = self.head(x)
        return x


def remove_leading_index(k):
    if k.startswith("0."):
        return k[2:]


def get_model_with_head(model_name, supervised, spatial_weight, model_seed):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    weight_path = get_eval_ckpts(
        float(spatial_weight), model_seed, supervised, model_name
    )

    ckpt = torch.load(weight_path, map_location=torch.device(device))
    model_params = ckpt["classy_state_dict"]["base_model"]["model"]
    trunk_params = model_params["trunk"]
    head_params = model_params["heads"]

    trunk = SpatialResNet18()
    trunk.load_state_dict(trunk_params, strict=False)

    if supervised:
        model_config = AttrDict({"HEAD": {"PARAMS_MULTIPLIER": 1.0}})
        head = MLP(model_config, [512, 1000])
    else:
        model_config = AttrDict(
            {
                "HEAD": {
                    "BATCHNORM_EPS": 1e-5,
                    "BATCHNORM_MOMENTUM": 0.1,
                    "PARAMS_MULTIPLIER": 1.0,
                }
            }
        )
        head = LinearEvalMLP(
            model_config, 512, [512, 1000], use_bn=False, use_relu=False
        )

    modified_head_params = {remove_leading_index(k): v for k, v in head_params.items()}
    head.load_state_dict(modified_head_params)
    model = CombinedModel(trunk, head)

    return model


def main(attack_name, attack_range, supervised, spatial_weight, seed):
    if attack_range == 1:
        assert attack_name == "Linf", "extended range only for Linf attack"

    if supervised:
        model_name = "spacetorch_supervised"
    else:
        model_name = "spacetorch"

    model = get_model_with_head(model_name, supervised, spatial_weight, seed)
    model.eval()  # set model to eval mode

    img_mean = NORM_CFG["mean"]
    img_std = NORM_CFG["std"]

    preprocessing = dict(mean=img_mean, std=img_std, axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    # get dataloader
    val_dataset = torchvision.datasets.ImageFolder(
        VAL_DIR,
        VAL_TRANSFORMS,
    )

    # set attack type
    num_steps = 5
    if attack_name == "Linf":
        if attack_range == -1:
            epsilons = [0.0, 1.0 / 1020]
        elif attack_range == 1:  # extended range
            epsilons = [
                0,
                0.00005,
                0.0001,
                0.00025,
                0.0005,
                0.00075,
                0.001,
                0.0025,
                0.0075,
            ]
            num_steps = 64
        else:
            epsilons = [0.0, 1.0 / 1020, 1.0 / 255, 4.0 / 255]
        attack_type = LinfPGD
        batch_size = 256
    elif attack_name == "L2":
        if attack_range == -1:
            epsilons = [0.0, 0.15]
        else:
            epsilons = [0.0, 0.15, 0.6, 2.4]
        attack_type = L2PGD
        batch_size = 64
    elif attack_name == "L1":
        if attack_range == -1:
            epsilons = [0.0, 40.0]
        else:
            epsilons = [0.0, 40.0, 160.0, 640.0]
        attack_type = L1PGD
        batch_size = 64
    else:
        assert 0

    print("==> Test data..")
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=None,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    rob_meters = dict()
    for eps in epsilons:
        rob_meters[eps] = AverageMeter(eps)

    for i, (images, labels) in enumerate(val_loader):
        print(f"Iteration {i+1}/{len(val_loader)}")

        N = images.shape[0]
        images, labels = images.cuda(), labels.cuda()
        images, labels = ep.astensor(images), ep.astensor(labels)

        for eps in epsilons:
            if eps == 0:
                robust_accuracy = [accuracy(fmodel, images, labels)]
            else:
                attack = attack_type(
                    abs_stepsize=eps * 1 / 4, steps=num_steps, random_start=False
                )
                _, _, success = attack(fmodel, images, labels, epsilons=[eps])
                robust_accuracy = 1 - ep.astensor(success).float32().mean(axis=-1)

            for e, rob_acc in zip([eps], robust_accuracy):
                # Update robust acc for each eps meter
                if e == 0:
                    rob_meters[e].update(rob_acc, N)
                else:
                    rob_meters[e].update(rob_acc.item(), N)

    # Print results
    results_dict = dict()
    for eps in epsilons:
        print(f"eps: {eps} || rob acc: {rob_meters[eps].avg}")
        results_dict[eps] = rob_meters[eps].avg

    # Save results
    full_model_name = f"{model_name}_sw{spatial_weight}_seed{seed}"
    save_dir = RESULTS_PATH + "analyses/robustness" + f"/{full_model_name}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if attack_range == -1:
        fname = save_dir + f"/{attack_name}_limited_range.pkl"
    elif attack_range == 1:
        fname = save_dir + f"/{attack_name}_extended_range.pkl"
    else:
        fname = save_dir + f"/{attack_name}.pkl"
    pickle.dump(results_dict, open(fname, "wb"))


if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attack_type", type=str, default="Linf"
    )  # type of adversarial attack - one of Linf, L1 and L2
    parser.add_argument(
        "--attack_range", type=int, default=0
    )  # limited attack range for speed-up (-1), standard (0) or extended range (1) for comparison to Dapello ICLR 2023
    parser.add_argument(
        "--supervised", type=int, default=0
    )  # 1 is supervised, 0 is self-sup
    parser.add_argument("--spatial_weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    ARGS, _ = parser.parse_known_args()

    main(
        ARGS.attack_type,
        ARGS.attack_range,
        ARGS.supervised,
        ARGS.spatial_weight,
        ARGS.seed,
    )
