import argparse
import logging
import socket
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR
from vissl.config import AttrDict
from vissl.models.heads.linear_eval_mlp import LinearEvalMLP

from spacestream.core.paths import RESULTS_PATH
from spacestream.models.spatial_resnet import SpatialResNet18
from spacestream.utils.get_utils import get_ckpts

# constants
STREAM_IDX = {"Ventral": 5, "Lateral": 6, "Parietal": 7}
VAL_DIR = "/oak/stanford/groups/kalanit/biac2/kgs/ml_datasets/imagenet/validation/"
TRAIN_DIR = "/oak/stanford/groups/kalanit/biac2/kgs/ml_datasets/imagenet/train/"
IMG_DIM = 224
tensorboard = False
if socket.gethostname() == "new-nefesh.stanford.edu":
    BATCH_SIZE = 512
    TEST_SET_BATCH_SIZE = 512
    NUM_WORKERS = 16
    VAL_DIR = "/data/dfinzi/imagenet/validation/"
    TRAIN_DIR = "/data/dfinzi/imagenet/train/"
else:
    BATCH_SIZE = 512
    TEST_SET_BATCH_SIZE = 512
    NUM_WORKERS = 8
if tensorboard:
    from torch.utils.tensorboard import SummaryWriter
NUM_EPOCHS = 28
TEST_EVERY = 4
MOMENTUM = 0.9
BASE_LR = 0.04
LOG_STEPS = 200
NUM_CORES = None


# masked MLP module for learning stream-specific readout to Imagenet
class MaskMLP(nn.Module):
    def __init__(self, unit_idx=None):
        super(MaskMLP, self).__init__()

        self.mask = unit_idx

        # create VISSL model config
        model_config = AttrDict(
            {
                "HEAD": {
                    "BATCHNORM_EPS": 1e-5,
                    "BATCHNORM_MOMENTUM": 0.1,
                    "PARAMS_MULTIPLIER": 1.0,
                }
            }
        )
        if unit_idx is not None:
            self.mlp = LinearEvalMLP(
                model_config,
                5000,
                [5000, 1000],
                use_bn=False,
                use_relu=False,  # bn is always included before linear layer, use_bn=False means no bn after
            )
        else:  # use all units
            self.mlp = LinearEvalMLP(
                model_config,
                25088,
                [25088, 1000],
                use_bn=False,
                use_relu=False,  # bn is always included before linear layer, use_bn=False means no bn after
            )

    def forward(self, x):
        flat_units = torch.flatten(x, 1)
        if self.mask is not None:
            x = flat_units[:, self.mask]
        else:
            x = flat_units

        # MLP
        x = self.mlp(x)

        return x


def get_model_base(model_name, spatial_weight, model_seed=0):
    model = SpatialResNet18()

    # drop the avgpool layer
    model.base_model.avgpool = nn.Identity()
    model.base_model.fc = nn.Identity()

    # get weights
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name == "spacetorch":
        supervised = False
    else:
        supervised = True
    weight_path = get_ckpts(float(spatial_weight), model_seed, supervised)
    ckpt = torch.load(weight_path, map_location=torch.device(device))
    model_params = ckpt["classy_state_dict"]["base_model"]["model"]["trunk"]

    # remove unnecc. weights
    new_state_dict = {}
    for k, v in model_params.items():
        if "fc." not in k and "avgpool." not in k:
            new_state_dict[k] = v
    # reload weights
    model.load_state_dict(new_state_dict)

    # freeze all weights
    for param in model.parameters():
        param.requires_grad = False

    return model, supervised


def get_mask_unit_idx(
    stream,
    spatial_weight,
    supervised,
    subj,
    hemi="rh",
    roi="ministreams",
    checkpoint="final",
    sampling=5,
    model_seed=0,
):
    # Retrieve mapping
    subj_name = "subj" + subj
    if supervised:
        stem = "supervised"
    else:
        stem = "self-supervised"
    corr_dir = (
        RESULTS_PATH
        + "mappings/one_to_one/unit2voxel/TDANNs/"
        + stem
        + (
            "/spatial_weight"
            + str(spatial_weight)
            + (("_seed" + str(model_seed)) if model_seed > 0 else "")
        )
    )
    mapping_path = Path(
        corr_dir
        / subj_name
        / (
            hemi
            + "_"
            + roi
            + "_CV_HVA_only_radius5.0_max_iters100_constant_radius_2.0dist_cutoff_constant_dist_cutoff_"
            + "spherical_target_radius_factor1.0_"
            + checkpoint
            + "_unit2voxel_correlation_info.hdf5"
        )
    )
    mapping = {}
    with h5py.File(mapping_path, "r") as f:
        keys = f.keys()
        for k in keys:
            mapping[k] = f[k][:]

    # Get units for stream
    unit_idx = np.where(mapping["winning_roi"] == STREAM_IDX[stream])[0]

    # sample x units with highest correlation
    n = int(1000 * sampling)
    unit_idx = unit_idx[np.argsort(mapping["winning_test_corr"][unit_idx])[::-1][0:n]]

    return unit_idx


def train_imagenet(stream, model_name, spatial_weight, model_seed, subj, hemi):
    # setup
    name = f"{model_name}_sw{str(spatial_weight)}_{hemi}_subj{subj}_{stream}_seed{str(model_seed)}"
    LOG_DIR = f"{RESULTS_PATH}/transfer/linear_eval/logs/{name}"
    SAVE_PATH = (
        f"{RESULTS_PATH}/transfer/linear_eval/checkpoints/{name}_linear_eval.torch"
    )

    logging.basicConfig(
        filename=LOG_DIR + "_training.log",
        format="%(asctime)s %(message)s",
        filemode="w",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info("Starting at {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("==> Preparing data..")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = torchvision.datasets.ImageFolder(
        TRAIN_DIR,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(IMG_DIM),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    resize_dim = max(IMG_DIM, 256)
    test_dataset = torchvision.datasets.ImageFolder(
        VAL_DIR,
        transforms.Compose(
            [
                transforms.Resize(resize_dim),
                transforms.CenterCrop(IMG_DIM),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    print("Done at {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    train_sampler, test_sampler = None, None
    print("==> Training data..")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=False if train_sampler else True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    print("==> Test data..")
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=TEST_SET_BATCH_SIZE,
        sampler=test_sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    print("Done at {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    torch.manual_seed(0)

    print("==> Building model..")
    # get base model and build MLP head
    model, supervised = get_model_base(model_name, spatial_weight, model_seed)
    if stream == "All":  # all units, no subsampling by stream!
        unit_idx = None
    else:
        unit_idx = get_mask_unit_idx(
            stream, spatial_weight, supervised, subj, hemi, model_seed
        )
    net_add = MaskMLP(unit_idx)
    model = nn.Sequential(model, net_add)

    device = "cuda"
    model = model.to(device)

    if tensorboard:
        writer = SummaryWriter(LOG_DIR)

    # manually set param groups, since we want to regularize some things and not others
    defaults = {"momentum": MOMENTUM, "nesterov": True, "lr": BASE_LR}
    param_groups = [
        {
            "params": model[1].mlp.channel_bn.parameters(),
            "weight_decay": 0.0,
            **defaults,
        },
        {"params": model[1].mlp.clf.parameters(), "weight_decay": 5e-4, **defaults},
    ]

    optimizer = optim.SGD(param_groups)
    lr_scheduler = MultiStepLR(optimizer, milestones=[8, 16, 24], gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()

    def train_loop_fn(loader, epoch, running_loss):
        size = len(loader.dataset)
        model.train()
        for step, (data, target) in enumerate(loader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            # for param in model.parameters():
            # param.grad=None
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            current_lr = lr_scheduler.get_last_lr()[0]
            running_loss += loss.item()
            if step % LOG_STEPS == 0:
                current = step * len(data)
                print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")
                logger.info(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")

                if tensorboard:
                    # log the current loss
                    writer.add_scalar(
                        "current training loss",
                        loss.item(),
                        epoch * len(loader) + step,
                    )
                    # log the running loss
                    writer.add_scalar(
                        "running training loss",
                        running_loss / (LOG_STEPS),
                        epoch * len(loader) + step,
                    )
                    writer.add_scalar(
                        "learning rate", current_lr, epoch * len(loader) + step
                    )
                running_loss = 0

        return running_loss

    def test_loop_fn(loader, epoch):
        total_samples, correct = 0, 0
        model.eval()
        for _, (data, target) in enumerate(loader):
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()
            total_samples += data.size()[0]

        accuracy = 100.0 * correct.item() / total_samples
        return accuracy

    running_loss, accuracy, max_accuracy = 0.0, 0.0, 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        print(
            "Epoch {} train begin {}".format(
                epoch, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        )
        running_loss = train_loop_fn(train_loader, epoch, running_loss)
        lr_scheduler.step()
        print(
            "Epoch {} train end {}".format(
                epoch, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        )
        if epoch == NUM_EPOCHS or epoch % TEST_EVERY == 0:
            accuracy = test_loop_fn(test_loader, epoch)
            print(
                "Epoch {} test end {}, Accuracy={:.2f}".format(
                    epoch, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), accuracy
                )
            )
            logger.info(
                "Epoch {} test end {}, Accuracy={:.2f}".format(
                    epoch, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), accuracy
                )
            )
            max_accuracy = max(accuracy, max_accuracy)
            if tensorboard:
                writer.add_scalar("Accuracy/test:", max_accuracy, epoch)

    # test_utils.close_summary_writer(writer)
    print("Max Accuracy: {:.2f}%".format(max_accuracy))
    logger.info("Max Accuracy: {:.2f}%".format(max_accuracy))
    if tensorboard:
        writer.close()
    torch.save(model.state_dict(), SAVE_PATH)
    return max_accuracy


if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stream", type=str, default="Ventral"
    )  # "Ventral", "Lateral", or "Parietal" - or "All" if all units
    parser.add_argument(
        "--model_name", type=str, default="spacetorch"
    )  # spacetorch or spacetorch_supervised
    parser.add_argument("--spatial_weight", type=float, default=0.5)
    parser.add_argument("--model_seed", type=int, default=0)
    parser.add_argument("--subj", type=str, default="01")
    parser.add_argument("--hemi", type=str, default="rh")

    ARGS, _ = parser.parse_known_args()

    train_imagenet(
        ARGS.stream,
        ARGS.model_name,
        ARGS.spatial_weight,
        ARGS.model_seed,
        ARGS.subj,
        ARGS.hemi,
    )
