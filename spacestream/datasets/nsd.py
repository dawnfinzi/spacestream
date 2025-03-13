from typing import Optional

import h5py
import numpy as np
import open_clip
import torch
import torchvision
from PIL import Image
from pytorchvideo.transforms import ShortSideScale, UniformTemporalSubsample
from torchvision.transforms._transforms_video import (CenterCropVideo,
                                                      NormalizeVideo)

from spacestream.core.paths import STIM_PATH
from spacestream.datasets.imagenet import NORM_CFG, VIDEO_NORM_CFG


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def __init__(self, slowfast_alpha):
        super().__init__()
        self.slowfast_alpha = slowfast_alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


MODEL_TRANSFORM_PARAMS = {
    "slowfast": {
        "side_size": 256,
        "crop_size": 256,
        "num_frames": 32,
        "sampling_rate": 2,
    },
    "slowfast18": {
        "side_size": 256,
        "crop_size": 256,
        "num_frames": 32,
        "sampling_rate": 2,
    },
    "x3d_xs": {
        "side_size": 182,
        "crop_size": 182,
        "num_frames": 4,
        "sampling_rate": 12,
    },
    "x3d_s": {
        "side_size": 182,
        "crop_size": 182,
        "num_frames": 13,
        "sampling_rate": 6,
    },
    "x3d_m": {
        "side_size": 256,
        "crop_size": 256,
        "num_frames": 16,
        "sampling_rate": 5,
    },
}


NSD_TRANSFORMS = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(**NORM_CFG),
    ]
)


class NSDataset(torch.utils.data.Dataset):
    """NSD stimuli."""

    def __init__(self, stim_path, video=False, transform=None, transform_params=None):
        """
        Args:
            stim_path (string): Path to the hdf file with stimuli.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        stim = h5py.File(stim_path, "r")  # 73k images
        self.data = stim["imgBrick"]
        self.video = video
        self.transform = transform
        self.transform_params = transform_params

    def __getitem__(self, index):
        x = self.data[index]

        if self.transform:
            if self.video:
                x = np.expand_dims(x, axis=-1)
                x = np.tile(
                    x,
                    self.transform_params["num_frames"]
                    * self.transform_params["sampling_rate"],
                )
                x = np.transpose(x, (2, 3, 0, 1))
                x = torch.from_numpy(x)
            else:
                x = Image.fromarray(x.astype(np.uint8))
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.data)


def nsd_dataloader(
    indices: Optional[list] = None,
    video: bool = False,
    batch_size: int = 32,
    model_name="slowfast",
    slowfast_alpha=4,  # 4 for pretrained model, 8 for resnet18 version
) -> torch.utils.data.DataLoader:
    full_stim_path = STIM_PATH + "nsd_stimuli.hdf5"

    transform_params = None
    if video:
        # Get transform parameters based on model
        transform_params = MODEL_TRANSFORM_PARAMS[model_name]
        transform_list = [
            UniformTemporalSubsample(transform_params["num_frames"]),
            torchvision.transforms.Lambda(lambda x: x / 255.0),
            NormalizeVideo(**VIDEO_NORM_CFG),
            ShortSideScale(size=transform_params["side_size"]),
            CenterCropVideo(transform_params["crop_size"]),
        ]
        if "slowfast" in model_name:
            transform_list.append(PackPathway(slowfast_alpha))
        transform = torchvision.transforms.Compose(transform_list)
    elif model_name.lower() == "faster_rcnn":  # object detection trained on COCO
        # don't need a diff collate_fn to deal with different image sizes as nsd images have already been square cropped
        # and don't normalize by the imagenet norms & stds
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=800, max_size=1333),
                torchvision.transforms.ToTensor(),
            ]
        )
    elif model_name.lower() == "ssd":
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=300),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.48, 0.46, 0.41], std=[1.0, 1.0, 1.0]
                ),
            ]
        )
    elif model_name.lower() == "open_clip_RN50":
        _, _, transform = open_clip.create_model_and_transforms('RN50', pretrained='openai')
    else:
        transform_params = None
        transform = NSD_TRANSFORMS

    dataset = NSDataset(full_stim_path, video, transform, transform_params)

    if indices:
        dataset = torch.utils.data.Subset(dataset, indices)
    data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    return data_loader
