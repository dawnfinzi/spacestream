from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision
from pytorchvideo.transforms import ShortSideScale, UniformTemporalSubsample
from torchvision.transforms._transforms_video import (CenterCropVideo,
                                                      NormalizeVideo)


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


NORM_CFG = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
VIDEO_NORM_CFG = dict(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])

MODEL_TRANSFORM_PARAMS = {
    "slowfast": {
        "side_size": 256,
        "crop_size": 256,
        "num_frames": 32,
        "sampling_rate": 2,  # though https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/models/hub/README.md suggests 8 x 8
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


IMAGENET_TRANSFORMS = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(**NORM_CFG),
    ]
)

NUM_TRAIN_IMAGES = 1_281_167
NUM_VALIDATION_IMAGES = 50_000


class ImageNetData(torchvision.datasets.ImageFolder):
    """ImageNet data"""

    def __init__(self, root, video=False, transform=None, transform_params=None):
        super(ImageNetData, self).__init__(root, transform)
        self.video = video
        self.transform_params = transform_params

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, _ = self.imgs[index]
        img = self.loader(path)
        if self.video:
            img = np.asarray(img)
            img = np.expand_dims(img, axis=-1)
            img = np.tile(
                img,
                self.transform_params["num_frames"]
                * self.transform_params["sampling_rate"],
            )
            img = np.transpose(img, (2, 3, 0, 1))
            img = torch.from_numpy(img)

        if self.transform is not None:
            img = self.transform(img)

        return img


def imagenet_validation_dataloader(
    indices: Optional[list] = None,
    video: bool = False,
    batch_size: int = 32,
    model_name: str = "slowfast",
    imagenet_val_dir: str = "/oak/stanford/groups/kalanit/biac2/kgs/ml_datasets/imagenet/validation/",
) -> torch.utils.data.DataLoader:
    assert Path(imagenet_val_dir).exists

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
        if model_name == "slowfast":
            transform_list.append(PackPathway(4))
        transform = torchvision.transforms.Compose(transform_list)
    elif (
        model_name == "faster_RCNN" or model_name == "faster_rcnn"
    ):  # object detection trained on COCO
        transform_params = None
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=800, max_size=1333),
                torchvision.transforms.CenterCrop(800),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(**NORM_CFG),
            ]
        )

    else:
        transform_params = None
        transform = IMAGENET_TRANSFORMS

    dataset = ImageNetData(imagenet_val_dir, video, transform, transform_params)
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
