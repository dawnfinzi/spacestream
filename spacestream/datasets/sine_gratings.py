import glob
from dataclasses import dataclass
from typing import Callable, List, Optional, Type

import numpy as np
import open_clip
import torch
import torchvision
import xarray as xr
from pytorchvideo.transforms import ShortSideScale, UniformTemporalSubsample
from skimage import io
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms._transforms_video import (CenterCropVideo,
                                                      NormalizeVideo)
from typing_extensions import Literal

from spacestream.core.constants import DVA_PER_IMAGE
from spacestream.core.paths import SINE_GRATING_PATH
from spacestream.datasets.imagenet import NORM_CFG, VIDEO_NORM_CFG
from spacestream.utils.array_utils import flatten
from spacestream.utils.plot_utils import nauhaus_colormaps

AggMode = Literal["mode", "mean", "circmean", "circstd"]

MODEL_TRANSFORM_PARAMS = {
    "slowfast": {
        "side_size": 256,
        "crop_size": 256,
        "num_frames": 32,
        "sampling_rate": 2,  # though https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/models/hub/README.md suggests 8 x 8
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

DEFAULT_TRANSFORMS = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(**NORM_CFG),
    ]
)

# types
ComposedTransforms = Type[torchvision.transforms.transforms.Compose]


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


@dataclass
class Metric:
    name: str
    n_unique: int
    high: float
    xticklabels: np.ndarray
    xlabel: str
    agg_mode: AggMode
    colormap: Callable


class SineGrating2019(Dataset):
    """Full-field sine gratings dataset"""

    metrics: List[Metric] = [
        Metric(
            name="angles",
            n_unique=8,
            high=180,
            xticklabels=[f"{x:.0f}" for x in np.linspace(0, 180, 9)],
            xlabel=r"Orientation ($^\circ$)",
            agg_mode="circmean",
            colormap=nauhaus_colormaps["angles"],
        ),
        Metric(
            name="sfs",
            n_unique=8,
            high=110,
            xticklabels=[
                f"{x:.0f}" for x in np.linspace(5.5, 110.0, 9) / DVA_PER_IMAGE
            ],
            xlabel="Spatial Frequency (cpd)",
            agg_mode="mean",
            colormap=nauhaus_colormaps["sfs"],
        ),
        Metric(
            name="colors",
            n_unique=2,
            high=1,
            xticklabels=["B/W", "Color"],
            xlabel="",
            agg_mode="mean",
            colormap=nauhaus_colormaps["colors"],
        ),
    ]

    def __init__(
        self,
        sine_dir: str,
        video: bool = False,
        transforms: Optional[ComposedTransforms] = None,
        transform_params: dict = None,
    ):
        self.video = video
        self.transforms = transforms
        self.transform_params = transform_params
        self.file_list = sorted(glob.glob(f"{sine_dir}/*.jpg"))

        self.labels = np.zeros([len(self.file_list), 4], dtype=float)
        for img_idx, fname in enumerate(self.file_list):
            parts = fname.split("/")[-1].split("_")

            angle = float(parts[1][:-3])
            sf = float(parts[2][:-2])
            phase = float(parts[3][:-5])
            color_string = parts[4].split(".jpg")[0]
            color = 0.0 if color_string == "bw" else 1.0

            self.labels[img_idx, 0] = angle
            self.labels[img_idx, 1] = sf
            self.labels[img_idx, 2] = phase
            self.labels[img_idx, 3] = color

    @classmethod
    def get_metrics(cls, as_dict: bool = False):
        if as_dict:
            return {
                "angles": cls.metrics[0],
                "sfs": cls.metrics[1],
                "colors": cls.metrics[2],
            }
        return cls.metrics

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = io.imread(self.file_list[idx])
        target = self.labels[idx, :]

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

        if self.transforms:
            img = self.transforms(img)

        if self.video:
            return img
        else:
            return img, target


class SineResponses:
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ):
        self.DVA_PER_IMAGE = 8.0  # degrees of visual angle per image

        self._data = xr.DataArray(
            data=flatten(features),
            coords={
                "angles": ("image_idx", labels[:, 0]),
                "sfs": ("image_idx", labels[:, 1]),
                "phases": ("image_idx", labels[:, 2]),
                "colors": ("image_idx", labels[:, 3]),
            },
            dims=["image_idx", "unit_idx"],
        )

        self._convert_sfs_to_cpd()
        self._compute_circular_variance()

    def __len__(self) -> int:
        return self._data.sizes["unit_idx"]

    @property
    def orientation_tuning_curves(self) -> xr.DataArray:
        mean_response_to_each_orientation = self._data.groupby("angles").mean()
        return mean_response_to_each_orientation.T

    @property
    def circular_variance(self) -> np.ndarray:
        return self._data.circular_variance.values

    def get_preferences(self, metric: str = "angles") -> xr.DataArray:
        mean_response = self._data.groupby(metric).mean()
        return mean_response.argmax(axis=0)

    def get_peak_heights(self, metric: str = "angles"):
        tuning_curve = self._data.groupby(metric).mean()
        return np.ptp(tuning_curve.data, axis=0)

    def _convert_sfs_to_cpd(self):
        cpd: np.ndarray = self._data["sfs"] / self.DVA_PER_IMAGE
        self._data = self._data.assign_coords({"sfs": ("image_idx", cpd.data)})

    def _compute_circular_variance(self):
        n_angles = 8

        # the angles we use evenly span 0 to pi, but do not wrap
        angles = np.linspace(0, np.pi, n_angles + 1)[:-1]

        # compute "R"
        numerator = np.sum(
            self.orientation_tuning_curves * np.exp(angles * 2 * 1j), axis=1
        )
        denominator = np.sum(self.orientation_tuning_curves, axis=1)
        R = numerator / denominator

        # compute circular variance
        CV = 1 - np.abs(R)

        self._data = self._data.assign_coords(
            {"circular_variance": ("unit_idx", CV.data)}
        )


def sine_dataloader(
    indices: Optional[list] = None,
    video: bool = False,
    batch_size: int = 32,
    model_name: str = "slowfast",
    slowfast_alpha: int = 4,  # 4 for pretrained model, 8 for resnet18 version
) -> torch.utils.data.DataLoader:

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
    elif model_name.lower() == "ssd":
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize(size=300),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.48, 0.46, 0.41], std=[1.0, 1.0, 1.0]
                ),
            ]
        )
    elif model_name.lower() == "open_clip_rn50" or model_name.lower() == "open_clip_vit_b_32": # Same preprocessing for both
        _, _, transform = open_clip.create_model_and_transforms('RN50', pretrained='openai')
        transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()] + list(transform.transforms))
    elif model_name.lower() == "convnext_tiny":
        weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(weights.transforms().resize_size, interpolation=weights.transforms().interpolation),
            torchvision.transforms.CenterCrop(weights.transforms().crop_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
        ])
    elif model_name.lower() == "detr_rn50":
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize(size=800, max_size=1333),  # Resize shortest edge to 800, keep max at 1333
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard for DETR
            ])
    elif model_name.lower() in ("depth_anything_v2"):
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((259,259), interpolation=InterpolationMode.BICUBIC), # divisible by 14
                torchvision.transforms.ToTensor(),  # scales to [0,1] -> matches rescale_factor 1/255
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        transform = DEFAULT_TRANSFORMS

    dataset = SineGrating2019(SINE_GRATING_PATH, video, transform, transform_params)

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
