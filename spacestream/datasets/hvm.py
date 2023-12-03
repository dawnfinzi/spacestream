import copy
from typing import Optional

import h5py
import numpy as np
import torch
import torchvision
from PIL import Image

from spacestream.core.paths import HVM_PATH
from spacestream.datasets.imagenet import NORM_CFG
from spacestream.utils.dataloader_utils import duplicate_channels

HVM_TRANSFORMS = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(**NORM_CFG),
    ]
)


class HVMDataset(torch.utils.data.Dataset):
    """HVM stimuli."""

    def __init__(
        self,
        stim_path="/oak/stanford/groups/kalanit/biac2/kgs/ml_datasets/hvm/ventral_neural_data.hdf5",
        transform=None,
        name="hvm_dataset",
    ):
        """
        Args:
            stim_path (string): Path to the hdf file with stimuli.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        stim = h5py.File(stim_path, "r")
        self.data, self.stim_meta = self._extract_image_data(stim)
        self.transform = transform
        self.name = name

    def _extract_image_data(self, data):
        images = np.array(data["images"])

        # Image metadata
        all_meta = dict()
        for meta in data["image_meta"].keys():
            all_meta[meta] = np.array(data["image_meta"][meta])

        # 'Animals', 'Boats', 'Cars', 'Chairs', 'Faces', 'Fruits',
        # 'Planes', 'Tables'
        labels = copy.deepcopy(all_meta["category"])
        labels[labels == b"Animals"] = 1
        labels[labels == b"Boats"] = 2
        labels[labels == b"Cars"] = 3
        labels[labels == b"Chairs"] = 4
        labels[labels == b"Faces"] = 5
        labels[labels == b"Fruits"] = 6
        labels[labels == b"Planes"] = 7
        labels[labels == b"Tables"] = 8
        labels = labels.astype(int)
        all_meta["category_index"] = labels

        # Change instance labels into ints
        instance_labels = copy.deepcopy(all_meta["object_name"])
        assert len(np.unique(instance_labels)) == 64
        for i, instance_label in enumerate(np.unique(instance_labels)):
            assert (instance_labels == instance_label).sum() == 90
            instance_labels[instance_labels == instance_label] = i
        all_meta["instance_index"] = instance_labels.astype(int)

        # Duplicate channels since original images are grayscale
        images = duplicate_channels(images)

        return images, all_meta

    def __getitem__(self, index):
        x = self.data[index]

        if self.transform:
            x = Image.fromarray(x.astype(np.uint8))
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.data)

    def get_name(self):
        return self.name


def hvm_dataloader(
    indices: Optional[list] = None,
    batch_size: int = 128,
) -> torch.utils.data.DataLoader:

    full_stim_path = HVM_PATH + "ventral_neural_data.hdf5"

    transform = HVM_TRANSFORMS

    dataset = HVMDataset(full_stim_path, transform)

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
