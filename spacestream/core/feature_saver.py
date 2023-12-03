"""
Feature Saver, highly inspired by SpaceTorch (NeuroAI Lab)
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import h5py
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from spacestream.core.feature_extractor import get_features_from_layer
from spacestream.utils.general_utils import make_iterable

# types
FeatureDict = Dict[str, np.ndarray]


class FeatureSaver:
    def __init__(
        self,
        model: nn.Module,
        layers: List[str],
        dataloader: DataLoader,
        save_path: Path,
    ):
        self.model = model
        self.layers = layers
        self.dataloader = dataloader
        self.dataset_len = len(dataloader.dataset)
        self.save_path = save_path

    def compute_features(self, batch_size: int = 32):

        self._features: FeatureDict = {
            layer: list(
                get_features_from_layer(
                    self.model, self.dataloader, layer, vectorize=True
                ).values()
            )[0]
            for layer in self.layers
        }

    def save_features(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(self.save_path, "w") as f:
            for k, v in self.features.items():
                f.create_dataset(k, data=v)

    @staticmethod
    def load_features(
        load_path: Path, keys: Optional[Union[List[str], str]] = None
    ) -> FeatureDict:

        features = {}
        with h5py.File(load_path, "r") as f:
            if keys is None:
                keys = f.keys()
            else:
                keys = make_iterable(keys)  # type: ignore

            for k in keys:
                features[k] = f[k][:]

        return features

    @property
    def features(self):
        if not hasattr(self, "_features"):
            raise Exception(
                "No features computed. Run the compute_features method first"
            )

        return self._features
