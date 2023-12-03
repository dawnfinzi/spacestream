import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from spacestream.core.constants import (IT_SIZE, RETINA_SIZE, V1_SIZE, V2_SIZE,
                                        V4_SIZE)
from spacestream.utils.array_utils import FlatIndices, get_flat_indices

# set up logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)-15s %(levelname)s:%(message)s"
)
logger = logging.getLogger(__name__)

TISSUE_SIZES = {
    "retina": RETINA_SIZE,
    "V1": V1_SIZE,
    "V2": V2_SIZE,
    "V4": V4_SIZE,
    "IT": IT_SIZE,
}

# V1 width comes from a combination of a few sources:
#    1. Stettler et al. 2002 show 400-500 microns as the area of dense horizontal
#         connections
#    2. Bosking et al. 1997 show 500 microns as the area of dense horizontal connections
#    3. Malach et al., 1993 write that in a range of 400 microns, "the local,
#         synaptic-rich axonal and dendritic arbors crossed freely through columns of
#         diverse functional properties."

NEIGHBORHOOD_WIDTHS = {"retina": 0.1, "V1": 0.4, "V2": 1.0, "V4": 2.0, "IT": 3.0}


@dataclass
class LayerPositions:
    # name should be the name of the layer these positions are for
    name: str

    # dims are in CHW format if conv, or a 1-tuple if fc
    dims: Union[Tuple[int, int, int], Tuple[int]]

    # coordinates should be an N x 2 matrix with the x-coordinates of each unit in the
    # first column and the y-coordinates in the second column
    coordinates: np.ndarray

    # coordinates should be an P x Q matrix. Each row is one neighborhood consisting
    # of P indices. For all p_i \in P, 0 <= p_i <= len(coordinates)
    neighborhood_indices: np.ndarray

    # neighborhood_width is the width, in mm, of the neighborhoods
    neighborhood_width: float

    def __post_init__(self):
        assert np.prod(self.dims) == len(
            self
        ), "dims don't match number of units provided"

    @property
    def flat_indices(self) -> FlatIndices:
        if len(self.dims) == 1:
            return np.arange(self.dims[0])
        elif len(self.dims) == 3:
            return get_flat_indices(self.dims)
        else:
            raise Exception(
                f"Sorry, only FC (1-D shape) and conv (3-D shape) kernels are accepted, and dims was provided with {len(self.dims)} dimensions"
            )

    def save(self, save_dir: Path):
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        path = save_dir / f"{self.name}.pkl"
        with path.open("wb") as stream:
            pickle.dump(self, stream)

    @staticmethod
    def load(path: Path) -> "LayerPositions":
        path = Path(path)
        assert path.exists()
        with path.open("rb") as stream:
            return pickle.load(stream)

    def __len__(self) -> int:
        return len(self.coordinates)


@dataclass
class NetworkPositions:
    layer_positions: Dict[str, LayerPositions]

    @classmethod
    def load_from_dir(cls, load_dir: Path):
        load_dir = Path(load_dir)
        assert load_dir.is_dir()
        layer_files = load_dir.glob("*.pkl")

        d = {}
        for layer_file in layer_files:
            layer_name = layer_file.stem
            d[layer_name] = LayerPositions.load(layer_file)

        return cls(layer_positions=d)

    def to_torch(self):
        """
        Converts each array or float in the original layer positions to be a torch
        Tensor of the appropriate type
        """
        for pos in self.layer_positions.values():
            pos.coordinates = torch.from_numpy(pos.coordinates.astype(np.float32))
            pos.neighborhood_indices = torch.from_numpy(
                pos.neighborhood_indices.astype(int)
            )
            pos.neighborhood_width = torch.tensor(pos.neighborhood_width)


def resolve_positions(
    model: torch.nn.Module, layer: str, position_dir: Optional[str]
) -> LayerPositions:
    """
    Tries to get positions from a model itself.
    If the model has positions but a position_dir is provided anyway at the command
    line, raises a warning but doesn't block execution.
    """

    def _load_from_position_dir(dir_str: str) -> LayerPositions:
        position_dir = Path(dir_str)
        assert position_dir.is_dir()
        return LayerPositions.load(position_dir / f"{layer}.pkl")

    # access positions from saved model
    positions: Optional[Dict[str, LayerPositions]] = getattr(model, "positions", None)

    # if the model doesn't have positions, we need to make sure that position
    # dir was provided and exists
    if positions is None:
        assert position_dir is not None
        logger.info("Positions resolved to position_dir")
        return _load_from_position_dir(position_dir)

    # if the model did have positions, check if there's an override dir
    # provided
    if position_dir is not None:
        logger.warning(
            (
                "Loaded model had positions, but a position dir was also specified at"
                "the command line as an override; resolved to position_dir"
            )
        )
        return _load_from_position_dir(position_dir)

    # finally, if the model has positions and no override provided, just return those
    logger.info("Positions resolved to model.positions")
    return positions[layer]
