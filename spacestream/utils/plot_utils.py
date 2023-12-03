# local imports
import sys

import matplotlib as mpl
import numpy as np

from spacestream.utils.general_utils import make_iterable


def remove_spines(axes, to_remove=None):
    """
    Removes spines from pyplot axis
    Inputs
        axes (list or np.ndarray): list of pyplot axes
        to_remove (str list): can be any combo of "top", "right", "left", "bottom"
    """

    # if axes is a 2d array, flatten
    if isinstance(axes, np.ndarray):
        axes = axes.ravel()

    # enforce inputs as list
    axes = make_iterable(axes)

    if to_remove is None:
        to_remove = ["top", "right"]

    for ax in axes:
        for spine in to_remove:
            ax.spines[spine].set_visible(False)


# colormaps for Nauhaus data
nauhaus_raw_colormaps = {}
nauhaus_raw_colormaps["angles"] = np.array(
    [
        np.array((190, 60, 60, 255)) / 255.0,
        np.array((210, 160, 60, 255)) / 255.0,
        np.array((230, 230, 70, 255)) / 255.0,
        np.array((85, 150, 85, 255)) / 255.0,
        np.array((90, 165, 185, 255)) / 255.0,
        np.array((70, 80, 140, 255)) / 255.0,
        np.array((170, 85, 140, 255)) / 255.0,
        np.array((190, 60, 60, 255)) / 255.0,
    ]
)
nauhaus_raw_colormaps["sfs"] = np.array(
    [
        np.array((80, 140, 185, 255)) / 255.0,
        np.array((90, 155, 190, 255)) / 255.0,
        np.array((110, 170, 106, 255)) / 255.0,
        np.array((160, 200, 85, 255)) / 255.0,
        np.array((220, 220, 70, 255)) / 255.0,
        np.array((210, 150, 60, 255)) / 255.0,
        np.array((200, 100, 50, 255)) / 255.0,
        np.array((190, 70, 50, 255)) / 255.0,
    ]
)
# nauhaus_raw_colormaps["colors"] = np.array([(0.8, 0.8, 0.8, 1), (0.4, 0.4, 0.4, 1)])
nauhaus_raw_colormaps["colors"] = np.array([(0.8, 0.8, 0.8, 1), (0.1, 0.1, 0.1, 1)])

nauhaus_colormaps = {}
for colormap_name, raw_colormap in nauhaus_raw_colormaps.items():
    cmap = mpl.colors.LinearSegmentedColormap.from_list(colormap_name, raw_colormap)
    nauhaus_colormaps[colormap_name] = cmap
