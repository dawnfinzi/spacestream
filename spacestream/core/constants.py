"""
Constants that might be used by multiple scripts
"""

import numpy as np

# how many times each image was shown
N_REPEATS = 3
SUBJECTS = ["01", "02", "03", "04", "05", "06", "07", "08"]

# layer names for different models
RESNET18_LAYERS = (
    ["relu", "maxpool"]
    + ["layer1.0", "layer1.1"]
    + ["layer2.0", "layer2.1"]
    + ["layer3.0", "layer3.1"]
    + ["layer4.0", "layer4.1"]
    + ["avgpool"]
)
RESNET50_LAYERS = (
    ["relu", "maxpool"]
    + [f"layer1.{i}" for i in range(3)]
    + [f"layer2.{i}" for i in range(4)]
    + [f"layer3.{i}" for i in range(6)]
    + [f"layer4.{i}" for i in range(3)]
    + ["avgpool"],
)
RESNET101_LAYERS = (
    ["relu", "maxpool"]
    + [f"layer1.{i}" for i in range(3)]
    + [f"layer2.{i}" for i in range(4)]
    + [f"layer3.{i}" for i in range(23)]
    + [f"layer4.{i}" for i in range(3)]
    + ["avgpool"],
)
SLOWFAST_LAYERS = [
    "blocks.1.multipathway_blocks.0.res_blocks.2",  # slow
    "blocks.1.multipathway_blocks.1.res_blocks.2",  # fast
    "blocks.2.multipathway_blocks.0.res_blocks.2",  # slow
    "blocks.2.multipathway_blocks.1.res_blocks.2",  # fast
    "blocks.3.multipathway_blocks.0.res_blocks.2",  # slow
    "blocks.3.multipathway_blocks.1.res_blocks.2",  # fast
    "blocks.4.multipathway_blocks.0.res_blocks.2",  # slow
    "blocks.4.multipathway_blocks.1.res_blocks.2",  # fast
    "blocks.5",
    "blocks.6.proj",
]
MATCHING_SLOWFAST_LAYERS = (
    ["blocks.0.multipathway_blocks.0", "blocks.0.multipathway_blocks.1"]
    + [f"blocks.1.multipathway_blocks.0.res_blocks.{i}" for i in range(3)]  # slow
    + [f"blocks.1.multipathway_blocks.1.res_blocks.{i}" for i in range(3)]  # fast
    + [f"blocks.2.multipathway_blocks.0.res_blocks.{i}" for i in range(4)]  # slow
    + [f"blocks.2.multipathway_blocks.1.res_blocks.{i}" for i in range(4)]  # fast
    + [f"blocks.3.multipathway_blocks.0.res_blocks.{i}" for i in range(6)]  # slow
    + [f"blocks.3.multipathway_blocks.1.res_blocks.{i}" for i in range(6)]  # fast
    + [f"blocks.4.multipathway_blocks.0.res_blocks.{i}" for i in range(3)]  # slow
    + [f"blocks.4.multipathway_blocks.1.res_blocks.{i}" for i in range(3)]  # fast
    + ["blocks.5", "blocks.6.proj"]
)
X3D_LAYERS = (
    ["blocks.0"]
    + [f"blocks.1.res_blocks.{i}" for i in range(3)]
    + [f"blocks.2.res_blocks.{i}" for i in range(5)]
    + [f"blocks.3.res_blocks.{i}" for i in range(10)]
    + [f"blocks.4.res_blocks.{i}" for i in range(7)]
    + ["blocks.5.proj"]
)
SPACETORCH_LAYERS = (
    ["base_model.conv1", "base_model.maxpool"]
    + ["base_model.layer1.0", "base_model.layer1.1"]
    + ["base_model.layer2.0", "base_model.layer2.1"]
    + ["base_model.layer3.0", "base_model.layer3.1"]
    + ["base_model.layer4.0", "base_model.layer4.1"]
    + ["base_model.avgpool"]
)

# list of areas included in streams ROIs
ROI_NAMES = [
    "Unknown",
    "Early",
    "Midventral",
    "Midlateral",
    "Midparietal",
    "Ventral",
    "Lateral",
    "Parietal",
]
CORE_ROI_NAMES = ["Ventral",
    "Lateral",
    "Dorsal"] #renaming Parietal to Dorsal for consistency with literature

# Color palette for stream ROIs
ROI_COLORS = ["#006600","#00008b",'#990000'] #color palette is reversed ordered (Dorsal, Lateral, Ventral)

# Pulled from Visual Cheese
# SKLEARN TRANSFER CONSTANTS
SVM_CV_C = [5e-2, 5e-1, 1.0, 5e1, 5e2, 5e3, 5e4]
SVM_CV_C_LONG = [
    1e-8,
    5e-8,
    1e-7,
    5e-7,
    1e-6,
    5e-6,
    1e-5,
    5e-5,
    1e-4,
    5e-4,
    1e-3,
    5e-3,
    1e-2,
    5e-2,
    1e-1,
    5e-1,
    1,
    1e1,
    5e1,
    1e2,
    5e2,
    1e3,
    5e3,
    1e4,
    5e4,
    1e5,
    5e5,
    1e6,
    5e6,
    1e7,
    5e7,
    1e8,
    5e8,
]
RIDGE_CV_ALPHA = [0.01, 0.1, 1, 10]
RIDGE_CV_ALPHA_LONG = list(1.0 / np.array(SVM_CV_C_LONG))

# Spacetorch constants
# how many degrees is the visual input?
DVA_PER_IMAGE = 8.0

# angles and spatial frequencies used most often
DEFAULT_ANGLES = np.linspace(0, 180, 9)[:-1]  # degrees
DEFAULT_SFS = np.linspace(5.5, 110.0, 9)[:-1]  # cycles per image
DEFAULT_PHASES = np.linspace(0, 2 * np.pi, 6)[:-1]

# estimate of how big, in mm, V1 in a single hemisphere is
# see here for math: https://github.com/neuroailab/spacenet/issues/21
V1_SIZE = 24.5  # mm

# estimate of how big, in mm, IT in a single hemisphere is
# based on assumptions made by Hyo/DiCarlo/et al
IT_SIZE = 10.0  # mm

# these are blind guesses, borderline irresponsible
RETINA_SIZE = 4.0  # mm, estimate from https://www.ncbi.nlm.nih.gov/pubmed/23500068
V2_SIZE = 20.0  # mm
V4_SIZE = 16.0  # mm

# Checkpoint naming convention
SW_PATH_STR_MAPPING = {
    0.0: "lw0_",
    0.1: "lw01_",
    0.25: "",
    0.5: "lwx2_",
    1.25: "lwx5_",
    2.5: "lwx10_",
    25.0: "lwx100_",
}
