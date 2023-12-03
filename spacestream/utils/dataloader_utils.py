"""
Dataloader utils for project
(highly inspired by/credit to: https://github.com/neuroailab/VisualCheese) 
"""

import numpy as np


def duplicate_channels(gray_images):
    """
    Converts single channel grayscale images into rgb channel images
    Input:
        gray_images : (N,H,W)
    Output:
        rgb : (N,H,W,3)
    """
    n, dim0, dim1 = gray_images.shape[:3]
    rgb = np.empty((n, dim0, dim1, 3), dtype=np.uint8)
    rgb[:, :, :, 2] = rgb[:, :, :, 1] = rgb[:, :, :, 0] = gray_images
    return rgb
