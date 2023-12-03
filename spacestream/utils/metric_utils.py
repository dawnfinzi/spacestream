import numpy as np


def gaussian_2d(positions: np.ndarray, center, sigma: float) -> np.ndarray:
    """
    Inputs:
        positions: N x 2
        center: [center_x, center_y]
        sigma: spread of gaussian
    """
    sigma_sq = sigma**2
    return (
        1.0
        / (2.0 * np.pi * sigma_sq)
        * np.exp(
            -(
                (positions[:, 0] - center[0]) ** 2.0 / (2.0 * sigma_sq)
                + (positions[:, 1] - center[1]) ** 2.0 / (2.0 * sigma_sq)
            )
        )
    )


def smooth(coordinates, data, n_anchors=100, sigma=0.1):
    anchors = np.linspace(np.min(coordinates), np.max(coordinates), n_anchors)
    cx, cy = np.meshgrid(anchors, anchors)

    smoothed = np.zeros((n_anchors, n_anchors))

    for row in range(n_anchors):
        for col in range(n_anchors):
            center = (cx[row, col], cy[row, col])
            dist_from_center = gaussian_2d(coordinates, center, sigma=0.1)
            weighted = np.average(data, weights=dist_from_center)
            smoothed[-row, col] = weighted

    return smoothed
