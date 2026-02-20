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


def iter_batches(indices, batch_size):
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        yield indices[start:end]


def frob_sq(mat):
    return np.sum(mat.astype(np.float64) ** 2)


def compute_streaming_means(
    keys,
    stim_indices,
    batch_size,
    extract_batch_fn,
    expected_dims=None,
):
    running_sum = {
        key: np.zeros(expected_dims[key], dtype=np.float64) if expected_dims and expected_dims[key] is not None else None
        for key in keys
    }
    means = {}
    total = 0
    for batch_idx in iter_batches(stim_indices, batch_size):
        batch_feats = extract_batch_fn(batch_idx)
        if total == 0:
            for key in keys:
                if key not in batch_feats:
                    raise KeyError(f"Missing key '{key}' from extract_batch_fn output.")
                if batch_feats[key].ndim != 2:
                    raise ValueError(f"Expected 2D feature matrix for key='{key}', got shape {batch_feats[key].shape}.")
                feat_dim = batch_feats[key].shape[1]
                if expected_dims and expected_dims[key] is not None and feat_dim != expected_dims[key]:
                    raise ValueError(
                        f"Unexpected feature width for key='{key}': got {feat_dim}, expected {expected_dims[key]}"
                    )
                if running_sum[key] is None:
                    running_sum[key] = np.zeros(feat_dim, dtype=np.float64)
        for key in keys:
            if expected_dims and expected_dims[key] is not None and batch_feats[key].shape[1] != expected_dims[key]:
                raise ValueError(
                    f"Unexpected feature width during mean pass for key='{key}': "
                    f"got {batch_feats[key].shape[1]}, expected {expected_dims[key]}"
                )
            running_sum[key] += batch_feats[key].sum(axis=0)
        total += batch_idx.shape[0]

    for key in keys:
        means[key] = running_sum[key] / float(total)
    return means


def accumulate_centered_covariances(
    keys,
    cross_pairs,
    stim_indices,
    batch_size,
    extract_batch_fn,
    means,
    expected_dims,
    cov_dtype=np.float32,
):
    cov_dtype = np.dtype(cov_dtype)
    centered_dtype = np.float16 if cov_dtype == np.float16 else np.float32

    xx = {
        key: np.zeros((expected_dims[key], expected_dims[key]), dtype=cov_dtype)
        for key in keys
    }
    xy = {
        pair: np.zeros((expected_dims[pair[0]], expected_dims[pair[1]]), dtype=cov_dtype)
        for pair in cross_pairs
    }

    for batch_idx in iter_batches(stim_indices, batch_size):
        batch_feats = extract_batch_fn(batch_idx)
        centered = {}
        for key in keys:
            feat = batch_feats[key]
            if feat.shape[1] != expected_dims[key]:
                raise ValueError(
                    f"Unexpected feature width during covariance pass for key='{key}': "
                    f"got {feat.shape[1]}, expected {expected_dims[key]}"
                )
            centered[key] = feat.astype(centered_dtype)
            centered[key] -= means[key].astype(centered_dtype)

        for key in keys:
            xx[key] += centered[key].T @ centered[key]
        for left_key, right_key in cross_pairs:
            xy[(left_key, right_key)] += centered[left_key].T @ centered[right_key]

    return xx, xy


def linear_cka_from_covariances(xx, xy):
    cka = {}
    for (left_key, right_key), cross_cov in xy.items():
        denom = np.sqrt(frob_sq(xx[left_key]) * frob_sq(xx[right_key]))
        cka[(left_key, right_key)] = float(frob_sq(cross_cov) / denom) if denom > 0 else np.nan
    return cka
