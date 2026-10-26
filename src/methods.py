"""
methods.py
----------
Unsupervised change-detection algorithms. Each function takes two
*preprocessed* RGB images (same shape, already resized + histogram
matched -- see `preprocessing.py`) and returns a binary change map:
an (H, W) uint8 array with values in {0, 255}, where 255 marks "changed".

Implemented methods:
    1. image_difference_otsu  -- grayscale absolute difference + Otsu.
    2. change_vector_analysis -- CVA: Euclidean magnitude of the RGB
                                  difference vector + Otsu.
    3. pca_kmeans              -- Celik (2009): PCA feature space built
                                  from the difference image, clustered
                                  with K-Means (k=2). This is the main
                                  method used in this project.
"""

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import KMeans

from .preprocessing import to_grayscale


def image_difference_otsu(img_a, img_b):
    """Simple grayscale absolute-difference change detector.

    Steps: convert both images to grayscale, take the absolute pixel-wise
    difference, then threshold it automatically with Otsu's method (which
    picks the threshold that best separates the difference histogram into
    two classes: "no change" / "change").

    Parameters
    ----------
    img_a, img_b : np.ndarray
        Preprocessed RGB (or grayscale) images of identical shape.

    Returns
    -------
    np.ndarray
        (H, W) uint8 binary change map, values in {0, 255}.
    """
    gray_a = to_grayscale(img_a)
    gray_b = to_grayscale(img_b)

    diff = cv2.absdiff(gray_a, gray_b)
    _, binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary.astype(np.uint8)


def change_vector_analysis(img_a, img_b):
    """Change Vector Analysis (CVA).

    Treats each pixel's RGB triplet as a vector in 3D color space. The
    "change vector" at a pixel is the difference between its vectors in
    image A and image B; its Euclidean norm (magnitude) summarizes how
    much that pixel changed across *all* color channels jointly (unlike
    a single-channel grayscale difference). The resulting magnitude map
    is then thresholded with Otsu's method.

    Parameters
    ----------
    img_a, img_b : np.ndarray
        Preprocessed RGB images of identical shape, (H, W, 3).

    Returns
    -------
    np.ndarray
        (H, W) uint8 binary change map, values in {0, 255}.
    """
    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)
    diff = a - b

    if diff.ndim == 3:
        magnitude = np.sqrt(np.sum(diff ** 2, axis=2))
    else:
        magnitude = np.abs(diff)

    # Rescale magnitude to the full 0-255 range before Otsu thresholding,
    # since Otsu (via OpenCV) expects an 8-bit single-channel input.
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude_u8 = magnitude_norm.astype(np.uint8)

    _, binary = cv2.threshold(magnitude_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary.astype(np.uint8)


def pca_kmeans(img_a, img_b, block_size=4, variance_threshold=0.95, random_state=0):
    """PCA + K-Means change detection (Celik, 2009).

    Reference: T. Celik, "Unsupervised Change Detection in Satellite
    Images Using Principal Component Analysis and k-Means Clustering",
    IEEE Geoscience and Remote Sensing Letters, 2009.

    Algorithm:
        1. Build the (grayscale) difference image between A and B.
        2. Partition it into non-overlapping `block_size` x `block_size`
           blocks and run PCA on this set of blocks to learn an
           orthonormal eigenvector basis ("feature space") for local
           difference patterns. Keep enough leading eigenvectors to
           explain `variance_threshold` of the total variance.
        3. Slide a `block_size` x `block_size` window over *every* pixel
           (overlapping neighborhoods) and project each, mean-centered,
           onto the eigenvector basis to obtain a per-pixel feature
           vector. This captures local texture/edge structure of the
           difference image around each pixel, not just its raw value.
        4. Cluster all per-pixel feature vectors into 2 groups with
           K-Means. The cluster whose pixels have the larger mean
           difference magnitude is labeled "change".

    Parameters
    ----------
    img_a, img_b : np.ndarray
        Preprocessed RGB (or grayscale) images of identical shape.
    block_size : int
        Side length of the square block used for both the PCA basis and
        the per-pixel feature neighborhoods (Celik's default: 4).
    variance_threshold : float
        Fraction of total variance the retained principal components
        must explain (controls how many eigenvectors are kept).
    random_state : int
        Seed for K-Means, for reproducible results.

    Returns
    -------
    np.ndarray
        (H, W) uint8 binary change map, values in {0, 255}.
    """
    gray_a = to_grayscale(img_a).astype(np.float64)
    gray_b = to_grayscale(img_b).astype(np.float64)
    diff = np.abs(gray_a - gray_b)

    h = block_size
    H, W = diff.shape

    # --- Step 1: learn a PCA basis from non-overlapping blocks -----------
    n_rows = H // h
    n_cols = W // h
    cropped = diff[: n_rows * h, : n_cols * h]
    # Reshape into (n_blocks, h*h) without an explicit Python loop.
    blocks = (
        cropped.reshape(n_rows, h, n_cols, h)
        .swapaxes(1, 2)
        .reshape(n_rows * n_cols, h * h)
    )

    mean_vec = blocks.mean(axis=0)
    blocks_centered = blocks - mean_vec

    cov = np.cov(blocks_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Keep the smallest number of leading components that explain at
    # least `variance_threshold` of the total variance (at least 1).
    total_var = eigvals.sum()
    if total_var <= 0:
        # Degenerate case: no difference anywhere -> nothing is "changed".
        return np.zeros((H, W), dtype=np.uint8)
    cum_var = np.cumsum(eigvals) / total_var
    n_components = int(np.searchsorted(cum_var, variance_threshold) + 1)
    n_components = max(1, min(n_components, h * h))
    eigvecs_s = eigvecs[:, :n_components]

    # --- Step 2: project overlapping per-pixel neighborhoods -------------
    pad_before = h // 2
    pad_after = h - 1 - pad_before
    diff_padded = np.pad(diff, ((pad_before, pad_after), (pad_before, pad_after)), mode="reflect")

    windows = sliding_window_view(diff_padded, (h, h))  # shape (H, W, h, h)
    windows_flat = windows.reshape(H * W, h * h)
    windows_centered = windows_flat - mean_vec
    feature_vectors = windows_centered @ eigvecs_s  # (H*W, n_components)

    # --- Step 3: cluster feature vectors into 2 groups with K-Means ------
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(feature_vectors)

    # The cluster with the larger mean difference magnitude is "change".
    diff_flat = diff.flatten()
    mean_mag_0 = diff_flat[labels == 0].mean() if np.any(labels == 0) else -np.inf
    mean_mag_1 = diff_flat[labels == 1].mean() if np.any(labels == 1) else -np.inf
    change_label = 0 if mean_mag_0 > mean_mag_1 else 1

    change_map = (labels == change_label).reshape(H, W).astype(np.uint8) * 255
    return change_map
