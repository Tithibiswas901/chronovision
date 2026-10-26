"""
preprocessing.py
----------------
Utilities for loading and normalizing bi-temporal image pairs before
change detection.

Pipeline for a pair (Image 1 / "A", Image 2 / "B"):
    1. Load both images (and the ground-truth label, if present) from disk.
    2. Resize both images to a common size (256x256) so all methods and
       metrics operate on comparable grids regardless of source resolution.
    3. Histogram-match Image 2 (B) onto Image 1 (A). This cancels global
       illumination / seasonal color-balance differences between the two
       acquisition dates so that the remaining pixel differences are more
       likely to be caused by genuine land-cover change rather than by
       sensor or lighting conditions.

Note on dataset naming: in this LEVIR-CD mirror, folder A is the
post-change image and folder B is the pre-change image. The change-detection
math below is symmetric in A/B (none of it assumes a temporal direction),
so this ordering has no effect on the algorithms -- it only matters for how
the images are labeled in the UI.
"""

import os

import cv2
import numpy as np
from skimage.exposure import match_histograms

# Target size used throughout the pipeline so every method, metric and
# plot operates on a consistent grid.
TARGET_SIZE = (256, 256)  # (width, height)


def load_image(path):
    """Load an image from disk as RGB uint8.

    Parameters
    ----------
    path : str
        Path to an image file (PNG/JPG/...). Grayscale and RGBA sources are
        converted to standard 3-channel RGB.

    Returns
    -------
    np.ndarray
        Image array of shape (H, W, 3), dtype uint8, channel order RGB.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")

    if img.ndim == 2:
        # Grayscale source -> replicate into 3 channels.
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        # RGBA/BGRA -> drop alpha, convert to RGB.
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        # Standard 3-channel BGR (OpenCV default) -> RGB.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def load_label(path):
    """Load a binary ground-truth change mask as a single-channel uint8 array.

    Values are normalized so that "change" pixels are 255 and "no-change"
    pixels are 0, regardless of whether the source mask uses {0, 1} or
    {0, 255}.

    Parameters
    ----------
    path : str
        Path to the label/mask image.

    Returns
    -------
    np.ndarray
        Array of shape (H, W), dtype uint8, values in {0, 255}.
    """
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read label at: {path}")
    mask = (mask > 0).astype(np.uint8) * 255
    return mask


def resize_image(img, size=TARGET_SIZE):
    """Resize an image (or mask) to `size` = (width, height).

    Uses area interpolation for photographic images (good anti-aliased
    downsampling) and nearest-neighbor for already-binary masks so label
    values stay exactly {0, 255} after resizing.
    """
    is_binary_mask = img.ndim == 2 and set(np.unique(img).tolist()) <= {0, 255}
    interp = cv2.INTER_NEAREST if is_binary_mask else cv2.INTER_AREA
    return cv2.resize(img, size, interpolation=interp)


def to_grayscale(img):
    """Convert an RGB (or already-grayscale) image to single-channel grayscale.

    Safe to call on images that are already grayscale -- they are returned
    unchanged (after ensuring uint8 dtype).

    Parameters
    ----------
    img : np.ndarray
        Either (H, W) grayscale or (H, W, 3) RGB.

    Returns
    -------
    np.ndarray
        (H, W) uint8 grayscale image.
    """
    if img.ndim == 2:
        return img.astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 1:
        return img[:, :, 0].astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def histogram_match(source, reference):
    """Histogram-match `source` onto `reference`'s intensity distribution.

    This is used to cancel global lighting/seasonal color differences
    between the two acquisition dates before computing pixel-wise
    differences. Works for both grayscale and multi-channel (RGB) inputs.

    Parameters
    ----------
    source : np.ndarray
        Image whose histogram will be adjusted (e.g. Image 2 / "B").
    reference : np.ndarray
        Image whose histogram serves as the target distribution
        (e.g. Image 1 / "A").

    Returns
    -------
    np.ndarray
        `source` remapped to match `reference`'s histogram, same shape
        and dtype (uint8) as the input.
    """
    # match_histograms requires same number of dimensions; channel_axis=-1
    # handles multi-channel inputs, and is omitted for 2D grayscale.
    if source.ndim == 3:
        matched = match_histograms(source, reference, channel_axis=-1)
    else:
        matched = match_histograms(source, reference)
    return np.clip(matched, 0, 255).astype(np.uint8)


def load_and_preprocess_pair(path_a, path_b, path_label=None, size=TARGET_SIZE):
    """Load, resize, and histogram-match an (Image 1, Image 2) pair.

    Parameters
    ----------
    path_a : str
        Path to Image 1 ("A").
    path_b : str
        Path to Image 2 ("B"). This image is histogram-matched onto A.
    path_label : str, optional
        Path to the ground-truth binary change mask, if available.
    size : tuple(int, int)
        Target (width, height) for resizing. Default 256x256.

    Returns
    -------
    dict with keys:
        'a'       : resized RGB image A, uint8 (H, W, 3)
        'b'       : resized + histogram-matched RGB image B, uint8 (H, W, 3)
        'b_raw'   : resized RGB image B before histogram matching
        'label'   : resized binary label (H, W) uint8 {0,255}, or None
    """
    img_a = resize_image(load_image(path_a), size)
    img_b_raw = resize_image(load_image(path_b), size)
    img_b = histogram_match(img_b_raw, img_a)

    label = None
    if path_label is not None and os.path.exists(path_label):
        label = resize_image(load_label(path_label), size)

    return {"a": img_a, "b": img_b, "b_raw": img_b_raw, "label": label}
