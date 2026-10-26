"""
postprocessing.py
------------------
Morphological clean-up for binary change maps produced by the methods in
`methods.py`. Raw pixel-wise / clustering outputs are typically noisy:
isolated "salt" pixels scattered across unchanged regions (speckle), and
small holes inside otherwise-solid changed regions.

We apply a morphological *opening* (erosion followed by dilation) to strip
away small speckle noise, followed by a *closing* (dilation followed by
erosion) to fill small holes and smooth region boundaries.
"""

import cv2
import numpy as np


def clean_change_map(change_map, kernel_size=3):
    """Denoise a binary change map with morphological opening + closing.

    Parameters
    ----------
    change_map : np.ndarray
        (H, W) uint8 binary map, values in {0, 255}.
    kernel_size : int
        Side length of the square structuring element used for both
        operations (default: 3, i.e. a small 3x3 kernel so fine change
        boundaries are preserved while speckle is removed).

    Returns
    -------
    np.ndarray
        (H, W) uint8 binary map, values in {0, 255}, with speckle noise
        removed and small holes filled.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    opened = cv2.morphologyEx(change_map, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return closed.astype(np.uint8)
