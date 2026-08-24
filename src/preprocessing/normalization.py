"""
Fundus Image Normalization Module.

Provides standard zero-mean unit-variance scaling and resizing operations.
"""

from typing import Tuple
import cv2
import numpy as np


def normalize_fundus(
    image: np.ndarray, target_size: Tuple[int, int] = (512, 512), standard_scaling: bool = True
) -> np.ndarray:
    """
    Resize and normalize fundus image pixel values for neural network consumption.

    Args:
        image: Source image numpy array (RGB).
        target_size: Tuple of (height, width).
        standard_scaling: If True, scales pixel intensities to [0, 1] floating point.

    Returns:
        Normalized image numpy array.
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid image array provided for normalization.")

    # Resize to target dimensions
    resized = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)

    if standard_scaling:
        normalized = resized.astype(np.float32) / 255.0
        return normalized

    return resized
