"""
Fundus Image Enhancement Module.

Provides CLAHE contrast enhancement, green channel extraction,
illumination normalization, and noise reduction techniques.
"""

from typing import Dict, Any
import cv2
import numpy as np


def enhance_fundus(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> Dict[str, Any]:
    """
    Enhance fundus image using green channel isolation and CLAHE contrast adjustment.

    Args:
        image: Fundus image numpy array (RGB).
        clip_limit: Threshold for contrast limiting in CLAHE.
        tile_grid_size: Size of grid for histogram equalization.

    Returns:
        Dictionary containing:
        - "enhanced": Preprocessed primary RGB image array.
        - "green_channel": Extracted green channel array.
        - "clahe_green": Contrast-enhanced green channel array.
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid image array provided for enhancement.")

    # Convert grayscale input to 3-channel if necessary
    if len(image.shape) == 2:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = image.copy()

    # Extract green channel (highest contrast for retinal structures & lesions)
    green_channel = img_rgb[:, :, 1]

    # Apply CLAHE to green channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_green = clahe.apply(green_channel)

    # Reconstruct RGB with CLAHE-enhanced green channel
    enhanced_rgb = img_rgb.copy()
    enhanced_rgb[:, :, 1] = clahe_green

    return {
        "enhanced": enhanced_rgb,
        "green_channel": green_channel,
        "clahe_green": clahe_green,
    }
