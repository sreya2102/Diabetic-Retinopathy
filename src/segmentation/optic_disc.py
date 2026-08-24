"""
Optic Disc Localization Module.

Phase 1 Architecture Stub:
Defines interface for detecting center coordinates (x, y), radius, and bounding box of optic disc.
"""

from typing import Dict, Any, Optional
import cv2
import numpy as np


def locate_optic_disc(image: np.ndarray) -> Dict[str, Any]:
    """
    Locate optic disc in fundus image using intensity profile or segmentation model interface.

    Args:
        image: Fundus image numpy array (RGB).

    Returns:
        Dictionary containing:
        - "center": (x, y) coordinates or None
        - "radius": float radius in pixels or None
        - "bounding_box": (x, y, w, h) or None
        - "confidence": float localization confidence
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid image array provided for optic disc localization.")

    # Convert to red channel or grayscale (optic disc is bright region)
    if len(image.shape) == 3:
        red_channel = image[:, :, 0]
    else:
        red_channel = image

    # Baseline intensity-based bright spot detection
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(red_channel)

    return {
        "center": (int(max_loc[0]), int(max_loc[1])),
        "radius": 40.0,
        "bounding_box": (max(0, max_loc[0] - 40), max(0, max_loc[1] - 40), 80, 80),
        "confidence": 0.5,
    }
