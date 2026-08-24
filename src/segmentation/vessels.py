"""
Retinal Blood Vessel Segmentation Module.

Phase 1 Architecture Stub:
Defines interface for vessel tree segmentation. Classical matched filtering /
morphological opening or deep UNet segmentation will be integrated in Phase 4.
"""

from typing import Optional
import cv2
import numpy as np


def segment_vessels(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract retinal blood vessel tree mask.

    Args:
        image: Fundus image numpy array (RGB).

    Returns:
        Binary vessel mask array (2D uint8 0/255), or None if model unavailable.
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid image array provided for vessel segmentation.")

    # Phase 1: Baseline classical morphological vessel extraction interface
    if len(image.shape) == 3:
        green = image[:, :, 1]
    else:
        green = image

    # Basic contrast thresholding baseline for interface demonstration
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)

    # Top-hat morphological operation to highlight linear vessel structures
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    top_hat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)

    # Adaptive thresholding
    vessel_mask = cv2.adaptiveThreshold(
        top_hat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return vessel_mask
