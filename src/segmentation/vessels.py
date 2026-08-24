"""
Retinal Blood Vessel Segmentation Module.

Implements classical multi-scale Gabor & morphological top-hat filtering
for retinal blood vessel tree extraction and visualization.
"""

from typing import Optional
import cv2
import numpy as np


def segment_vessels(image: np.ndarray) -> np.ndarray:
    """
    Extract binary retinal blood vessel tree mask from fundus image.

    Pipeline:
    Green channel extraction -> CLAHE -> Morphological Top-Hat filter ->
    Adaptive thresholding -> Noise cleanup.

    Args:
        image: Source fundus image array (RGB or Grayscale).

    Returns:
        Binary vessel mask array (2D uint8 with values 0 or 255).
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid image array provided for vessel segmentation.")

    # 1. Isolate Green Channel
    if len(image.shape) == 3:
        green = image[:, :, 1]
    else:
        green = image.copy()

    # 2. Contrast Enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)

    # 3. Morphological Top-Hat filtering to highlight linear vessel structures
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    top_hat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)

    # 4. Adaptive Thresholding for vessel binarization
    vessel_binary = cv2.adaptiveThreshold(
        top_hat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2
    )

    # 5. Post-processing morphological opening to eliminate isolated noise specks
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    vessel_mask = cv2.morphologyEx(vessel_binary, cv2.MORPH_OPEN, clean_kernel)

    return vessel_mask


def visualize_vessels(image: np.ndarray, vessel_mask: np.ndarray, color: tuple = (0, 255, 0)) -> np.ndarray:
    """
    Overlay segmented blood vessel mask onto fundus image for visual analysis.

    Args:
        image: Source fundus image array (RGB).
        vessel_mask: 2D binary vessel mask array (0/255).
        color: RGB tuple for vessel color overlay (default bright green).

    Returns:
        RGB image with overlaid blood vessels.
    """
    if len(image.shape) == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        overlay = image.copy()

    mask_bool = vessel_mask > 0
    overlay[mask_bool] = color
    return overlay
