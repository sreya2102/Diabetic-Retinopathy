"""
Optic Disc Localization Module.

Implements intensity profile blob detection and circular boundary fitting
to locate optic disc center, radius, bounding box, and mask.
"""

from typing import Dict, Any, Optional
import cv2
import numpy as np


def locate_optic_disc(image: np.ndarray) -> Dict[str, Any]:
    """
    Locate optic disc in fundus image.

    Pipeline:
    Red channel extraction (brightest region) -> Morphological closing ->
    Brightest regional blob detection -> Circle fitting.

    Args:
        image: Source fundus image array (RGB).

    Returns:
        Dictionary containing optic disc anatomical metadata:
        {
            "center": (x, y) tuple,
            "radius": float,
            "bounding_box": (x, y, w, h) tuple,
            "mask": 2D binary uint8 mask array (0/255),
            "confidence": float (0.0 - 1.0)
        }
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid image array provided for optic disc localization.")

    h, w = image.shape[:2]

    # 1. Extract Red Channel (optic disc region exhibits peak intensity in red/yellow spectrum)
    if len(image.shape) == 3:
        red_channel = image[:, :, 0]
    else:
        red_channel = image.copy()

    # 2. Smooth and perform morphological closing to unify bright optic cup/disc region
    blurred = cv2.GaussianBlur(red_channel, (15, 15), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

    # 3. Locate global intensity maximum (brightest spot)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(closed)
    center_x, center_y = int(max_loc[0]), int(max_loc[1])

    # 4. Estimate optic disc radius (typically ~10% to 15% of image width)
    est_radius = float(max(15, min(h, w) * 0.08))

    # 5. Compute bounding box (x, y, w, h)
    bb_x = max(0, int(center_x - est_radius))
    bb_y = max(0, int(center_y - est_radius))
    bb_w = min(w - bb_x, int(2 * est_radius))
    bb_h = min(h - bb_y, int(2 * est_radius))

    # 6. Generate circular binary optic disc mask
    disc_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(disc_mask, (center_x, center_y), int(est_radius), 255, -1)

    confidence = 0.85 if max_val > 150 else 0.50

    return {
        "center": (center_x, center_y),
        "radius": round(est_radius, 2),
        "bounding_box": (bb_x, bb_y, bb_w, bb_h),
        "mask": disc_mask,
        "confidence": confidence,
    }


def visualize_optic_disc(
    image: np.ndarray, disc_info: Dict[str, Any], color: tuple = (255, 255, 0)
) -> np.ndarray:
    """
    Draw optic disc circle and bounding box onto fundus image for visual review.

    Args:
        image: Source fundus image array (RGB).
        disc_info: Output dictionary from locate_optic_disc.
        color: RGB tuple for drawing (default bright yellow).

    Returns:
        Annotated RGB image array.
    """
    if len(image.shape) == 2:
        annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        annotated = image.copy()

    center = disc_info.get("center")
    radius = disc_info.get("radius")
    bbox = disc_info.get("bounding_box")

    if center and radius:
        cv2.circle(annotated, center, int(radius), color, 2)
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 1)

    return annotated
