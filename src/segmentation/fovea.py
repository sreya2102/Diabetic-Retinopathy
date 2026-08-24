"""
Fovea Localization Module.

Implements anatomical spatial constraint searching and dark avascular region minimum intensity search
to pinpoint the macula / fovea center coordinates.
"""

from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np


def locate_fovea(
    image: np.ndarray, optic_disc_center: Optional[Tuple[int, int]] = None
) -> Dict[str, Any]:
    """
    Locate fovea / macula center coordinates in fundus image.

    Anatomical Constraints:
    - Fovea is a dark, avascular circular region.
    - Located temporal to optic disc (~2.0 - 2.5 optic disc diameters horizontally).

    Args:
        image: Source fundus image array (RGB).
        optic_disc_center: Optional (x, y) coordinates of optic disc center to constrain search.

    Returns:
        Dictionary containing fovea anatomical metadata:
        {
            "center": (x, y) tuple,
            "radius": float,
            "confidence": float (0.0 - 1.0)
        }
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid image array provided for fovea localization.")

    h, w = image.shape[:2]

    # Convert to green channel (macula exhibits dark absorption profile)
    if len(image.shape) == 3:
        green = image[:, :, 1]
    else:
        green = image.copy()

    # Smooth image to eliminate vessel noise
    blurred = cv2.GaussianBlur(green, (25, 25), 0)

    if optic_disc_center is not None:
        od_x, od_y = optic_disc_center
        # Determine temporal search direction based on optic disc position
        if od_x > w // 2:
            # Optic disc on right side -> Fovea to the left
            search_x_min = max(0, int(od_x - w * 0.35))
            search_x_max = max(0, int(od_x - w * 0.10))
        else:
            # Optic disc on left side -> Fovea to the right
            search_x_min = min(w - 1, int(od_x + w * 0.10))
            search_x_max = min(w - 1, int(od_x + w * 0.35))

        search_y_min = max(0, int(od_y - h * 0.15))
        search_y_max = min(h - 1, int(od_y + h * 0.15))

        # Extract search ROI
        roi = blurred[search_y_min:search_y_max, search_x_min:search_x_max]
        if roi.size > 0:
            _, _, min_loc, _ = cv2.minMaxLoc(roi)
            fovea_x = search_x_min + min_loc[0]
            fovea_y = search_y_min + min_loc[1]
            confidence = 0.80
        else:
            fovea_x, fovea_y = int(w * 0.5), int(h * 0.5)
            confidence = 0.40
    else:
        # Default global dark spot search in center region of fundus
        center_roi = blurred[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)]
        _, _, min_loc, _ = cv2.minMaxLoc(center_roi)
        fovea_x = int(w * 0.25) + min_loc[0]
        fovea_y = int(h * 0.25) + min_loc[1]
        confidence = 0.50

    est_radius = float(max(10, min(h, w) * 0.04))

    return {
        "center": (int(fovea_x), int(fovea_y)),
        "radius": round(est_radius, 2),
        "confidence": confidence,
    }


def visualize_fovea(
    image: np.ndarray, fovea_info: Dict[str, Any], color: tuple = (255, 0, 255)
) -> np.ndarray:
    """
    Draw fovea crosshair and boundary circle onto fundus image for visual review.

    Args:
        image: Source fundus image array (RGB).
        fovea_info: Output dictionary from locate_fovea.
        color: RGB tuple for drawing (default bright magenta).

    Returns:
        Annotated RGB image array.
    """
    if len(image.shape) == 2:
        annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        annotated = image.copy()

    center = fovea_info.get("center")
    radius = fovea_info.get("radius")

    if center and radius:
        cx, cy = center
        r = int(radius)
        cv2.circle(annotated, (cx, cy), r, color, 2)
        # Draw crosshair
        cv2.line(annotated, (cx - r - 5, cy), (cx + r + 5, cy), color, 1)
        cv2.line(annotated, (cx, cy - r - 5), (cx, cy + r + 5), color, 1)

    return annotated
