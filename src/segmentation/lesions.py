"""
Diabetic Retinopathy Lesion Analysis & Detection Module.

Implements detection and segmentation for key DR clinical lesions:
- Microaneurysms (MA): Tiny red punctate spots (initial sign of NPDR)
- Hemorrhages (HE): Larger intraretinal blood spots
- Hard Exudates (EX): Bright yellowish lipid deposits
- Soft Exudates (SE / Cotton Wool Spots): Pale fluffy white nerve layer micro-infarcts
"""

from typing import Dict, Any, List
import cv2
import numpy as np

from src.segmentation.optic_disc import locate_optic_disc
from src.segmentation.vessels import segment_vessels


def detect_lesions(image: np.ndarray) -> Dict[str, Any]:
    """
    Detect, count, and segment diabetic retinopathy lesions in fundus image.

    Pipeline:
    1. Extract anatomical masks (vessels, optic disc) to exclude normal structures.
    2. Dark lesion detection (MAs & HEs) via Bottom-Hat morphology on green channel.
    3. Bright lesion detection (Hard & Soft Exudates) via L*a*b* & red/green thresholding.

    Args:
        image: Source fundus image array (RGB).

    Returns:
        Structured dictionary containing lesion counts, 2D binary masks (uint8 0/255),
        and bounding box lists:
        {
            "microaneurysms": {"count": int, "mask": ndarray, "bounding_boxes": List},
            "hemorrhages": {"count": int, "mask": ndarray, "bounding_boxes": List},
            "hard_exudates": {"count": int, "mask": ndarray, "bounding_boxes": List},
            "soft_exudates": {"count": int, "mask": ndarray, "bounding_boxes": List},
            "neovascularization": {"present": bool, "mask": ndarray},
            "status": str
        }
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid image array provided for lesion detection.")

    h, w = image.shape[:2]

    # Convert 2D grayscale to 3-channel RGB if needed
    if len(image.shape) == 2:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = image.copy()

    # Step 1: Anatomical Masking (Exclude blood vessels & optic disc to prevent false positives)
    vessel_mask = segment_vessels(img_rgb)
    disc_info = locate_optic_disc(img_rgb)
    disc_mask = disc_info.get("mask", np.zeros((h, w), dtype=np.uint8))

    # Dilate vessel & disc masks slightly for safe exclusion margin
    vessel_dilated = cv2.dilate(vessel_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    disc_dilated = cv2.dilate(disc_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    exclusion_mask = cv2.bitwise_or(vessel_dilated, disc_dilated)

    # Step 2: Dark Lesion Candidate Detection (Microaneurysms & Hemorrhages)
    green = img_rgb[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_green = clahe.apply(green)

    # Bottom-Hat morphological transform highlights small dark lesions
    kernel_dark = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    bottom_hat = cv2.morphologyEx(enhanced_green, cv2.MORPH_BLACKHAT, kernel_dark)

    # Threshold dark candidates and mask out vessels/disc
    _, dark_thresh = cv2.threshold(bottom_hat, 20, 255, cv2.THRESH_BINARY)
    dark_candidates = cv2.bitwise_and(dark_thresh, cv2.bitwise_not(exclusion_mask))

    # Separate Microaneurysms (small, circular <= 15 px) vs Hemorrhages (larger > 15 px)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_candidates)

    ma_mask = np.zeros((h, w), dtype=np.uint8)
    he_mask = np.zeros((h, w), dtype=np.uint8)
    ma_boxes: List[tuple] = []
    he_boxes: List[tuple] = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x, y, bw, bh = (
            stats[i, cv2.CC_STAT_LEFT],
            stats[i, cv2.CC_STAT_TOP],
            stats[i, cv2.CC_STAT_WIDTH],
            stats[i, cv2.CC_STAT_HEIGHT],
        )

        if 2 <= area <= 15:
            ma_mask[labels == i] = 255
            ma_boxes.append((x, y, bw, bh))
        elif 15 < area <= 500:
            he_mask[labels == i] = 255
            he_boxes.append((x, y, bw, bh))

    # Step 3: Bright Lesion Candidate Detection (Hard & Soft Exudates)
    # Convert to L*a*b* color space for luminance L* and yellow b* isolation
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    # Hard Exudates exhibit high luminance and high b* (yellowish hue)
    _, bright_thresh = cv2.threshold(l_chan, 180, 255, cv2.THRESH_BINARY)
    bright_candidates = cv2.bitwise_and(bright_thresh, cv2.bitwise_not(exclusion_mask))

    num_bright, bright_labels, bright_stats, _ = cv2.connectedComponentsWithStats(bright_candidates)

    ex_mask = np.zeros((h, w), dtype=np.uint8)
    se_mask = np.zeros((h, w), dtype=np.uint8)
    ex_boxes: List[tuple] = []
    se_boxes: List[tuple] = []

    for i in range(1, num_bright):
        area = bright_stats[i, cv2.CC_STAT_AREA]
        x, y, bw, bh = (
            bright_stats[i, cv2.CC_STAT_LEFT],
            bright_stats[i, cv2.CC_STAT_TOP],
            bright_stats[i, cv2.CC_STAT_WIDTH],
            bright_stats[i, cv2.CC_STAT_HEIGHT],
        )

        if 3 <= area <= 150:
            ex_mask[bright_labels == i] = 255
            ex_boxes.append((x, y, bw, bh))
        elif 150 < area <= 800:
            se_mask[bright_labels == i] = 255
            se_boxes.append((x, y, bw, bh))

    return {
        "microaneurysms": {
            "count": len(ma_boxes),
            "mask": ma_mask,
            "bounding_boxes": ma_boxes,
        },
        "hemorrhages": {
            "count": len(he_boxes),
            "mask": he_mask,
            "bounding_boxes": he_boxes,
        },
        "hard_exudates": {
            "count": len(ex_boxes),
            "mask": ex_mask,
            "bounding_boxes": ex_boxes,
        },
        "soft_exudates": {
            "count": len(se_boxes),
            "mask": se_mask,
            "bounding_boxes": se_boxes,
        },
        "neovascularization": {
            "present": False,
            "mask": np.zeros((h, w), dtype=np.uint8),
        },
        "status": "Classical Baseline Morphology Detection (IDRiD Calibration Benchmark)",
    }


def visualize_lesions(image: np.ndarray, lesions_data: Dict[str, Any]) -> np.ndarray:
    """
    Overlay detected DR lesion bounding boxes and masks onto fundus image.

    Color Codes:
    - Microaneurysms (MA): Red bounding boxes
    - Hemorrhages (HE): Dark Magenta bounding boxes
    - Hard Exudates (EX): Yellow bounding boxes
    - Soft Exudates (SE): Cyan bounding boxes

    Args:
        image: Source fundus image array (RGB).
        lesions_data: Result dictionary returned by detect_lesions.

    Returns:
        Annotated RGB image array.
    """
    if len(image.shape) == 2:
        annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        annotated = image.copy()

    # Draw Microaneurysms (Red)
    ma_boxes = lesions_data.get("microaneurysms", {}).get("bounding_boxes", [])
    for x, y, w, h in ma_boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 1)

    # Draw Hemorrhages (Dark Red / Magenta)
    he_boxes = lesions_data.get("hemorrhages", {}).get("bounding_boxes", [])
    for x, y, w, h in he_boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (180, 0, 90), 1)

    # Draw Hard Exudates (Yellow)
    ex_boxes = lesions_data.get("hard_exudates", {}).get("bounding_boxes", [])
    for x, y, w, h in ex_boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 255, 0), 1)

    # Draw Soft Exudates (Cyan)
    se_boxes = lesions_data.get("soft_exudates", {}).get("bounding_boxes", [])
    for x, y, w, h in se_boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 1)

    return annotated
