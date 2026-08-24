"""
Fundus Image Quality Assessment Module.

Provides comprehensive assessment of fundus image quality factors:
- Focus and Sharpness / Blur (Laplacian Variance metric)
- Illumination quality & Exposure (Luminance mean and standard deviation)
- Retinal Field of View (FOV) & Usable retinal mask ratio
- Gradability determination with actionable recapture feedback
"""

from typing import Dict, Any, List, Union
import cv2
import numpy as np


# Documented Quality Thresholds
BLUR_THRESHOLD_LOW: float = 80.0       # Below this is considered significantly blurry
BLUR_THRESHOLD_GOOD: float = 200.0     # Sharp image threshold
DARKNESS_THRESHOLD: float = 35.0       # Mean luminance below this is underexposed/dark
BRIGHTNESS_THRESHOLD: float = 220.0    # Mean luminance above this is overexposed/bright
CONTRAST_THRESHOLD_LOW: float = 20.0   # Standard deviation below this is low contrast
FOV_RATIO_MIN: float = 0.30            # Minimum valid retinal area ratio
FOV_RATIO_MAX: float = 0.88            # Maximum valid retinal area ratio


def assess_image(image: np.ndarray) -> Dict[str, Any]:
    """
    Assess fundus image quality, exposure, blur, field of view, and gradability.

    Args:
        image: Input fundus image array (RGB or Grayscale).

    Returns:
        Structured dictionary matching API contract:
        {
            "score": float (0.0 - 1.0),
            "gradable": bool,
            "blur_score": float,
            "illumination_score": float,
            "field_of_view_score": float,
            "feedback": List[str]
        }
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid image array provided for quality assessment.")

    # Convert to grayscale if 3-channel RGB/BGR
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # 1. Focus / Blur Metric: Laplacian Variance
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 2. Illumination & Exposure Metrics: Intensity Mean & Std Dev
    illumination_score = float(np.mean(gray))
    contrast_score = float(np.std(gray))

    # 3. Retinal Field of View (FOV) & Usable Area Ratio Estimation
    # Threshold background noise (black border) to find retinal circular mask
    _, fov_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    field_of_view_score = float(np.count_nonzero(fov_mask) / gray.size)

    feedback: List[str] = []
    gradable = True

    # Check Blur / Sharpness
    if blur_score < BLUR_THRESHOLD_LOW:
        gradable = False
        feedback.append("Image too blurry - re-focus camera before acquisition.")
    elif blur_score < BLUR_THRESHOLD_GOOD:
        feedback.append("Borderline image sharpness - ensure steady focus.")

    # Check Exposure / Illumination
    if illumination_score < DARKNESS_THRESHOLD:
        gradable = False
        feedback.append("Excessive darkness - increase flash or illumination level.")
    elif illumination_score > BRIGHTNESS_THRESHOLD:
        gradable = False
        feedback.append("Excessive brightness / overexposed - reduce illumination level.")
    elif contrast_score < CONTRAST_THRESHOLD_LOW:
        feedback.append("Poor illumination contrast - check lighting settings.")

    # Check Retinal Field of View
    if field_of_view_score < FOV_RATIO_MIN:
        gradable = False
        feedback.append("Insufficient retinal field - center fundus on macula or optic disc.")

    # Composite Quality Score (0.0 to 1.0)
    # Normalized components: blur (0-1), illumination balance (0-1), FOV ratio (0-1)
    norm_blur = min(1.0, blur_score / BLUR_THRESHOLD_GOOD)
    norm_illum = 1.0 - abs(illumination_score - 128.0) / 128.0
    norm_fov = min(1.0, field_of_view_score / 0.5) if field_of_view_score < 0.5 else 1.0

    score = float(np.clip(0.5 * norm_blur + 0.3 * norm_illum + 0.2 * norm_fov, 0.0, 1.0))

    if gradable and len(feedback) == 0:
        feedback.append("Image quality acceptable for DR screening.")

    return {
        "score": round(score, 4),
        "gradable": gradable,
        "blur_score": round(blur_score, 2),
        "illumination_score": round(illumination_score, 2),
        "field_of_view_score": round(field_of_view_score, 4),
        "feedback": feedback,
    }


def assess_image_quality(image: np.ndarray) -> Dict[str, Any]:
    """
    Alias wrapper for assess_image for backward compatibility with pipeline contract.
    """
    return assess_image(image)
