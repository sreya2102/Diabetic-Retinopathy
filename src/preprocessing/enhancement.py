"""
Fundus Image Enhancement Module.

Provides comprehensive fundus image preprocessing:
- Green-channel extraction (optimal contrast for retinal structures & lesions)
- Illumination normalization & background shade correction
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Edge-preserving bilateral denoising to protect clinically relevant lesions (MAs, HEs, EXs)
- RGB image re-synthesis
"""

from typing import Dict, Any
import cv2
import numpy as np


def preprocess_fundus(
    image: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: tuple = (8, 8),
    apply_denoising: bool = True,
) -> Dict[str, Any]:
    """
    Enhance fundus image with illumination normalization, CLAHE, and edge-preserving noise reduction.

    Args:
        image: Source fundus image array (RGB).
        clip_limit: CLAHE contrast limiting parameter (default 2.5).
        tile_grid_size: Grid size for localized histogram equalization (default 8x8).
        apply_denoising: If True, applies edge-preserving bilateral filtering.

    Returns:
        Dictionary containing enhanced image and intermediate representations:
        {
            "enhanced": RGB enhanced image numpy array,
            "green_channel": Raw isolated green channel array,
            "illumination_bg": Estimated non-uniform background illumination map,
            "clahe_green": Contrast-enhanced green channel array,
            "denoised_green": Denoised contrast-enhanced green channel array
        }
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid image array provided for enhancement.")

    # Convert 2D grayscale input to 3-channel RGB if needed
    if len(image.shape) == 2:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = image.copy()

    # Step 1: Green Channel Extraction
    # Green channel provides maximum contrast for retinal vessels, microaneurysms, and hemorrhages
    green_channel = img_rgb[:, :, 1]

    # Step 2: Illumination Normalization / Background Correction
    # Estimate background illumination using large Gaussian kernel
    kernel_size = max(31, (min(green_channel.shape[:2]) // 8) | 1)
    illumination_bg = cv2.GaussianBlur(green_channel, (kernel_size, kernel_size), 0)

    # Background subtraction: subtract illumination variation and normalize around mid-gray (128)
    norm_green = cv2.addWeighted(green_channel, 1.0, illumination_bg, -1.0, 128)

    # Step 3: Contrast-Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_green = clahe.apply(norm_green)

    # Step 4: Edge-Preserving Denoising (Bilateral Filter)
    # Preserves crisp lesion/vessel boundaries while smoothing sensor noise
    if apply_denoising:
        denoised_green = cv2.bilateralFilter(clahe_green, d=5, sigmaColor=25, sigmaSpace=25)
    else:
        denoised_green = clahe_green.copy()

    # Step 5: Re-synthesize Enhanced 3-Channel RGB Image
    enhanced_rgb = img_rgb.copy()
    enhanced_rgb[:, :, 1] = denoised_green

    # Boost red channel contrast slightly for exudates (yellowish/white lesions)
    red_clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=tile_grid_size)
    enhanced_rgb[:, :, 0] = red_clahe.apply(img_rgb[:, :, 0])

    return {
        "enhanced": enhanced_rgb,
        "green_channel": green_channel,
        "illumination_bg": illumination_bg,
        "clahe_green": clahe_green,
        "denoised_green": denoised_green,
    }


def enhance_fundus(image: np.ndarray, clip_limit: float = 2.5, tile_grid_size: tuple = (8, 8)) -> Dict[str, Any]:
    """
    Alias wrapper for preprocess_fundus for backward compatibility.
    """
    return preprocess_fundus(image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
