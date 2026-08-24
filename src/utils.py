"""
Common Utility Functions for Fundus Image Processing and Pipeline Utilities.
"""

import os
from typing import Optional, Tuple, Union
import cv2
import numpy as np


def validate_image(image: Union[np.ndarray, str]) -> Tuple[bool, Optional[str]]:
    """
    Validate that an input is a valid image or existing file path.

    Args:
        image: Image array (numpy ndarray) or file path string.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if isinstance(image, str):
        if not os.path.exists(image):
            return False, f"File path does not exist: {image}"
        img_array = cv2.imread(image)
        if img_array is None:
            return False, f"Failed to decode image file: {image}"
        return True, None

    elif isinstance(image, np.ndarray):
        if image.size == 0:
            return False, "Input image array is empty."
        if len(image.shape) not in (2, 3):
            return False, f"Invalid image dimensions: {image.shape}. Expected 2D or 3D array."
        return True, None

    return False, f"Unsupported input type: {type(image)}"


def load_image(image_input: Union[str, np.ndarray], color_mode: str = "RGB") -> np.ndarray:
    """
    Safely load or validate a fundus image array in specified color mode.

    Args:
        image_input: File path string or existing numpy array (BGR/RGB).
        color_mode: Desired output format ('RGB', 'BGR', or 'GRAY').

    Returns:
        Loaded image array as numpy ndarray.

    Raises:
        ValueError: If input image is invalid or file cannot be read.
    """
    is_valid, err = validate_image(image_input)
    if not is_valid:
        raise ValueError(f"Invalid image input: {err}")

    if isinstance(image_input, str):
        img_bgr = cv2.imread(image_input)
        if img_bgr is None:
            raise ValueError(f"Could not read image file: {image_input}")
    else:
        img_bgr = image_input.copy()

    if color_mode.upper() == "RGB":
        if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3:
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_bgr
    elif color_mode.upper() == "GRAY":
        if len(img_bgr.shape) == 3:
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return img_bgr
    elif color_mode.upper() == "BGR":
        return img_bgr
    else:
        raise ValueError(f"Unsupported color mode: {color_mode}")


def convert_color_space(image: np.ndarray, conversion_code: int) -> np.ndarray:
    """
    Wrapper for OpenCV color space conversion with input checks.

    Args:
        image: Source image numpy array.
        conversion_code: OpenCV conversion code (e.g. cv2.COLOR_RGB2GRAY).

    Returns:
        Converted image numpy array.
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid numpy array provided for color space conversion.")
    return cv2.cvtColor(image, conversion_code)
