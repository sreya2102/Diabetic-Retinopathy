"""
Unit tests for Phase 2 fundus image quality assessment module.
"""

import cv2
import numpy as np
import pytest
from src.preprocessing.quality import assess_image, assess_image_quality


def test_assess_image_quality_sharp_vs_blurry():
    """Verify quality assessment detects blur and marks blurred images ungradable."""
    # Create sharp synthetic fundus image with high edge contrast
    sharp_img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(sharp_img, (128, 128), 100, (150, 100, 50), -1)
    # Add high-frequency noise / grid for sharpness
    sharp_img[::10, :] = 200
    sharp_img[:, ::10] = 200

    sharp_res = assess_image(sharp_img)
    assert sharp_res["blur_score"] > 80.0

    # Artificially blur the image heavily using Gaussian blur
    blurry_img = cv2.GaussianBlur(sharp_img, (31, 31), 0)
    blurry_res = assess_image(blurry_img)

    assert blurry_res["blur_score"] < sharp_res["blur_score"]
    assert blurry_res["gradable"] is False
    assert any("blurry" in msg.lower() for msg in blurry_res["feedback"])


def test_assess_image_dark_underexposed():
    """Verify quality assessment flags excessively dark fundus images."""
    dark_img = np.ones((256, 256, 3), dtype=np.uint8) * 10  # Mean intensity = 10
    dark_res = assess_image(dark_img)

    assert dark_res["gradable"] is False
    assert any("darkness" in msg.lower() for msg in dark_res["feedback"])


def test_assess_image_overexposed():
    """Verify quality assessment flags excessively bright fundus images."""
    bright_img = np.ones((256, 256, 3), dtype=np.uint8) * 245  # Mean intensity = 245
    bright_res = assess_image(bright_img)

    assert bright_res["gradable"] is False
    assert any("brightness" in msg.lower() or "overexposed" in msg.lower() for msg in bright_res["feedback"])


def test_assess_image_alias():
    """Verify assess_image_quality is identical to assess_image."""
    img = np.random.randint(40, 180, (128, 128, 3), dtype=np.uint8)
    res1 = assess_image(img)
    res2 = assess_image_quality(img)

    assert res1 == res2


def test_assess_image_invalid_input():
    """Verify ValueError raised for empty array or invalid inputs."""
    with pytest.raises(ValueError):
        assess_image(np.array([]))

    with pytest.raises(ValueError):
        assess_image(None)
