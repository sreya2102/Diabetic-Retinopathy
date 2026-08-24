"""
Unit tests for retinal structure and lesion segmentation modules.
"""

import numpy as np
import pytest
from src.segmentation.fovea import locate_fovea
from src.segmentation.lesions import detect_lesions
from src.segmentation.optic_disc import locate_optic_disc
from src.segmentation.vessels import segment_vessels


def test_vessel_segmentation():
    """Verify segment_vessels produces binary mask matching image dimensions."""
    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    mask = segment_vessels(img)

    assert mask is not None
    assert mask.shape == (128, 128)


def test_optic_disc_localization():
    """Verify locate_optic_disc returns center and bounding box."""
    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    res = locate_optic_disc(img)

    assert isinstance(res, dict)
    assert "center" in res
    assert "bounding_box" in res


def test_fovea_localization():
    """Verify locate_fovea returns center coordinates."""
    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    res = locate_fovea(img)

    assert isinstance(res, dict)
    assert "center" in res


def test_lesions_detection():
    """Verify detect_lesions schema interface contract."""
    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    res = detect_lesions(img)

    assert isinstance(res, dict)
    assert "microaneurysms" in res
    assert "hemorrhages" in res
    assert "hard_exudates" in res
    assert "soft_exudates" in res
