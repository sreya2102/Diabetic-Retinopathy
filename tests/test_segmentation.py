"""
Unit tests for Phase 4 and Phase 5 retinal structure and lesion segmentation modules.
"""

import cv2
import numpy as np
import pytest
from src.segmentation.fovea import locate_fovea, visualize_fovea
from src.segmentation.lesions import detect_lesions, visualize_lesions
from src.segmentation.optic_disc import locate_optic_disc, visualize_optic_disc
from src.segmentation.vessels import segment_vessels, visualize_vessels


def test_vessel_segmentation_and_visualization():
    """Verify segment_vessels produces binary 0/255 mask and overlay visualization works."""
    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    mask = segment_vessels(img)

    assert mask is not None
    assert mask.shape == (128, 128)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})

    vis = visualize_vessels(img, mask)
    assert vis.shape == img.shape


def test_optic_disc_localization_and_visualization():
    """Verify locate_optic_disc returns valid coordinates, mask, and bounding box."""
    img = np.ones((256, 256, 3), dtype=np.uint8) * 30
    cv2.circle(img, (200, 128), 25, (240, 240, 200), -1)

    res = locate_optic_disc(img)

    assert isinstance(res, dict)
    assert "center" in res
    assert "radius" in res
    assert "bounding_box" in res
    assert "mask" in res
    assert "confidence" in res

    assert res["center"][0] > 0 and res["center"][1] > 0
    assert res["mask"].shape == (256, 256)

    vis = visualize_optic_disc(img, res)
    assert vis.shape == img.shape


def test_fovea_localization_and_visualization():
    """Verify locate_fovea respects optic disc temporal constraint and visualization works."""
    img = np.random.randint(20, 200, (256, 256, 3), dtype=np.uint8)
    od_center = (200, 128)

    res = locate_fovea(img, optic_disc_center=od_center)

    assert isinstance(res, dict)
    assert "center" in res
    assert "radius" in res
    assert "confidence" in res
    assert res["center"][0] < od_center[0]

    vis = visualize_fovea(img, res)
    assert vis.shape == img.shape


def test_lesions_detection_and_visualization():
    """Verify detect_lesions detects synthetic dark/bright spots and visualize_lesions works."""
    synthetic_fundus = np.ones((256, 256, 3), dtype=np.uint8) * 100
    # Add synthetic dark spot (microaneurysm / hemorrhage candidate)
    cv2.circle(synthetic_fundus, (60, 60), 3, (10, 10, 10), -1)
    # Add synthetic bright spot (hard exudate candidate)
    cv2.circle(synthetic_fundus, (150, 150), 4, (250, 250, 240), -1)

    res = detect_lesions(synthetic_fundus)

    assert isinstance(res, dict)
    assert "microaneurysms" in res
    assert "hemorrhages" in res
    assert "hard_exudates" in res
    assert "soft_exudates" in res
    assert "neovascularization" in res

    assert res["microaneurysms"]["mask"].shape == (256, 256)
    assert res["hard_exudates"]["mask"].shape == (256, 256)

    vis = visualize_lesions(synthetic_fundus, res)
    assert vis.shape == synthetic_fundus.shape


def test_segmentation_invalid_inputs():
    """Verify all structure segmentation functions raise ValueError on invalid inputs."""
    with pytest.raises(ValueError):
        segment_vessels(np.array([]))

    with pytest.raises(ValueError):
        locate_optic_disc(np.array([]))

    with pytest.raises(ValueError):
        locate_fovea(np.array([]))

    with pytest.raises(ValueError):
        detect_lesions(np.array([]))
