"""
Unit tests for fundus enhancement and normalization.
"""

import numpy as np
import pytest
from src.preprocessing.enhancement import enhance_fundus, preprocess_fundus
from src.preprocessing.normalization import normalize_fundus


def test_preprocess_fundus():
    """Verify preprocess_fundus returns enhanced array and all intermediate components."""
    synthetic_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    res = preprocess_fundus(synthetic_img)

    assert isinstance(res, dict)
    assert "enhanced" in res
    assert "green_channel" in res
    assert "illumination_bg" in res
    assert "clahe_green" in res
    assert "denoised_green" in res

    assert res["enhanced"].shape == synthetic_img.shape
    assert res["green_channel"].shape == (128, 128)
    assert res["illumination_bg"].shape == (128, 128)
    assert res["clahe_green"].shape == (128, 128)
    assert res["denoised_green"].shape == (128, 128)


def test_enhance_fundus_alias():
    """Verify enhance_fundus produces identical dictionary output as preprocess_fundus."""
    synthetic_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    res1 = preprocess_fundus(synthetic_img)
    res2 = enhance_fundus(synthetic_img)

    assert res1["enhanced"].shape == res2["enhanced"].shape
    assert np.array_equal(res1["green_channel"], res2["green_channel"])


def test_preprocess_fundus_invalid_input():
    """Verify preprocess_fundus raises ValueError on empty or invalid inputs."""
    with pytest.raises(ValueError):
        preprocess_fundus(np.array([]))


def test_normalize_fundus():
    """Verify normalize_fundus scales and resizes image correctly."""
    synthetic_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    norm = normalize_fundus(synthetic_img, target_size=(256, 256), standard_scaling=True)

    assert norm.shape == (256, 256, 3)
    assert norm.dtype == np.float32
    assert norm.max() <= 1.0
    assert norm.min() >= 0.0
