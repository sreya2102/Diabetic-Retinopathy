"""
Unit tests for fundus enhancement and normalization.
"""

import numpy as np
import pytest
from src.preprocessing.enhancement import enhance_fundus
from src.preprocessing.normalization import normalize_fundus


def test_enhance_fundus():
    """Verify enhance_fundus returns enhanced array and green channel components."""
    synthetic_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    res = enhance_fundus(synthetic_img)

    assert isinstance(res, dict)
    assert "enhanced" in res
    assert "green_channel" in res
    assert "clahe_green" in res
    assert res["enhanced"].shape == synthetic_img.shape


def test_normalize_fundus():
    """Verify normalize_fundus scales and resizes image correctly."""
    synthetic_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    norm = normalize_fundus(synthetic_img, target_size=(256, 256), standard_scaling=True)

    assert norm.shape == (256, 256, 3)
    assert norm.dtype == np.float32
    assert norm.max() <= 1.0
    assert norm.min() >= 0.0
