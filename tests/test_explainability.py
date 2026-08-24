"""
Unit tests for explainability modules (Grad-CAM, clinical evidence, confidence calibration).
"""

import numpy as np
from src.explainability.confidence import calibrate_confidence
from src.explainability.evidence import extract_clinical_evidence
from src.explainability.gradcam import generate_gradcam


def test_generate_gradcam_stub():
    """Verify generate_gradcam returns expected schema keys."""
    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    res = generate_gradcam(None, img)

    assert isinstance(res, dict)
    assert "heatmap" in res
    assert "overlay" in res
    assert "status" in res


def test_extract_clinical_evidence():
    """Verify evidence extraction parses lesion dictionary counts."""
    lesions_sample = {
        "microaneurysms": {"count": 12, "mask": None},
        "hemorrhages": {"count": 3, "mask": None},
        "hard_exudates": {"count": 0, "mask": None},
        "soft_exudates": {"count": 1, "mask": None},
    }
    evidence = extract_clinical_evidence(lesions_sample)

    assert isinstance(evidence, list)
    assert len(evidence) == 3  # MA, HE, SE
    assert evidence[0]["lesion_type"] == "microaneurysms"
    assert evidence[0]["count"] == 12


def test_calibrate_confidence():
    """Verify confidence calibration helper outputs."""
    res = calibrate_confidence([0.1, 0.1, 0.7, 0.05, 0.05], calibrated=False)
    assert res["confidence"] == 0.7
    assert res["is_calibrated"] is False
