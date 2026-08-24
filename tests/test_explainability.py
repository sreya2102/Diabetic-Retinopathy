"""
Unit tests for Phase 7 & 8 explainability modules (Grad-CAM, clinical evidence, confidence calibration).
"""

import numpy as np
import pytest
import torch
from src.classification.model import DRClassifier
from src.explainability.confidence import calibrate_confidence
from src.explainability.evidence import extract_clinical_evidence
from src.explainability.gradcam import generate_gradcam


def test_generate_gradcam_pytorch_execution():
    """Verify generate_gradcam runs forward/backward pass on DRClassifier and outputs valid heatmap and overlay."""
    model = DRClassifier(num_classes=5)
    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

    res = generate_gradcam(model, img)

    assert isinstance(res, dict)
    assert "original" in res
    assert "heatmap" in res
    assert "overlay" in res
    assert "status" in res

    assert res["heatmap"].shape == (128, 128)
    assert res["overlay"].shape == (128, 128, 3)
    assert res["heatmap"].min() >= 0.0
    assert res["heatmap"].max() <= 1.0


def test_extract_clinical_evidence():
    """Verify clinical evidence parses real lesion detection counts."""
    lesions_sample = {
        "microaneurysms": {"count": 14, "mask": None},
        "hemorrhages": {"count": 3, "mask": None},
        "hard_exudates": {"count": 8, "mask": None},
        "soft_exudates": {"count": 0, "mask": None},
    }
    evidence = extract_clinical_evidence(lesions_sample)

    assert isinstance(evidence, dict)
    assert "lesion_counts" in evidence
    assert "findings" in evidence
    assert "severity_rationale" in evidence

    assert evidence["lesion_counts"]["microaneurysms"] == 14
    assert evidence["lesion_counts"]["hemorrhages"] == 3
    assert evidence["lesion_counts"]["hard_exudates"] == 8
    assert len(evidence["findings"]) == 3


def test_calibrate_confidence():
    """Verify confidence calibration handles raw softmax vs temperature scaling."""
    raw_res = calibrate_confidence([0.05, 0.10, 0.75, 0.05, 0.05], calibrated=False)
    assert raw_res["confidence"] == 0.75
    assert raw_res["is_calibrated"] is False
    assert "Uncalibrated" in raw_res["method"]

    temp_res = calibrate_confidence([0.05, 0.10, 0.75, 0.05, 0.05], temperature=1.5, calibrated=True)
    assert temp_res["is_calibrated"] is True
    assert "Temperature Scaling" in temp_res["method"]


def test_explainability_invalid_inputs():
    """Verify explainability functions raise ValueError on invalid array inputs."""
    with pytest.raises(ValueError):
        generate_gradcam(None, np.array([]))
