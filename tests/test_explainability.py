"""
Integrated explainability tests for RETINASCAN.

Covers:
- AI Grad-CAM generation
- Clinical lesion evidence
- Confidence calibration
- Invalid input handling
- UI Grad-CAM visualization
- Mock pipeline explainability contract
"""

import io

import numpy as np
import pytest
import torch
from PIL import Image

from src.classification.model import DRClassifier
from src.explainability.confidence import calibrate_confidence
from src.explainability.evidence import extract_clinical_evidence
from src.explainability.gradcam import generate_gradcam

from app.utils.image_processing import create_demo_fundus_sample
from app.components.gradcam_viewer import blend_heatmap_on_image
from app.utils.mock_pipeline import analyze_fundus as mock_analyze_fundus


# ============================================================
# AI EXPLAINABILITY TESTS
# ============================================================


def test_generate_gradcam_pytorch_execution():
    """Verify Grad-CAM executes on the DR classifier and returns valid outputs."""
    model = DRClassifier(num_classes=5)

    img = np.random.randint(
        0,
        255,
        (128, 128, 3),
        dtype=np.uint8,
    )

    result = generate_gradcam(model, img)

    assert isinstance(result, dict)
    assert "original" in result
    assert "heatmap" in result
    assert "overlay" in result
    assert "status" in result

    assert result["heatmap"].shape == (128, 128)
    assert result["overlay"].shape == (128, 128, 3)

    assert result["heatmap"].min() >= 0.0
    assert result["heatmap"].max() <= 1.0


def test_extract_clinical_evidence():
    """Verify clinical evidence is extracted from lesion detection results."""
    lesions_sample = {
        "microaneurysms": {
            "count": 14,
            "mask": None,
        },
        "hemorrhages": {
            "count": 3,
            "mask": None,
        },
        "hard_exudates": {
            "count": 8,
            "mask": None,
        },
        "soft_exudates": {
            "count": 0,
            "mask": None,
        },
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
    """Verify raw and temperature-scaled confidence calibration."""
    raw_result = calibrate_confidence(
        [0.05, 0.10, 0.75, 0.05, 0.05],
        calibrated=False,
    )

    assert raw_result["confidence"] == 0.75
    assert raw_result["is_calibrated"] is False
    assert "Uncalibrated" in raw_result["method"]

    temperature_result = calibrate_confidence(
        [0.05, 0.10, 0.75, 0.05, 0.05],
        temperature=1.5,
        calibrated=True,
    )

    assert temperature_result["is_calibrated"] is True
    assert "Temperature Scaling" in temperature_result["method"]


def test_explainability_invalid_inputs():
    """Verify invalid Grad-CAM inputs raise ValueError."""
    with pytest.raises(ValueError):
        generate_gradcam(
            None,
            np.array([]),
        )


# ============================================================
# UI EXPLAINABILITY TESTS
# ============================================================


def test_blend_heatmap_on_image():
    """Verify heatmap overlay blending produces a valid RGB image."""
    raw_bytes, _ = create_demo_fundus_sample("normal")

    original_image = Image.open(
        io.BytesIO(raw_bytes)
    )

    width, height = original_image.size

    dummy_heatmap = np.zeros(
        (30, 30),
        dtype=np.float32,
    )

    dummy_heatmap[10:20, 10:20] = 1.0

    blended = blend_heatmap_on_image(
        original_image,
        dummy_heatmap,
        alpha=0.5,
    )

    assert blended.size == (width, height)
    assert blended.mode == "RGB"


def test_explainability_evidence_contract():
    """Verify mock pipeline explainability output follows the UI contract."""
    raw_bytes, _ = create_demo_fundus_sample(
        "moderate_npdr"
    )

    original_image = Image.open(
        io.BytesIO(raw_bytes)
    )

    result = mock_analyze_fundus(
        original_image
    )

    assert "explainability" in result

    explainability = result["explainability"]

    assert "gradcam" in explainability
    assert "lesion_evidence" in explainability

    assert isinstance(
        explainability["lesion_evidence"],
        list,
    )

    if explainability["lesion_evidence"]:
        first_item = explainability["lesion_evidence"][0]

        assert "type" in first_item
        assert "region" in first_item
        assert "count" in first_item