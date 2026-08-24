"""
Unit Tests for RETINASCAN UI Helpers, Session State, and Pipeline Contract Compliance
"""

import pytest
import io
import numpy as np
from PIL import Image

from app.utils.image_processing import create_demo_fundus_sample

from app.utils.formatting import (
    get_dr_grade_info,
    is_referable,
    format_referral_status,
    format_confidence,
    format_gradability,
)
from app.utils.mock_pipeline import analyze_fundus as mock_analyze_fundus
from app.utils.pipeline_adapter import analyze_fundus as adapter_analyze_fundus, is_using_real_pipeline
from src.simulation.capacity import ScreeningProgramParams, CapacityEstimator
from src.simulation.rural_network import PHCNode, RuralNetworkSimulator


def test_dr_grade_mappings():
    """Verify standard ICDR 5-level severity scale definitions."""
    grade_0 = get_dr_grade_info(0)
    assert grade_0["short_label"] == "No DR"
    assert is_referable(0) is False

    grade_1 = get_dr_grade_info(1)
    assert "Mild" in grade_1["label"]
    assert is_referable(1) is False

    grade_2 = get_dr_grade_info(2)
    assert "Moderate" in grade_2["label"]
    assert is_referable(2) is True

    grade_3 = get_dr_grade_info(3)
    assert "Severe" in grade_3["label"]
    assert is_referable(3) is True

    grade_4 = get_dr_grade_info(4)
    assert "Proliferative" in grade_4["label"]
    assert is_referable(4) is True


def test_referral_status_formatting():
    """Verify referable status text and color indicators."""
    text_ref, col_ref, bg_ref = format_referral_status(True)
    assert "Referral Required" in text_ref or "REFERABLE" in text_ref
    assert col_ref == "#dc2626"

    text_non, col_non, bg_non = format_referral_status(False)
    assert "NON-REFERABLE" in text_non
    assert col_non == "#059669"


def test_confidence_formatting():
    """Verify percentage formatting for normalized confidence scores."""
    assert format_confidence(0.942) == "94.2%"
    assert format_confidence(1.0) == "100.0%"
    assert format_confidence(0.0) == "0.0%"
    assert format_confidence(None) == "N/A"


def test_gradability_formatting():
    """Verify image quality status formatting."""
    label_grad, col_grad, _ = format_gradability(True, 0.88)
    assert "GRADABLE (88%)" in label_grad
    assert col_grad == "#059669"

    label_ungrad, col_ungrad, _ = format_gradability(False, 0.35)
    assert "UNGRADABLE (35%)" in label_ungrad
    assert col_ungrad == "#dc2626"


def test_mock_pipeline_contract_compliance():
    """Verify that mock pipeline output strictly adheres to the AI contract schema."""
    valid_bytes, _ = create_demo_fundus_sample("normal")
    valid_img = Image.open(io.BytesIO(valid_bytes))
    result = mock_analyze_fundus(valid_img)

    # 1. Top-level contract keys
    expected_keys = [
        "is_mock",
        "quality",
        "preprocessing",
        "structures",
        "lesions",
        "grading",
        "explainability",
        "recommendation",
    ]
    for k in expected_keys:
        assert k in result, f"Missing required top-level key '{k}' in pipeline result."

    # 2. Mock flag must be explicitly True
    assert result["is_mock"] is True

    # 3. Quality schema
    quality = result["quality"]
    assert "score" in quality and isinstance(quality["score"], float)
    assert "gradable" in quality and isinstance(quality["gradable"], bool)
    assert "blur_score" in quality
    assert "illumination_score" in quality
    assert "field_of_view_score" in quality
    assert "feedback" in quality and isinstance(quality["feedback"], list)

    # 4. Grading schema
    grading = result["grading"]
    assert "class_id" in grading and grading["class_id"] in [-1, 0, 1, 2, 3, 4]
    assert "label" in grading and isinstance(grading["label"], str)
    assert "confidence" in grading and isinstance(grading["confidence"], float)
    assert "probabilities" in grading and len(grading["probabilities"]) == 5

    # 5. Lesions schema
    lesions = result["lesions"]
    for l_key in ["microaneurysms", "hemorrhages", "hard_exudates", "soft_exudates"]:
        assert l_key in lesions and isinstance(lesions[l_key], int)

    # 6. Explainability schema
    explainability = result["explainability"]
    assert "gradcam" in explainability
    assert "lesion_evidence" in explainability


def test_pipeline_adapter_fallback():
    """Verify adapter functions properly and defaults to mock fallback when real model is absent."""
    dummy_img = Image.new("RGB", (256, 256), color="black")
    result = adapter_analyze_fundus(dummy_img)
    assert result is not None
    assert "grading" in result
    assert "is_mock" in result


def test_capacity_simulation_calculations():
    """Verify rural telemedicine screening capacity estimator."""
    params = ScreeningProgramParams(
        num_phcs=20,
        patients_per_phc_per_day=20,
        working_days_per_year=250,
        num_ophthalmologists=2,
        referral_rate=0.20
    )
    report = CapacityEstimator.estimate_annual_capacity(params)
    assert report.daily_district_volume > 300
    assert report.annual_projected_capacity > 75000
    assert report.annual_target_patients == 100000
    assert report.doctor_utilization_pct > 0.0


def test_rural_network_transmission_time():
    """Verify transmission time calculation under constrained bandwidth."""
    sim = RuralNetworkSimulator()
    # 3.5 MB image over 2 Mbps uplink with 5% packet loss
    t_seconds = sim.calculate_transmission_time(image_size_mb=3.5, bandwidth_mbps=2.0, packet_loss_rate=0.05)
    # (3.5 * 8) / (2.0 * 0.95) = 14.736 seconds +/- jitter
    assert 13.0 < t_seconds < 18.0
