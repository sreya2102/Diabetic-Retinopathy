"""
Unit tests for integrated RETINASCAN analyze_fundus pipeline contract.
"""

import numpy as np
import pytest
from src.classification.model import build_dr_classifier
from src.pipeline import analyze_fundus


def test_analyze_fundus_schema_contract():
    """Verify analyze_fundus returns all required top-level and nested UI contract keys."""
    img = np.random.randint(20, 200, (256, 256, 3), dtype=np.uint8)
    model = build_dr_classifier(weights_path=None)
    res = analyze_fundus(img, model=model)

    assert isinstance(res, dict)
    assert "quality" in res
    assert "preprocessing" in res
    assert "structures" in res
    assert "lesions" in res
    assert "grading" in res
    assert "explainability" in res
    assert "recommendation" in res

    # Verify quality schema
    assert "score" in res["quality"]
    assert "gradable" in res["quality"]
    assert "blur_score" in res["quality"]
    assert "illumination_score" in res["quality"]
    assert "field_of_view_score" in res["quality"]

    # Verify structures schema
    assert "vessels" in res["structures"]
    assert "optic_disc" in res["structures"]
    assert "fovea" in res["structures"]

    # Verify lesions schema
    assert "microaneurysms" in res["lesions"]
    assert "hemorrhages" in res["lesions"]
    assert "hard_exudates" in res["lesions"]
    assert "soft_exudates" in res["lesions"]

    # Verify grading schema
    assert "class_id" in res["grading"]
    assert 0 <= res["grading"]["class_id"] <= 4
    assert "label" in res["grading"]
    assert "referable" in res["grading"]
    assert "confidence" in res["grading"]
    assert "probabilities" in res["grading"]
    assert len(res["grading"]["probabilities"]) == 5

    # Verify explainability schema
    assert "gradcam" in res["explainability"]
    assert "lesion_evidence" in res["explainability"]

    assert isinstance(res["recommendation"], str)


def test_analyze_fundus_ungradable_handling():
    """Verify analyze_fundus handles ungradable (excessively dark) fundus image correctly."""
    dark_img = np.ones((256, 256, 3), dtype=np.uint8) * 10
    res = analyze_fundus(dark_img)

    assert res["quality"]["gradable"] is False
    assert "UNGRADABLE" in res["recommendation"]


def test_analyze_fundus_invalid_input():
    """Verify analyze_fundus raises ValueError on invalid input paths or empty arrays."""
    with pytest.raises(ValueError):
        analyze_fundus("non_existent_file.jpg")

    with pytest.raises(ValueError):
        analyze_fundus(np.array([]))
