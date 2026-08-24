"""
Unit tests for integrated RETINASCAN analyze_fundus pipeline contract.
"""

import numpy as np
import pytest
from src.pipeline import analyze_fundus


def test_analyze_fundus_schema_contract():
    """Verify analyze_fundus returns all top-level keys required by Streamlit UI contract."""
    img = np.random.randint(20, 200, (256, 256, 3), dtype=np.uint8)
    res = analyze_fundus(img)

    assert isinstance(res, dict)
    assert "quality" in res
    assert "preprocessing" in res
    assert "structures" in res
    assert "lesions" in res
    assert "grading" in res
    assert "explainability" in res
    assert "recommendation" in res

    # Verify nested structures schema
    assert "vessels" in res["structures"]
    assert "optic_disc" in res["structures"]
    assert "fovea" in res["structures"]

    # Verify nested lesions schema
    assert "microaneurysms" in res["lesions"]
    assert "hemorrhages" in res["lesions"]
    assert "hard_exudates" in res["lesions"]
    assert "soft_exudates" in res["lesions"]

    # Verify nested grading schema
    assert "class_id" in res["grading"]
    assert "label" in res["grading"]
    assert "referable" in res["grading"]
    assert "confidence" in res["grading"]
    assert "probabilities" in res["grading"]

    # Verify nested explainability schema
    assert "gradcam" in res["explainability"]
    assert "lesion_evidence" in res["explainability"]


def test_analyze_fundus_invalid_input():
    """Verify analyze_fundus raises ValueError on invalid image inputs."""
    with pytest.raises(ValueError):
        analyze_fundus("non_existent_file.jpg")
