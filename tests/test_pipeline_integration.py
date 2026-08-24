"""
Unit & Contract Integration Tests for Phase 8: AI Pipeline Integration
Verifies contract adherence between UI consumers and src.pipeline.analyze_fundus.
"""

from typing import Dict, Any
import io
import numpy as np
from PIL import Image
import pytest

from app.utils.pipeline_adapter import run_screening_analysis, is_using_real_pipeline, analyze_fundus
from app.utils.mock_pipeline import MockPipelineAdapter
from app.utils.image_processing import create_demo_fundus_sample


def test_mock_pipeline_adapter_contract():
    """Verify mock adapter conforms 100% to the published AI contract schema."""
    raw_bytes, _ = create_demo_fundus_sample("moderate_npdr")
    test_img = Image.open(io.BytesIO(raw_bytes))
    result: Dict[str, Any] = MockPipelineAdapter.analyze_fundus(test_img)
    
    # 1. Root keys
    expected_root_keys = [
        "quality",
        "preprocessing",
        "structures",
        "lesions",
        "grading",
        "explainability",
        "recommendation",
        "is_mock"
    ]
    for key in expected_root_keys:
        assert key in result, f"Missing root key in AI contract: {key}"
        
    # 2. Grading contract
    grading = result["grading"]
    assert "class_id" in grading
    assert "label" in grading
    assert "referable" in grading
    assert "confidence" in grading
    assert "probabilities" in grading
    assert len(grading["probabilities"]) == 5
    assert sum(grading["probabilities"]) == pytest.approx(1.0, rel=1e-2)

    # 3. Quality contract
    quality = result["quality"]
    assert "gradable" in quality
    assert "score" in quality

    # 4. Explainability contract
    exp = result["explainability"]
    assert "gradcam" in exp
    assert "lesion_evidence" in exp
    assert isinstance(exp["lesion_evidence"], list)


def test_ungradable_mock_sample_contract():
    """Verify ungradable sample yields grade -1 and triggers clinical recapture gating."""
    # Defocus blur ungradable sample
    raw_bytes, _ = create_demo_fundus_sample("ungradable_blur")
    test_img = Image.open(io.BytesIO(raw_bytes))
    result = MockPipelineAdapter.analyze_fundus(test_img)
    
    assert result["quality"]["gradable"] is False
    assert result["grading"]["class_id"] == -1
    assert "Ungradable" in result["grading"]["label"]
    assert "recapture" in result["recommendation"].lower()


def test_pipeline_adapter_fallback_mode():
    """Verify adapter handles absence of real AI model safely with fallback flag."""
    raw_bytes, _ = create_demo_fundus_sample("normal_grade_0")
    test_img = Image.open(io.BytesIO(raw_bytes))
    result = run_screening_analysis(test_img)
    
    assert isinstance(result, dict)
    assert "grading" in result
    assert "quality" in result
    assert result["is_mock"] is True
    assert is_using_real_pipeline() is False
