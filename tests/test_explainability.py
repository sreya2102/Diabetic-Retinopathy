"""
Unit Tests for Phase 4: Explainable AI, Grad-CAM Blending & Decision Pathway
"""

import io
import numpy as np
from PIL import Image

from app.utils.image_processing import create_demo_fundus_sample
from app.components.gradcam_viewer import blend_heatmap_on_image
from app.utils.mock_pipeline import analyze_fundus as mock_analyze_fundus


def test_blend_heatmap_on_image():
    """Verify that heatmap overlay blending produces a valid composite image."""
    raw_bytes, _ = create_demo_fundus_sample("normal")
    orig_img = Image.open(io.BytesIO(raw_bytes))
    w, h = orig_img.size
    
    # Create 2D float heatmap array (0.0 to 1.0)
    dummy_heatmap = np.zeros((30, 30), dtype=np.float32)
    dummy_heatmap[10:20, 10:20] = 1.0
    
    blended = blend_heatmap_on_image(orig_img, dummy_heatmap, alpha=0.5)
    assert blended.size == (w, h)
    assert blended.mode == "RGB"


def test_explainability_evidence_contract():
    """Verify that explainability contract payload contains valid lesion evidence fields."""
    raw_bytes, _ = create_demo_fundus_sample("moderate_npdr")
    orig_img = Image.open(io.BytesIO(raw_bytes))
    result = mock_analyze_fundus(orig_img)
    
    assert "explainability" in result
    explainability = result["explainability"]
    assert "gradcam" in explainability
    assert "lesion_evidence" in explainability
    assert isinstance(explainability["lesion_evidence"], list)
    
    if explainability["lesion_evidence"]:
        first_item = explainability["lesion_evidence"][0]
        assert "type" in first_item
        assert "region" in first_item
        assert "count" in first_item
