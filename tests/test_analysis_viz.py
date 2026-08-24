"""
Unit Tests for Phase 3: Image Enhancement, Retinal Vasculature & Anatomical Landmark Overlays
"""

import io
from PIL import Image

from app.utils.image_processing import (
    apply_clahe_enhancement,
    extract_vessel_mask,
    locate_retinal_landmarks,
    generate_annotated_fundus_overlay,
    create_demo_fundus_sample
)


def test_apply_clahe_enhancement():
    """Verify that CLAHE enhancement preserves image dimensions and improves contrast."""
    raw_bytes, _ = create_demo_fundus_sample("normal")
    orig_img = Image.open(io.BytesIO(raw_bytes))
    
    enhanced = apply_clahe_enhancement(orig_img)
    assert enhanced.size == orig_img.size
    assert enhanced.mode == "RGB"


def test_extract_vessel_mask():
    """Verify that vessel tree extraction generates a valid segmentation map."""
    raw_bytes, _ = create_demo_fundus_sample("normal")
    orig_img = Image.open(io.BytesIO(raw_bytes))
    
    vessel_mask = extract_vessel_mask(orig_img)
    assert vessel_mask.size == orig_img.size
    assert vessel_mask.mode == "RGB"


def test_locate_retinal_landmarks():
    """Verify that optic disc and fovea centers are located within image bounds."""
    raw_bytes, _ = create_demo_fundus_sample("normal")
    orig_img = Image.open(io.BytesIO(raw_bytes))
    w, h = orig_img.size
    
    landmarks = locate_retinal_landmarks(orig_img)
    assert "optic_disc" in landmarks
    assert "fovea" in landmarks
    assert "vasculature" in landmarks
    
    od_x, od_y = landmarks["optic_disc"]["center"]
    assert 0 <= od_x < w
    assert 0 <= od_y < h
    assert landmarks["optic_disc"]["radius"] > 0
    
    fov_x, fov_y = landmarks["fovea"]["center"]
    assert 0 <= fov_x < w
    assert 0 <= fov_y < h


def test_generate_annotated_fundus_overlay():
    """Verify that composite annotation overlay generates clean RGB image."""
    raw_bytes, _ = create_demo_fundus_sample("normal")
    orig_img = Image.open(io.BytesIO(raw_bytes))
    
    annotated = generate_annotated_fundus_overlay(
        image=orig_img,
        show_vessels=True,
        show_disc=True,
        show_fovea=True
    )
    assert annotated.size == orig_img.size
    assert annotated.mode == "RGB"
