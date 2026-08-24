"""
Unit Tests for Phase 2: Screening Flow, Image Validation & Ungradable Protocol
"""

import pytest
import io
from PIL import Image

from app.utils.image_processing import (
    validate_image_file,
    estimate_image_quality,
    create_demo_fundus_sample
)
from app.utils.mock_pipeline import analyze_fundus as mock_analyze_fundus


def test_validate_valid_image():
    """Verify that clean JPG/PNG fundus imagery passes file validation."""
    valid_bytes, filename = create_demo_fundus_sample("normal")
    is_valid, err_msg, img = validate_image_file(valid_bytes, filename)
    assert is_valid is True
    assert err_msg is None
    assert img is not None
    assert img.width >= 200 and img.height >= 200


def test_validate_corrupt_image():
    """Verify that corrupted byte streams are caught safely with a clean error message."""
    corrupt_bytes = b"NOT_A_VALID_IMAGE_FILE_RANDOM_BYTES"
    is_valid, err_msg, img = validate_image_file(corrupt_bytes, "fundus.jpg")
    assert is_valid is False
    assert "Corrupt" in err_msg or "unreadable" in err_msg
    assert img is None


def test_validate_empty_image():
    """Verify that 0-byte uploads are flagged."""
    is_valid, err_msg, img = validate_image_file(b"", "empty.png")
    assert is_valid is False
    assert "empty" in err_msg.lower()


def test_validate_unsupported_format():
    """Verify that unsupported file extensions are rejected."""
    buf = io.BytesIO()
    dummy_img = Image.new("RGB", (300, 300), color="white")
    dummy_img.save(buf, format="GIF")
    is_valid, err_msg, _ = validate_image_file(buf.getvalue(), "scan.gif")
    assert is_valid is False
    assert "Unsupported file type" in err_msg


def test_quality_estimator_on_normal():
    """Verify quality score on a clear fundus sample."""
    valid_bytes, _ = create_demo_fundus_sample("normal")
    img = Image.open(io.BytesIO(valid_bytes))
    metrics = estimate_image_quality(img)
    assert metrics["gradable"] is True
    assert metrics["score"] >= 0.55
    assert len(metrics["feedback"]) > 0


def test_quality_estimator_on_blurry_ungradable():
    """Verify that severe blur sample is recognized as ungradable."""
    blur_bytes, _ = create_demo_fundus_sample("ungradable_blur")
    img = Image.open(io.BytesIO(blur_bytes))
    metrics = estimate_image_quality(img)
    assert metrics["gradable"] is False
    assert metrics["blur_score"] < 0.40


def test_ungradable_protocol_gating():
    """Verify that ungradable fundus photography disables automated DR diagnosis and mandates recapture."""
    blur_bytes, _ = create_demo_fundus_sample("ungradable_blur")
    img = Image.open(io.BytesIO(blur_bytes))
    result = mock_analyze_fundus(img)
    
    assert result["quality"]["gradable"] is False
    # Diagnosis must be gated
    assert result["grading"]["class_id"] == -1
    assert "Ungradable" in result["grading"]["label"]
    assert result["grading"]["referable"] is None
    assert "recapture with improved focus" in result["recommendation"]
