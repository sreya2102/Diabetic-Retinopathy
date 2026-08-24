"""
Unit tests for PyTorch DR classifier model structure and inference.
"""

import numpy as np
import pytest
import torch
from src.classification.inference import predict_dr_grade
from src.classification.model import DRClassifier, build_dr_classifier


def test_dr_classifier_forward():
    """Verify DRClassifier forward pass returns logits shape (N, 5)."""
    model = DRClassifier(num_classes=5)
    dummy_input = torch.randn(2, 3, 512, 512)
    logits = model(dummy_input)

    assert logits.shape == (2, 5)


def test_predict_dr_grade_execution():
    """Verify predict_dr_grade runs PyTorch model inference and outputs valid softmax probabilities."""
    model = build_dr_classifier(weights_path=None, device="cpu")
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    res = predict_dr_grade(model, img)

    assert isinstance(res, dict)
    assert "class_id" in res
    assert 0 <= res["class_id"] <= 4
    assert "label" in res
    assert "referable" in res
    assert "confidence" in res
    assert "probabilities" in res
    assert len(res["probabilities"]) == 5

    # Check softmax probabilities sum to approximately 1.0
    assert abs(sum(res["probabilities"]) - 1.0) < 1e-3


def test_predict_dr_grade_invalid_input():
    """Verify predict_dr_grade raises ValueError on empty array input."""
    with pytest.raises(ValueError):
        predict_dr_grade(None, np.array([]))
