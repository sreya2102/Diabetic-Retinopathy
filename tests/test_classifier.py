"""
Unit tests for PyTorch DR classifier model structure and inference.
"""

import numpy as np
import torch
from src.classification.inference import predict_dr_grade
from src.classification.model import DRClassifier, build_dr_classifier


def test_dr_classifier_forward():
    """Verify DRClassifier forward pass returns logits shape (N, 5)."""
    model = DRClassifier(num_classes=5)
    dummy_input = torch.randn(2, 3, 512, 512)
    logits = model(dummy_input)

    assert logits.shape == (2, 5)


def test_predict_dr_grade():
    """Verify predict_dr_grade runs inference and formats result dict."""
    model = build_dr_classifier(weights_path=None)
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
