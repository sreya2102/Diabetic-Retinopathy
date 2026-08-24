"""
Inference Interface for DR Severity Classification.
"""

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F

from src.classification.grading import format_grading_result
from src.classification.model import DRClassifier, build_dr_classifier
from src.preprocessing.normalization import normalize_fundus


def predict_dr_grade(model: Optional[DRClassifier], image: np.ndarray) -> Dict[str, Any]:
    """
    Perform DR classification inference on fundus image array.

    Args:
        model: Loaded PyTorch DRClassifier instance or None.
        image: Preprocessed fundus image numpy array (RGB).

    Returns:
        Structured grading dictionary (class_id, label, referable, confidence, probabilities).
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid image array provided for classification inference.")

    # Phase 1 Fallback / Architecture execution when model is instantiated or None
    if model is None:
        # Phase 1: Default placeholder for baseline pipeline contract
        return format_grading_result(
            class_id=0,
            confidence=0.0,
            probabilities=[0.0, 0.0, 0.0, 0.0, 0.0],
        )

    model.eval()

    # Preprocess image to tensor
    norm_img = normalize_fundus(image, target_size=(512, 512), standard_scaling=True)
    tensor_img = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        logits = model(tensor_img)
        probs = F.softmax(logits, dim=1).squeeze(0).tolist()
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

    return format_grading_result(
        class_id=pred_class,
        confidence=confidence,
        probabilities=probs,
    )
