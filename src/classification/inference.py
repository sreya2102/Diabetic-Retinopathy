"""
Inference Engine for DR Severity Classification.

Runs PyTorch evaluation pass on preprocessed fundus images and computes class probabilities,
predicted class ID, confidence, and referable DR status.
"""

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F

from src.classification.grading import format_grading_result
from src.classification.model import DRClassifier, build_dr_classifier
from src.preprocessing.normalization import normalize_fundus


def predict_dr_grade(
    model: Optional[DRClassifier], image: np.ndarray, device: str = "cpu"
) -> Dict[str, Any]:
    """
    Perform DR classification inference on fundus image array using PyTorch model.

    Args:
        model: PyTorch DRClassifier instance. If None, builds default architecture.
        image: Preprocessed fundus image numpy array (RGB).
        device: 'cpu' or 'cuda'.

    Returns:
        Structured grading result matching RETINASCAN contract:
        {
            "class_id": 0-4,
            "label": "...",
            "referable": True/False,
            "confidence": 0.0-1.0,
            "probabilities": [p0, p1, p2, p3, p4]
        }
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid image array provided for classification inference.")

    # Instantiate default model if none provided
    if model is None:
        model = build_dr_classifier(weights_path=None, device=device)

    model.to(device)
    model.eval()

    # Preprocess image array to (1, 3, 512, 512) PyTorch tensor
    norm_img = normalize_fundus(image, target_size=(512, 512), standard_scaling=True)
    # Re-order HWC -> CHW and add batch dimension (1, C, H, W)
    tensor_img = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(tensor_img)
        probs_tensor = F.softmax(logits, dim=1).squeeze(0)
        probs = [float(p) for p in probs_tensor.cpu().numpy()]

    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])

    return format_grading_result(
        class_id=pred_class,
        confidence=confidence,
        probabilities=probs,
    )
