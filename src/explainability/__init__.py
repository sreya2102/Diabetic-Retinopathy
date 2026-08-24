"""Explainable AI (XAI) package for Grad-CAM, lesion evidence extraction, and confidence calibration."""

from src.explainability.confidence import calibrate_confidence
from src.explainability.evidence import extract_clinical_evidence
from src.explainability.gradcam import PyTorchGradCAM, generate_gradcam

__all__ = [
    "PyTorchGradCAM",
    "generate_gradcam",
    "extract_clinical_evidence",
    "calibrate_confidence",
]
