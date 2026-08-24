"""Explainable AI (XAI) package for Grad-CAM, lesion evidence extraction, and confidence calibration."""

from src.explainability.confidence import calibrate_confidence
from src.explainability.evidence import extract_clinical_evidence
from src.explainability.gradcam import generate_gradcam

__all__ = [
    "generate_gradcam",
    "extract_clinical_evidence",
    "calibrate_confidence",
]
