"""
Mock AI Screening Pipeline Adapter for UI/UX Development
Strictly used as a fallback when the real AI pipeline (src.pipeline) is not yet attached.
Dynamically assesses image gradability and prevents automated diagnosis on ungradable images.
Clearly flagged as is_mock: True.
"""

from typing import Any, Dict
import numpy as np
from PIL import Image

from app.utils.image_processing import estimate_image_quality


def analyze_fundus(image: Any) -> Dict[str, Any]:
    """
    Mock screening pipeline function matching the exact contract required by RETINASCAN.
    
    If an image is evaluated as ungradable (e.g. extreme blur or glare), the grading
    output is gated and the recommendation mandates recapture rather than assigning a diagnosis.
    """
    # Ensure PIL Image
    if not isinstance(image, Image.Image):
        try:
            pil_img = Image.fromarray(np.uint8(image))
        except Exception:
            pil_img = Image.new("RGB", (512, 512), color="black")
    else:
        pil_img = image

    # Run image quality evaluation
    quality_metrics = estimate_image_quality(pil_img)
    is_gradable = quality_metrics.get("gradable", True)

    if not is_gradable:
        # Ungradable protocol: Do NOT diagnose DR on ungradable images
        return {
            "is_mock": True,
            "quality": quality_metrics,
            "preprocessing": {
                "enhanced_image": None
            },
            "structures": {
                "vessels": None,
                "optic_disc": None,
                "fovea": None
            },
            "lesions": {
                "microaneurysms": 0,
                "hemorrhages": 0,
                "hard_exudates": 0,
                "soft_exudates": 0
            },
            "grading": {
                "class_id": -1,
                "label": "Ungradable / Indeterminate",
                "referable": None,
                "confidence": quality_metrics.get("score", 0.0),
                "probabilities": [0.0, 0.0, 0.0, 0.0, 0.0]
            },
            "explainability": {
                "gradcam": None,
                "lesion_evidence": []
            },
            "recommendation": "Image may be ungradable. Suggested action: recapture with improved focus and illumination before diagnostic evaluation."
        }

    # Gradable Mock Output (Realistic Moderate NPDR simulation for UI verification)
    return {
        "is_mock": True,
        "quality": quality_metrics,
        "preprocessing": {
            "enhanced_image": None
        },
        "structures": {
            "vessels": None,
            "optic_disc": {"center": (380, 240), "radius": 35},
            "fovea": {"center": (240, 260), "radius": 20}
        },
        "lesions": {
            "microaneurysms": 6,
            "hemorrhages": 4,
            "hard_exudates": 8,
            "soft_exudates": 0
        },
        "grading": {
            "class_id": 2,
            "label": "Moderate Non-Proliferative DR",
            "referable": True,
            "confidence": 0.89,
            "probabilities": [0.03, 0.05, 0.89, 0.02, 0.01]
        },
        "explainability": {
            "gradcam": None,
            "lesion_evidence": [
                {"type": "Microaneurysms", "region": "Macular perimeter", "count": 6},
                {"type": "Hemorrhages", "region": "Inferior temporal arcade", "count": 4},
                {"type": "Hard Exudates", "region": "Superior temporal quadrant", "count": 8}
            ]
        },
        "recommendation": (
            "Referral to ophthalmology clinic recommended for formal dilated fundus "
            "evaluation within 4 to 8 weeks (DEMO OUTPUT)."
        )
    }


class MockPipelineAdapter:
    """Wrapper class providing static analyze_fundus method."""
    @staticmethod
    def analyze_fundus(image: Any) -> Dict[str, Any]:
        return analyze_fundus(image)
