"""
Single Integrated Pipeline Contract for RETINASCAN AI Analysis.

This module serves as the primary integration contract consumed by the Streamlit UI.
"""

from typing import Dict, Any, Union, Optional
import numpy as np

from src.classification.grading import is_referable
from src.classification.inference import predict_dr_grade
from src.explainability.confidence import calibrate_confidence
from src.explainability.evidence import extract_clinical_evidence
from src.explainability.gradcam import generate_gradcam
from src.preprocessing.enhancement import preprocess_fundus
from src.preprocessing.quality import assess_image
from src.segmentation.fovea import locate_fovea
from src.segmentation.lesions import detect_lesions
from src.segmentation.optic_disc import locate_optic_disc
from src.segmentation.vessels import segment_vessels
from src.utils import load_image


def analyze_fundus(
    image_input: Union[np.ndarray, str], model: Any = None, device: str = "cpu"
) -> Dict[str, Any]:
    """
    Run complete end-to-end RETINASCAN Diabetic Retinopathy analysis pipeline on a fundus image.

    Pipeline sequence:
    Fundus Image -> Quality Assessment -> Preprocessing/Enhancement ->
    Structure Analysis -> Lesion Detection -> DR Classification ->
    Explainable AI (Grad-CAM/Evidence) -> Screening Recommendation -> Structured UI Output.

    Args:
        image_input: Numpy ndarray (RGB) or valid image file path.
        model: Optional PyTorch DRClassifier model instance.
        device: 'cpu' or 'cuda'.

    Returns:
        Stable, UI-friendly structured analysis dictionary matching RETINASCAN contract schema:
        {
            "quality": {...},
            "preprocessing": {...},
            "structures": {"vessels": ..., "optic_disc": ..., "fovea": ...},
            "lesions": {"microaneurysms": ..., "hemorrhages": ..., "hard_exudates": ..., "soft_exudates": ...},
            "grading": {"class_id": ..., "label": ..., "referable": ..., "confidence": ..., "probabilities": ...},
            "explainability": {"gradcam": ..., "lesion_evidence": ...},
            "recommendation": ...
        }
    """
    # Load and validate input image array (RGB format)
    image = load_image(image_input, color_mode="RGB")

    # Step 1: Image Quality Assessment (Blur, Illumination, Exposure, FOV, Gradability, Feedback)
    quality_res = assess_image(image)

    # Step 2: Image Enhancement & Preprocessing (CLAHE, Green Channel, Denoising)
    enhancement_res = preprocess_fundus(image)

    # Step 3: Retinal Structure Analysis (Vessels, Optic Disc, Fovea)
    vessel_mask = segment_vessels(image)
    optic_disc_res = locate_optic_disc(image)
    fovea_res = locate_fovea(image, optic_disc_center=optic_disc_res.get("center"))

    structures_res = {
        "vessels": vessel_mask,
        "optic_disc": optic_disc_res,
        "fovea": fovea_res,
    }

    # Step 4: Lesion Detection & Segmentation (MAs, HEs, EXs, SEs)
    lesions_res = detect_lesions(image)

    # Step 5: DR Severity Classification (PyTorch 5-class model inference)
    grading_res = predict_dr_grade(model, image, device=device)

    # Step 6: Explainable AI (Grad-CAM visual heatmap/overlay & Lesion Clinical Evidence)
    gradcam_res = generate_gradcam(model, image)
    evidence_res = extract_clinical_evidence(lesions_res, grading_dict=grading_res)

    # Step 7: Screening Recommendation Synthesis
    if not quality_res.get("gradable", True):
        recommendation = (
            "UNGRADABLE FUNDUS IMAGE: Re-capture required. "
            + "; ".join(quality_res.get("feedback", []))
        )
    elif grading_res.get("referable", False):
        recommendation = (
            f"REFERABLE DIABETIC RETINOPATHY ({grading_res.get('label')}): "
            "Urgent clinical evaluation by an ophthalmologist is recommended."
        )
    else:
        recommendation = (
            f"NON-REFERABLE DIABETIC RETINOPATHY ({grading_res.get('label')}): "
            "Routine annual diabetic eye screening recommended."
        )

    # Final Integrated Contract Schema Output
    return {
        "quality": quality_res,
        "preprocessing": {
            "enhanced": enhancement_res.get("enhanced"),
            "clahe_green": enhancement_res.get("clahe_green"),
        },
        "structures": structures_res,
        "lesions": {
            "microaneurysms": lesions_res.get("microaneurysms"),
            "hemorrhages": lesions_res.get("hemorrhages"),
            "hard_exudates": lesions_res.get("hard_exudates"),
            "soft_exudates": lesions_res.get("soft_exudates"),
        },
        "grading": grading_res,
        "explainability": {
            "gradcam": gradcam_res,
            "lesion_evidence": evidence_res,
        },
        "recommendation": recommendation,
    }
