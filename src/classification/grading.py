"""
DR Severity Grading Definitions & Clinical Scale Mapping.

Implements standard mapping for International Clinical Diabetic Retinopathy Severity Scale:
- Grade 0: No DR (Non-referable)
- Grade 1: Mild NPDR (Non-referable)
- Grade 2: Moderate NPDR (Referable)
- Grade 3: Severe NPDR (Referable)
- Grade 4: Proliferative DR (Referable)

Referable Diabetic Retinopathy is defined as Grade 2 or higher (Moderate NPDR+).
"""

from typing import Dict, List, Any
from config.settings import DR_GRADES, REFERABLE_GRADES


def grade_to_label(grade: int) -> str:
    """
    Convert numerical DR severity grade (0-4) to clinical label string.

    Args:
        grade: Integer class ID (0, 1, 2, 3, 4).

    Returns:
        Clinical severity label string.

    Raises:
        ValueError: If grade is outside [0, 4].
    """
    if not isinstance(grade, int) or grade not in DR_GRADES:
        raise ValueError(f"Invalid DR grade: {grade}. Must be an integer between 0 and 4.")
    return DR_GRADES[grade]


def is_referable(grade: int) -> bool:
    """
    Determine whether a DR severity grade requires clinical referral (Level 2+).

    Args:
        grade: Integer class ID (0, 1, 2, 3, 4).

    Returns:
        True if grade is 2, 3, or 4; False if grade is 0 or 1.

    Raises:
        ValueError: If grade is outside [0, 4].
    """
    if not isinstance(grade, int) or grade not in DR_GRADES:
        raise ValueError(f"Invalid DR grade: {grade}. Must be an integer between 0 and 4.")
    return grade in REFERABLE_GRADES


def format_grading_result(class_id: int, confidence: float, probabilities: List[float]) -> Dict[str, Any]:
    """
    Format DR classification model output into structured grading result dict matching API schema:
    {
        "class_id": 0-4,
        "label": str,
        "referable": bool,
        "confidence": float (0.0-1.0),
        "probabilities": list of length 5
    }

    Args:
        class_id: Predicted class ID (0-4).
        confidence: Top-class prediction confidence / probability (0.0 - 1.0).
        probabilities: Full class probability distribution list of length 5.

    Returns:
        Structured dictionary matching RETINASCAN grading schema.
    """
    if not isinstance(class_id, int) or class_id not in DR_GRADES:
        raise ValueError(f"Invalid class_id: {class_id}. Must be integer 0-4.")

    label = grade_to_label(class_id)
    referable_status = is_referable(class_id)

    return {
        "class_id": class_id,
        "label": label,
        "referable": referable_status,
        "confidence": round(float(confidence), 4),
        "probabilities": [round(float(p), 4) for p in probabilities],
    }
