"""
Formatting Utilities for Clinical Presentation
Provides clinical labels, severity mappings, color tags, and string helpers.
"""

from datetime import datetime
from typing import Dict, Tuple

# Standard International Clinical Diabetic Retinopathy Disease Severity Scale
DR_SEVERITY_MAP: Dict[int, Dict[str, str]] = {
    0: {
        "label": "No Diabetic Retinopathy",
        "short_label": "No DR",
        "color": "#10b981",  # Emerald green
        "bg_color": "#ecfdf5",
        "description": "No apparent retinal vascular abnormalities detected.",
        "referral_recommendation": "Routine annual screening recommended."
    },
    1: {
        "label": "Mild Non-Proliferative DR",
        "short_label": "Mild NPDR",
        "color": "#3b82f6",  # Blue
        "bg_color": "#eff6ff",
        "description": "Microaneurysms only. Early disease manifestation.",
        "referral_recommendation": "Follow-up rescreening in 6 to 12 months."
    },
    2: {
        "label": "Moderate Non-Proliferative DR",
        "short_label": "Moderate NPDR",
        "color": "#f59e0b",  # Amber/Orange
        "bg_color": "#fffbeb",
        "description": "Microaneurysms, blot hemorrhages, and/or hard exudates present.",
        "referral_recommendation": "Referral to ophthalmology within 4 to 8 weeks."
    },
    3: {
        "label": "Severe Non-Proliferative DR",
        "short_label": "Severe NPDR",
        "color": "#ea580c",  # Deep orange/Rust
        "bg_color": "#fff7ed",
        "description": "Extensive intraretinal hemorrhages, venous beading, or IRMA present.",
        "referral_recommendation": "Prompt ophthalmologist evaluation within 2 to 4 weeks."
    },
    4: {
        "label": "Proliferative Diabetic Retinopathy",
        "short_label": "Proliferative DR",
        "color": "#dc2626",  # Crimson red
        "bg_color": "#fef2f2",
        "description": "Neovascularization or vitreous/preretinal hemorrhage observed.",
        "referral_recommendation": "URGENT referral to retina specialist within 1 to 2 weeks."
    },
    -1: {
        "label": "Ungradable Image (Recapture Required)",
        "short_label": "Ungradable",
        "color": "#e11d48",  # Rose
        "bg_color": "#fff1f2",
        "description": "Severe defocus, improper illumination, or obscured posterior pole.",
        "referral_recommendation": "Recapture fundus photography with improved focus and illumination."
    }
}


def get_dr_grade_info(class_id: int) -> Dict[str, str]:
    """Retrieve severity label, color styling, and clinical advice for a given DR grade ID (0-4)."""
    return DR_SEVERITY_MAP.get(
        class_id,
        {
            "label": f"Grade {class_id} (Unknown)",
            "short_label": f"G{class_id}",
            "color": "#6b7280",
            "bg_color": "#f3f4f6",
            "description": "Unclassified retinopathy pattern.",
            "referral_recommendation": "Consult ophthalmologist for evaluation."
        }
    )


def is_referable(class_id: int) -> bool:
    """Determine if a DR severity grade is referable (Grade >= 2 under standard screening protocols)."""
    return class_id >= 2


def format_referral_status(referable: bool) -> Tuple[str, str, str]:
    """
    Format referability status.
    Returns: (Status text, Text color, Background color)
    """
    if referable:
        return "REFERABLE DR (Referral Required)", "#dc2626", "#fef2f2"
    return "NON-REFERABLE (Routine Rescreening)", "#059669", "#ecfdf5"


def format_confidence(confidence: float) -> str:
    """Format confidence value as percentage string."""
    if confidence is None:
        return "N/A"
    return f"{confidence * 100:.1f}%" if 0.0 <= confidence <= 1.0 else f"{confidence:.1f}%"


def format_gradability(gradable: bool, score: float = 0.0) -> Tuple[str, str, str]:
    """
    Format image gradability.
    Returns: (Gradability Label, Text Color, Background Color)
    """
    if gradable:
        return f"GRADABLE ({score * 100:.0f}%)", "#059669", "#ecfdf5"
    return f"UNGRADABLE ({score * 100:.0f}%)", "#dc2626", "#fef2f2"


def get_current_timestamp_formatted() -> str:
    """Return ISO standard formatted readable clinical timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
