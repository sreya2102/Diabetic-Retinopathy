"""
RETINASCAN - Application Configuration & Clinical Definitions

Explainable AI for Rural Diabetic Retinopathy Screening.
"""

from typing import Dict, List, Set, Tuple


# ============================================================
# Application Metadata
# ============================================================

APP_NAME: str = "RETINASCAN"

APP_SUBTITLE: str = (
    "Explainable AI for Rural Diabetic Retinopathy Screening"
)

APP_VERSION: str = "0.1.0"

APP_ORGANIZATION: str = (
    "Rural Health Tele-Ophthalmology Initiative"
)


# ============================================================
# International Clinical Diabetic Retinopathy Severity Scale
# ============================================================

DR_GRADES: Dict[int, str] = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}


# Referable DR:
# Grade 2 (Moderate NPDR) and above
REFERABLE_GRADES: Set[int] = {2, 3, 4}

REFERABLE_GRADE_THRESHOLD: int = 2


# ============================================================
# Image / Screening Configuration
# ============================================================

SUPPORTED_IMAGE_TYPES: List[str] = [
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
]

MAX_IMAGE_SIZE_MB: int = 25

DEFAULT_INPUT_RESOLUTION: Tuple[int, int] = (512, 512)


# ============================================================
# Demonstration / AI Pipeline Safety
# ============================================================

# True when the real trained AI model/pipeline is not attached.
# The UI must clearly identify mock/demo results.
DEMO_MODE: bool = True

# Prevent processing of images that have been determined
# to be ungradable by the quality assessment module.
ALLOW_UNGRADABLE_PROCESSING: bool = False


# ============================================================
# Clinical Disclaimer
# ============================================================

CLINICAL_DISCLAIMER: str = (
    "AI output is intended to support, not replace, clinical diagnosis. "
    "All screening recommendations require examination or validation "
    "by a qualified ophthalmologist."
)


# ============================================================
# Mock / Demonstration UI
# ============================================================

MOCK_BANNER_TEXT: str = (
    "DEMO / MOCK RESULT — AI pipeline not connected. "
    "Demonstration view only."
)