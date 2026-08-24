"""
RETINASCAN - Application Configuration
Explainable AI for Rural Diabetic Retinopathy Screening
"""

from typing import List

# Application Metadata
APP_NAME: str = "RETINASCAN"
APP_SUBTITLE: str = "Explainable AI for Rural Diabetic Retinopathy Screening"
APP_VERSION: str = "0.1.0"
APP_ORGANIZATION: str = "Rural Health Tele-Ophthalmology Initiative"

# Screening & Media Rules
SUPPORTED_IMAGE_TYPES: List[str] = ["jpg", "jpeg", "png", "tif", "tiff"]
MAX_IMAGE_SIZE_MB: int = 25

# Demonstration / Safety Flags
DEMO_MODE: bool = True  # True when real AI weights/pipeline are not yet attached

# Referability Threshold
# 0 = No DR, 1 = Mild NPDR, 2 = Moderate NPDR, 3 = Severe NPDR, 4 = Proliferative DR
REFERABLE_GRADE_THRESHOLD: int = 2

# Clinical Disclaimer Text
CLINICAL_DISCLAIMER: str = (
    "AI output is intended to support, not replace, clinical diagnosis. "
    "All screening recommendations require examination or validation by a qualified ophthalmologist."
)

MOCK_BANNER_TEXT: str = "DEMO / MOCK RESULT — AI pipeline not connected. Demonstration view only."
