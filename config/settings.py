"""
Project Configuration & Clinical Definitions for RETINASCAN AI Pipeline.
"""

from typing import Dict, List, Set

# International Clinical Diabetic Retinopathy Severity Scale (ICDR Standard)
DR_GRADES: Dict[int, str] = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}

# Referable Diabetic Retinopathy Definition: Grade 2 (Moderate NPDR) and above
REFERABLE_GRADES: Set[int] = {2, 3, 4}

# Supported image file extensions
SUPPORTED_IMAGE_TYPES: List[str] = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

# AI System Version
APP_VERSION: str = "0.1.0-phase1-prototype"

# System Execution Safety Settings
ALLOW_UNGRADABLE_PROCESSING: bool = False
DEFAULT_INPUT_RESOLUTION: tuple = (512, 512)
