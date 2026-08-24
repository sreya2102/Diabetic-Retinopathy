"""
AI Pipeline Adapter for RETINASCAN
Provides a clean unified interface between the UI and the real AI screening pipeline.
Attempts to import `src.pipeline.analyze_fundus` from the teammate's branch.
If unavailable, gracefully falls back to the isolated mock adapter.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

_REAL_PIPELINE_AVAILABLE: bool = False

try:
    from src.pipeline import analyze_fundus as _real_analyze_fundus
    _REAL_PIPELINE_AVAILABLE = True
    logger.info("Successfully connected to real AI pipeline (src.pipeline).")
except (ImportError, ModuleNotFoundError):
    from app.utils.mock_pipeline import analyze_fundus as _mock_analyze_fundus
    _REAL_PIPELINE_AVAILABLE = False
    logger.info("Real AI pipeline not found. Falling back to isolated UI mock adapter.")


def is_using_real_pipeline() -> bool:
    """Return True if connected to the real AI pipeline, False if using mock fallback."""
    return _REAL_PIPELINE_AVAILABLE


def analyze_fundus(image: Any) -> Dict[str, Any]:
    """
    Unified entrypoint for fundus image analysis.
    
    Routes to the teammate's real AI pipeline if present, or to the mock adapter for UI testing.
    """
    if _REAL_PIPELINE_AVAILABLE:
        result = _real_analyze_fundus(image)
        # Ensure result indicates real pipeline
        if "is_mock" not in result:
            result["is_mock"] = False
        return result
    else:
        return _mock_analyze_fundus(image)


# Convenience alias
run_screening_analysis = analyze_fundus
