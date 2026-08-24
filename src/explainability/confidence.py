"""
Confidence Calibration Module.

Provides temperature scaling and calibration handling to distinguish raw softmax outputs
from clinically calibrated confidence estimates.
"""

from typing import Dict, Any, List
import numpy as np


def calibrate_confidence(
    probabilities: List[float], temperature: float = 1.0, calibrated: bool = False
) -> Dict[str, Any]:
    """
    Compute confidence score and specify calibration status.

    Args:
        probabilities: List of raw softmax probability values for classes [0-4].
        temperature: Temperature scaling factor (T > 0).
        calibrated: True if model weights and logits have undergone Platt/Temperature calibration.

    Returns:
        Dictionary containing:
        - "confidence": float value
        - "is_calibrated": bool
        - "method": string calibration description
    """
    if not probabilities or len(probabilities) == 0:
        return {
            "confidence": 0.0,
            "is_calibrated": False,
            "method": "Uncalibrated - empty input probabilities",
        }

    raw_max_prob = float(np.max(probabilities))

    if not calibrated:
        return {
            "confidence": round(raw_max_prob, 4),
            "is_calibrated": False,
            "method": "Raw Softmax Probability (Uncalibrated)",
        }

    # Temperature scaling application placeholder for Phase 8
    scaled_probs = np.exp(np.log(np.array(probabilities) + 1e-12) / temperature)
    scaled_probs /= np.sum(scaled_probs)
    calibrated_conf = float(np.max(scaled_probs))

    return {
        "confidence": round(calibrated_conf, 4),
        "is_calibrated": True,
        "method": f"Temperature Scaling (T={temperature})",
    }
