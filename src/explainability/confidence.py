"""
Confidence Calibration Module.

Provides Temperature Scaling logit transformation and distinguishes raw softmax
probabilities from clinically calibrated confidence estimates.
"""

from typing import Dict, Any, List
import numpy as np


def calibrate_confidence(
    probabilities: List[float], temperature: float = 1.0, calibrated: bool = False
) -> Dict[str, Any]:
    """
    Compute confidence score and explicitly distinguish raw vs calibrated confidence.

    Args:
        probabilities: List of 5 class softmax probabilities.
        temperature: Temperature scaling factor (T > 0, default 1.0).
        calibrated: Set to True ONLY if logits/model have undergone empirical calibration.

    Returns:
        Structured confidence dictionary:
        {
            "confidence": float,
            "is_calibrated": bool,
            "method": str,
            "raw_max_prob": float
        }
    """
    if not probabilities or len(probabilities) == 0:
        return {
            "confidence": 0.0,
            "is_calibrated": False,
            "method": "Uncalibrated - Empty probabilities list",
            "raw_max_prob": 0.0,
        }

    raw_max_prob = float(np.max(probabilities))

    if not calibrated:
        return {
            "confidence": round(raw_max_prob, 4),
            "is_calibrated": False,
            "method": "Raw Softmax Probability (Uncalibrated)",
            "raw_max_prob": round(raw_max_prob, 4),
        }

    # Temperature Scaling: scaled_logits = logits / T
    # Re-apply Softmax to scaled probabilities
    logits = np.log(np.array(probabilities) + 1e-12)
    scaled_logits = logits / temperature
    scaled_probs = np.exp(scaled_logits - np.max(scaled_logits))
    scaled_probs /= np.sum(scaled_probs)

    calibrated_conf = float(np.max(scaled_probs))

    return {
        "confidence": round(calibrated_conf, 4),
        "is_calibrated": True,
        "method": f"Temperature Scaling Calibration (T={temperature:.2f})",
        "raw_max_prob": round(raw_max_prob, 4),
    }
