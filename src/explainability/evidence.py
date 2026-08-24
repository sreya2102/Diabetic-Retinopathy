"""
Clinical Evidence Extraction Module.

Correlates detected lesion regions and features (microaneurysms, hemorrhages,
hard/soft exudates) with predicted DR severity.
"""

from typing import Dict, Any, List


def extract_clinical_evidence(lesions_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Synthesize detected lesion metadata into human-readable clinical evidence strings.

    Args:
        lesions_dict: Dictionary containing detected lesion counts/masks from lesion analysis.

    Returns:
        List of structured clinical evidence dicts:
        [{"lesion_type": str, "count": int, "severity_relevance": str}]
    """
    evidence: List[Dict[str, Any]] = []

    if not isinstance(lesions_dict, dict):
        return evidence

    for lesion_key in ["microaneurysms", "hemorrhages", "hard_exudates", "soft_exudates"]:
        val = lesions_dict.get(lesion_key, {})
        count = val.get("count", 0) if isinstance(val, dict) else (val if isinstance(val, int) else 0)

        if count > 0:
            evidence.append(
                {
                    "lesion_type": lesion_key,
                    "count": count,
                    "severity_relevance": f"Detected {count} {lesion_key} in fundus region.",
                }
            )

    return evidence
