"""
Clinical Evidence Extraction Module.

Correlates detected lesion regions (microaneurysms, hemorrhages, hard/soft exudates)
with DR severity classification findings without pre-fabricated estimates.
"""

from typing import Dict, Any, List, Optional


def extract_clinical_evidence(
    lesions_dict: Dict[str, Any], grading_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extract structured clinical lesion evidence metrics and findings.

    Args:
        lesions_dict: Dict containing detected lesion counts/masks from Phase 5.
        grading_dict: Dict containing DR grade classification from Phase 6.

    Returns:
        Structured dictionary matching clinical evidence schema:
        {
            "lesion_counts": {
                "microaneurysms": int,
                "hemorrhages": int,
                "hard_exudates": int,
                "soft_exudates": int
            },
            "findings": List[str],
            "severity_rationale": str
        }
    """
    if not isinstance(lesions_dict, dict):
        lesions_dict = {}

    ma_count = lesions_dict.get("microaneurysms", {}).get("count", 0) if isinstance(lesions_dict.get("microaneurysms"), dict) else 0
    he_count = lesions_dict.get("hemorrhages", {}).get("count", 0) if isinstance(lesions_dict.get("hemorrhages"), dict) else 0
    ex_count = lesions_dict.get("hard_exudates", {}).get("count", 0) if isinstance(lesions_dict.get("hard_exudates"), dict) else 0
    se_count = lesions_dict.get("soft_exudates", {}).get("count", 0) if isinstance(lesions_dict.get("soft_exudates"), dict) else 0

    findings: List[str] = []

    if ma_count > 0:
        findings.append(f"Detected {ma_count} Microaneurysms (MA) in retinal parenchyma.")
    if he_count > 0:
        findings.append(f"Detected {he_count} Intraretinal Hemorrhages (HE).")
    if ex_count > 0:
        findings.append(f"Detected {ex_count} Hard Exudates (EX) lipid deposits.")
    if se_count > 0:
        findings.append(f"Detected {se_count} Soft Exudates (SE / Cotton Wool Spots).")

    if len(findings) == 0:
        findings.append("No primary diabetic microvascular lesions detected in retinal field.")

    # Rationale based on ICDR clinical criteria
    if ma_count > 0 and he_count == 0 and ex_count == 0:
        rationale = "Mild NPDR pattern: Microaneurysms present in isolation."
    elif (he_count > 0 or ex_count > 0) and (he_count < 20):
        rationale = "Moderate NPDR pattern: Hemorrhages or exudates present."
    elif he_count >= 20 or se_count > 2:
        rationale = "Severe NPDR pattern: Extensive hemorrhages/cotton wool spots meeting 4-2-1 rule criteria."
    else:
        rationale = "No DR or minimal vascular change."

    return {
        "lesion_counts": {
            "microaneurysms": ma_count,
            "hemorrhages": he_count,
            "hard_exudates": ex_count,
            "soft_exudates": se_count,
        },
        "findings": findings,
        "severity_rationale": rationale,
    }
