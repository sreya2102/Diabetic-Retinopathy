"""
Benchmarking Suite for DR Screening Performance Evaluation.

Compares dataset validation evaluation against target performance metrics:
- Sensitivity Target: > 90.0%
- Specificity Target: > 85.0%
"""

from typing import Dict, Any, Sequence, Optional
from src.evaluation.metrics import calculate_classification_metrics, calculate_referable_metrics


def run_benchmark(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Optional[Sequence[float]] = None,
    dataset_name: str = "IDRiD Validation Split",
) -> Dict[str, Any]:
    """
    Run evaluation benchmark pass and compare against clinical target goals.

    Args:
        y_true: Ground truth DR labels (0-4).
        y_pred: Predicted DR labels (0-4).
        y_prob: Optional positive referable DR class probabilities.
        dataset_name: Benchmark dataset name.

    Returns:
        Structured evaluation dictionary containing dataset metadata, 5-class metrics,
        referable DR metrics, and target compliance evaluation.
    """
    class_metrics = calculate_classification_metrics(y_true, y_pred)
    referable_metrics = calculate_referable_metrics(y_true, y_pred, y_prob=y_prob)

    sens = referable_metrics.get("sensitivity", 0.0)
    spec = referable_metrics.get("specificity", 0.0)

    sens_target_met = sens >= 0.90
    spec_target_met = spec >= 0.85

    return {
        "dataset": dataset_name,
        "sample_count": len(y_true),
        "multi_class_metrics": class_metrics,
        "referable_metrics": referable_metrics,
        "target_benchmarks": {
            "sensitivity_target": 0.90,
            "sensitivity_achieved": sens,
            "sensitivity_target_met": sens_target_met,
            "specificity_target": 0.85,
            "specificity_achieved": spec,
            "specificity_target_met": spec_target_met,
        },
    }
