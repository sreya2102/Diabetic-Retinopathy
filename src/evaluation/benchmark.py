"""
Pipeline Benchmarking & Evaluation Suite Module.

Phase 1 Architecture Stub:
Defines interface for running benchmarking runs across datasets (e.g. IDRiD test set).
"""

from typing import Dict, Any, Optional
from src.evaluation.metrics import calculate_classification_metrics, calculate_referable_metrics


def run_benchmark(
    y_true: Sequence[int], y_pred: Sequence[int], dataset_name: str = "IDRiD"
) -> Dict[str, Any]:
    """
    Execute benchmark evaluation pass on predictions vs targets.

    Args:
        y_true: True ground truth DR labels (0-4).
        y_pred: Model predicted DR labels (0-4).
        dataset_name: Name of benchmark dataset.

    Returns:
        Dictionary containing dataset metadata, 5-class metrics, and referable DR metrics.
    """
    class_metrics = calculate_classification_metrics(y_true, y_pred)
    referable_metrics = calculate_referable_metrics(y_true, y_pred)

    return {
        "dataset": dataset_name,
        "sample_count": len(y_true),
        "multi_class_metrics": class_metrics,
        "referable_metrics": referable_metrics,
    }
