"""Evaluation metrics, confusion matrix, and benchmarking package."""

from src.evaluation.benchmark import run_benchmark
from src.evaluation.confusion_matrix import generate_confusion_matrix
from src.evaluation.metrics import calculate_classification_metrics, calculate_referable_metrics

__all__ = [
    "calculate_classification_metrics",
    "calculate_referable_metrics",
    "generate_confusion_matrix",
    "run_benchmark",
]
