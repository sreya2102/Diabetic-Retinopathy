"""
Evaluation Metrics Module for DR Classification and Referable DR Screening.

Computes multi-class metrics (Accuracy, Quadratic Weighted Kappa) and binary
referable DR metrics (Sensitivity, Specificity, Precision, Recall, F1, ROC-AUC).
"""

from typing import Dict, Any, Sequence, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from config.settings import REFERABLE_GRADES


def calculate_referable_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], y_prob: Optional[Sequence[float]] = None
) -> Dict[str, float]:
    """
    Calculate binary classification metrics for Referable DR Screening.

    Binary Mapping:
    - Positive (1): Grade 2, 3, 4 (Moderate NPDR+)
    - Negative (0): Grade 0, 1 (No DR or Mild NPDR)

    Args:
        y_true: True DR grade targets (0-4).
        y_pred: Predicted DR grade predictions (0-4).
        y_prob: Optional predicted probabilities for referable DR class.

    Returns:
        Dictionary containing sensitivity, specificity, precision, recall, f1, and optional roc_auc.
    """
    # Map multi-class DR grades to binary referable DR status (0 or 1)
    binary_true = np.array([1 if grade in REFERABLE_GRADES else 0 for grade in y_true])
    binary_pred = np.array([1 if grade in REFERABLE_GRADES else 0 for grade in y_pred])

    tn, fp, fn, tp = confusion_matrix(binary_true, binary_pred, labels=[0, 1]).ravel()

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(precision_score(binary_true, binary_pred, zero_division=0))
    recall = float(recall_score(binary_true, binary_pred, zero_division=0))
    f1 = float(f1_score(binary_true, binary_pred, zero_division=0))

    auc = None
    if y_prob is not None and len(set(binary_true)) > 1:
        try:
            auc = float(roc_auc_score(binary_true, y_prob))
        except ValueError:
            auc = None

    results = {
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }
    if auc is not None:
        results["roc_auc"] = round(auc, 4)

    return results


def calculate_classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> Dict[str, Any]:
    """
    Calculate 5-class DR classification evaluation metrics.

    Args:
        y_true: True DR grades (0-4).
        y_pred: Predicted DR grades (0-4).

    Returns:
        Dictionary containing multi-class accuracy and Quadratic Weighted Kappa.
    """
    acc = float(accuracy_score(y_true, y_pred))
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))

    return {
        "accuracy": round(acc, 4),
        "quadratic_weighted_kappa": round(qwk, 4),
    }
