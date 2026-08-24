"""
Confusion Matrix Generation & Formatting Module.
"""

from typing import Optional, Sequence
import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_cm


def generate_confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int], labels: Optional[Sequence[int]] = None
) -> np.ndarray:
    """
    Generate confusion matrix array for 5-class DR grading.

    Args:
        y_true: True ground truth grade integer labels (0-4).
        y_pred: Predicted grade integer labels (0-4).
        labels: Optional explicit label ordering list (default [0, 1, 2, 3, 4]).

    Returns:
        2D numpy array confusion matrix of shape (5, 5).
    """
    if labels is None:
        labels = [0, 1, 2, 3, 4]

    cm = sklearn_cm(y_true, y_pred, labels=labels)
    return cm
