"""
Confusion Matrix Generation & Visualization Module.
"""

from typing import Optional, Sequence
import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_cm


def generate_confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int], labels: Optional[Sequence[int]] = None
) -> np.ndarray:
    """
    Generate 5x5 confusion matrix array for DR severity grading.

    Args:
        y_true: Ground truth DR severity grade labels (0-4).
        y_pred: Predicted DR severity grade labels (0-4).
        labels: Explicit class label order list (default [0, 1, 2, 3, 4]).

    Returns:
        2D numpy array confusion matrix of shape (5, 5).
    """
    if labels is None:
        labels = [0, 1, 2, 3, 4]

    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Empty ground truth or prediction array provided for confusion matrix.")

    cm = sklearn_cm(y_true, y_pred, labels=labels)
    return cm
