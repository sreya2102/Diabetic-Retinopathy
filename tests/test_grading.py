"""
Unit tests for DR clinical grading definitions and referable mapping.
"""

import pytest
from src.classification.grading import format_grading_result, grade_to_label, is_referable


def test_grade_to_label():
    """Verify grade_to_label returns exact clinical scale labels for 0-4."""
    assert grade_to_label(0) == "No DR"
    assert grade_to_label(1) == "Mild NPDR"
    assert grade_to_label(2) == "Moderate NPDR"
    assert grade_to_label(3) == "Severe NPDR"
    assert grade_to_label(4) == "Proliferative DR"


def test_is_referable():
    """
    Explicitly verify referable DR definitions:
    0 -> No DR -> non-referable (False)
    1 -> Mild NPDR -> non-referable (False)
    2 -> Moderate NPDR -> referable (True)
    3 -> Severe NPDR -> referable (True)
    4 -> Proliferative DR -> referable (True)
    """
    assert is_referable(0) is False
    assert is_referable(1) is False
    assert is_referable(2) is True
    assert is_referable(3) is True
    assert is_referable(4) is True


def test_invalid_grade_raises_error():
    """Verify invalid grade integers raise ValueError."""
    with pytest.raises(ValueError):
        grade_to_label(5)

    with pytest.raises(ValueError):
        is_referable(-1)


def test_format_grading_result():
    """Verify format_grading_result constructs valid response dictionary."""
    res = format_grading_result(2, 0.85, [0.05, 0.1, 0.85, 0.0, 0.0])
    assert res["class_id"] == 2
    assert res["label"] == "Moderate NPDR"
    assert res["referable"] is True
    assert res["confidence"] == 0.85
    assert len(res["probabilities"]) == 5
