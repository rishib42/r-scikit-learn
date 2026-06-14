"""Classification and regression metrics."""

from ._classification import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from ._regression import mean_absolute_error, mean_squared_error, r2_score
from ._validation import UndefinedMetricWarning

__all__ = [
    "UndefinedMetricWarning",
    "accuracy_score",
    "confusion_matrix",
    "f1_score",
    "mean_absolute_error",
    "mean_squared_error",
    "precision_score",
    "r2_score",
    "recall_score",
]
