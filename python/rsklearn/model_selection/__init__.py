"""Dataset splitting and cross-validation utilities."""

from ._split import BaseCrossValidator, KFold, StratifiedKFold, train_test_split
from ._validation import FitFailedWarning, cross_val_score

__all__ = [
    "BaseCrossValidator",
    "FitFailedWarning",
    "KFold",
    "StratifiedKFold",
    "cross_val_score",
    "train_test_split",
]
