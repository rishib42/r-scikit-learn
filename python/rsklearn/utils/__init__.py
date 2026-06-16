"""Utility functions for estimator authors and users."""

from .sparse import (
    SparseComponents,
    scale_sparse_columns,
    sparse_components,
    sparse_from_components,
    sparse_max_abs,
    sparse_max_abs_checked,
    sparse_standard_stats,
    sparse_standard_stats_checked,
    validate_compressed_structure,
)
from .validation import (
    NotFittedError,
    check_array,
    check_is_fitted,
    check_X_y,
    validate_data,
)

__all__ = [
    "SparseComponents",
    "NotFittedError",
    "check_array",
    "check_is_fitted",
    "check_X_y",
    "scale_sparse_columns",
    "sparse_components",
    "sparse_from_components",
    "sparse_max_abs",
    "sparse_max_abs_checked",
    "sparse_standard_stats",
    "sparse_standard_stats_checked",
    "validate_compressed_structure",
    "validate_data",
]
