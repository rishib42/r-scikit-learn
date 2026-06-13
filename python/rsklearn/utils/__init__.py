"""Utility functions for estimator authors and users."""

from .sparse import (
    SparseComponents,
    scale_sparse_columns,
    sparse_components,
    sparse_from_components,
    validate_compressed_structure,
)
from .validation import check_array, check_is_fitted, check_X_y, validate_data

__all__ = [
    "SparseComponents",
    "check_array",
    "check_is_fitted",
    "check_X_y",
    "scale_sparse_columns",
    "sparse_components",
    "sparse_from_components",
    "validate_compressed_structure",
    "validate_data",
]
