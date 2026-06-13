"""Shared validation helpers for public estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def validate_numeric_2d(X: Any, *, estimator: str) -> NDArray[np.float64]:
    """Return a finite, non-empty, C-contiguous float64 matrix."""
    try:
        array = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{estimator} requires numeric input") from error
    if array.ndim != 2:
        raise ValueError(
            f"{estimator} expected a 2-dimensional array, got {array.ndim}D"
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{estimator} requires at least one sample and one feature")
    if not np.isfinite(array).all():
        raise ValueError(f"{estimator} does not support NaN or infinity")
    return np.ascontiguousarray(array, dtype=np.float64)


def check_feature_count(
    X: NDArray[np.float64], n_features_in: int, *, estimator: str
) -> None:
    """Reject matrices with a different feature count than fitted data."""
    if X.shape[1] != n_features_in:
        raise ValueError(
            f"{estimator} expected {n_features_in} features, got {X.shape[1]}"
        )


def validate_labels(y: Any) -> tuple[NDArray[Any], str]:
    """Validate supported homogeneous numeric or Unicode labels."""
    if isinstance(y, (list, tuple)):
        has_strings = any(isinstance(value, str) for value in y)
        has_non_strings = any(not isinstance(value, str) for value in y)
        if has_strings and has_non_strings:
            raise TypeError(
                "LabelEncoder supports homogeneous numeric or string labels only"
            )
    array = np.asarray(y)
    if array.ndim != 1:
        raise ValueError(
            f"LabelEncoder expected a 1-dimensional array, got {array.ndim}D"
        )
    if array.size == 0:
        raise ValueError("LabelEncoder requires at least one label")
    if np.issubdtype(array.dtype, np.bool_):
        raise TypeError("LabelEncoder does not support boolean labels")
    if np.issubdtype(array.dtype, np.number):
        numeric = np.asarray(array, dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError("LabelEncoder does not support NaN or infinity")
        return np.ascontiguousarray(numeric), "numeric"
    if np.issubdtype(array.dtype, np.str_):
        return np.ascontiguousarray(array), "string"
    if array.dtype == object:
        values = array.tolist()
        if all(isinstance(value, str) for value in values):
            return np.ascontiguousarray(array.astype(str)), "string"
        if all(
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            for value in values
        ):
            numeric = np.asarray(values, dtype=np.float64)
            if not np.isfinite(numeric).all():
                raise ValueError("LabelEncoder does not support NaN or infinity")
            return numeric, "numeric"
    raise TypeError("LabelEncoder supports homogeneous numeric or string labels only")


def validate_codes(y: Any) -> NDArray[np.int64]:
    """Validate one-dimensional integer encoded labels."""
    array = np.asarray(y)
    if array.ndim != 1:
        raise ValueError(
            f"LabelEncoder expected a 1-dimensional array, got {array.ndim}D"
        )
    if array.size == 0:
        return np.asarray(array, dtype=np.int64)
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise TypeError("inverse_transform requires integer encoded labels")
    return np.ascontiguousarray(array, dtype=np.int64)
