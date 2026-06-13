"""Shared validation helpers for public estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from rsklearn.utils.validation import check_array, validate_data


def validate_numeric_2d_with_dtype(
    X: Any, *, estimator: Any, reset: bool | None = None
) -> tuple[NDArray[np.float64], np.dtype[Any]]:
    """Return a non-empty contiguous float64 matrix and its output dtype."""
    original = np.asarray(X)
    parameters = {"dtype": np.float64, "order": "C", "ensure_all_finite": "allow-nan"}
    if reset is None or isinstance(estimator, str):
        array = check_array(X, estimator=estimator, **parameters)
    else:
        array = validate_data(estimator, X, reset=reset, **parameters)
    output_dtype = (
        np.dtype(np.float32)
        if original.dtype == np.dtype(np.float32)
        else np.dtype(np.float64)
    )
    return np.ascontiguousarray(array, dtype=np.float64), output_dtype


def validate_numeric_2d(
    X: Any, *, estimator: Any, reset: bool | None = None
) -> NDArray[np.float64]:
    """Return a non-empty, C-contiguous float64 matrix."""
    return validate_numeric_2d_with_dtype(X, estimator=estimator, reset=reset)[0]


def validate_normalizer_2d(
    X: Any, *, estimator: Any, reset: bool, copy: bool
) -> NDArray[np.float32] | NDArray[np.float64]:
    """Return finite contiguous float32 or float64 input for Normalizer."""
    array = validate_data(
        estimator,
        X,
        reset=reset,
        dtype="numeric",
        order="C",
        copy=copy,
        ensure_all_finite=True,
    )
    dtype = np.float32 if array.dtype == np.dtype(np.float32) else np.float64
    try:
        return np.ascontiguousarray(array, dtype=dtype)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{type(estimator).__name__} requires numeric input") from error


def check_feature_count(
    X: NDArray[np.float64], n_features_in: int, *, estimator: str
) -> None:
    """Reject matrices with a different feature count than fitted data.

    Kept for backward compatibility. New estimators should use ``validate_data``.
    """
    if X.shape[1] != n_features_in:
        raise ValueError(
            f"{estimator} expected {n_features_in} features, got {X.shape[1]}"
        )


def validate_labels(y: Any) -> tuple[NDArray[Any], str, np.dtype[Any]]:
    """Validate one-dimensional labels and select a dtype-safe core pathway."""
    array = np.asarray(y)
    input_dtype = array.dtype
    if array.ndim != 1:
        raise ValueError(
            f"LabelEncoder expected a 1-dimensional array, got {array.ndim}D"
        )
    if np.issubdtype(array.dtype, np.bool_):
        return np.ascontiguousarray(array, dtype=np.bool_), "bool", input_dtype
    if np.issubdtype(array.dtype, np.signedinteger):
        return np.ascontiguousarray(array, dtype=np.int64), "signed", input_dtype
    if np.issubdtype(array.dtype, np.unsignedinteger):
        return np.ascontiguousarray(array, dtype=np.uint64), "unsigned", input_dtype
    if np.issubdtype(array.dtype, np.floating):
        return np.ascontiguousarray(array, dtype=np.float64), "float", input_dtype
    if np.issubdtype(array.dtype, np.str_):
        return np.ascontiguousarray(array), "string", input_dtype
    if array.dtype == object:
        values = array.tolist()
        if all(isinstance(value, str) for value in values):
            strings = np.ascontiguousarray(array.astype(str))
            return strings, "string", strings.dtype
        if all(isinstance(value, (bool, np.bool_)) for value in values):
            return np.asarray(values, dtype=np.bool_), "bool", np.dtype(np.bool_)
        if all(
            isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_))
            for value in values
        ):
            minimum = min(values, default=0)
            maximum = max(values, default=0)
            if minimum < np.iinfo(np.int64).min or maximum > np.iinfo(np.uint64).max:
                raise ValueError(
                    "LabelEncoder integer labels must fit in int64 or uint64"
                )
            if minimum >= 0 and maximum > np.iinfo(np.int64).max:
                return (
                    np.asarray(values, dtype=np.uint64),
                    "unsigned",
                    np.dtype(np.uint64),
                )
            return np.asarray(values, dtype=np.int64), "signed", np.dtype(np.int64)
        if all(
            isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating))
            for value in values
        ):
            return np.asarray(values, dtype=np.float64), "float", np.dtype(np.float64)
    raise TypeError("LabelEncoder supports numeric, boolean, or string labels")


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
