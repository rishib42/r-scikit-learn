"""Shared metric input validation."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class UndefinedMetricWarning(UserWarning):
    """Warning used when a metric has no mathematically defined value."""


def validate_sample_weight(
    sample_weight: Any,
    samples: int,
    *,
    allow_zero_total: bool = False,
    zero_total_error: type[Exception] = ZeroDivisionError,
) -> NDArray[np.float64]:
    if sample_weight is None:
        weights = np.ones(samples, dtype=np.float64)
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)
        if weights.ndim != 1 or weights.size != samples:
            raise ValueError("sample_weight must contain one value per sample")
        if not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("sample_weight must contain finite non-negative values")
        weights = np.ascontiguousarray(weights)
    if not allow_zero_total and float(np.sum(weights)) == 0:
        raise zero_total_error("sample_weight contains only zero values")
    return weights


def validate_regression_targets(
    y_true: Any, y_pred: Any
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    expected = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    if expected.ndim not in (1, 2) or predicted.ndim not in (1, 2):
        raise ValueError("regression targets must be one- or two-dimensional")
    if expected.ndim == 1:
        expected = expected[:, None]
    if predicted.ndim == 1:
        predicted = predicted[:, None]
    if expected.shape != predicted.shape:
        raise ValueError(
            "y_true and y_pred have different shapes: "
            f"{expected.shape} and {predicted.shape}"
        )
    if expected.shape[0] < 1:
        raise ValueError("regression metrics require at least one sample")
    return np.ascontiguousarray(expected), np.ascontiguousarray(predicted)


def aggregate_outputs(values: NDArray[np.float64], multioutput: Any) -> Any:
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return values
        if multioutput != "uniform_average":
            raise ValueError(
                "multioutput must be 'raw_values', 'uniform_average', or weights"
            )
        return float(np.mean(values))
    weights = np.asarray(multioutput, dtype=np.float64)
    if weights.ndim != 1 or weights.size != values.size:
        raise ValueError("multioutput weights must match the number of outputs")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("multioutput weights must be finite and non-negative")
    return float(np.average(values, weights=weights))
