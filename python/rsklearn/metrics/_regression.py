"""Regression metrics backed by Rust reductions."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from rsklearn import _core
from rsklearn.metrics._validation import (
    UndefinedMetricWarning,
    aggregate_outputs,
    validate_regression_targets,
    validate_sample_weight,
)


def _reductions(y_true: Any, y_pred: Any, sample_weight: Any) -> tuple[Any, ...]:
    expected, predicted = validate_regression_targets(y_true, y_pred)
    weights = validate_sample_weight(sample_weight, expected.shape[0])
    return (
        *_core.metric_regression_reductions(expected, predicted, weights),
        expected.shape[0],
    )


def _error(
    y_true: Any, y_pred: Any, sample_weight: Any, *, squared: bool
) -> tuple[Any, float]:
    expected, predicted = validate_regression_targets(y_true, y_pred)
    if sample_weight is None:
        return (
            _core.metric_regression_error_unweighted(expected, predicted, squared),
            float(expected.shape[0]),
        )
    weights = validate_sample_weight(sample_weight, expected.shape[0])
    return _core.metric_regression_error(expected, predicted, weights, squared)


def mean_squared_error(
    y_true: Any,
    y_pred: Any,
    *,
    sample_weight: Any = None,
    multioutput: Any = "uniform_average",
) -> Any:
    """Return the non-negative mean squared prediction error."""
    squared, weight_sum = _error(y_true, y_pred, sample_weight, squared=True)
    return aggregate_outputs(np.asarray(squared) / weight_sum, multioutput)


def mean_absolute_error(
    y_true: Any,
    y_pred: Any,
    *,
    sample_weight: Any = None,
    multioutput: Any = "uniform_average",
) -> Any:
    """Return the non-negative mean absolute prediction error."""
    absolute, weight_sum = _error(y_true, y_pred, sample_weight, squared=False)
    return aggregate_outputs(np.asarray(absolute) / weight_sum, multioutput)


def r2_score(
    y_true: Any,
    y_pred: Any,
    *,
    sample_weight: Any = None,
    multioutput: Any = "uniform_average",
    force_finite: bool = True,
) -> Any:
    """Return the coefficient of determination."""
    if not isinstance(force_finite, (bool, np.bool_)):
        raise TypeError("force_finite must be bool")
    _, squared, true_sum, true_squared_sum, weight_sum, samples = _reductions(
        y_true, y_pred, sample_weight
    )
    if samples < 2:
        warnings.warn(
            "R^2 score is not well-defined with less than two samples.",
            UndefinedMetricWarning,
            stacklevel=2,
        )
        values = np.full(np.asarray(squared).shape, np.nan)
        return aggregate_outputs(values, multioutput)
    numerator = np.asarray(squared)
    denominator = np.asarray(true_squared_sum) - np.asarray(true_sum) ** 2 / weight_sum
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = 1.0 - numerator / denominator
    if force_finite:
        constant = denominator == 0
        scores[constant & (numerator == 0)] = 1.0
        scores[constant & (numerator != 0)] = 0.0
    if multioutput == "variance_weighted":
        if float(np.sum(denominator)) == 0:
            return float(np.mean(scores))
        return float(np.average(scores, weights=denominator))
    return aggregate_outputs(scores, multioutput)
