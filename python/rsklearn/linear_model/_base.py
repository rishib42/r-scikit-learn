"""Shared dense linear-model infrastructure."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from rsklearn import _core
from rsklearn.metrics._validation import validate_sample_weight
from rsklearn.utils.validation import check_is_fitted, validate_data


def validate_regression_fit(
    estimator: Any, X: Any, y: Any, sample_weight: Any
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], bool]:
    """Validate finite dense regression data and preserve target dimensionality."""
    if y is None:
        raise ValueError(
            f"{type(estimator).__name__} requires y to be passed, but the target "
            "y is None"
        )
    original_y = np.asarray(y)
    X_array = validate_data(
        estimator,
        X,
        reset=True,
        dtype=np.float64,
        order="C",
        ensure_all_finite=True,
    )
    try:
        y_array = np.asarray(y, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError("regression targets must be numeric") from error
    if y_array.ndim not in (1, 2) or y_array.shape[0] != X_array.shape[0]:
        raise ValueError(
            "y must be one- or two-dimensional with one target row per X sample"
        )
    if y_array.shape[0] == 0 or not np.isfinite(y_array).all():
        raise ValueError("regression targets must contain finite values")
    single_output = original_y.ndim == 1
    if y_array.ndim == 1:
        y_array = y_array[:, None]
    weights = validate_sample_weight(sample_weight, X_array.shape[0])
    return (
        np.ascontiguousarray(X_array, dtype=np.float64),
        np.ascontiguousarray(y_array, dtype=np.float64),
        weights,
        single_output,
    )


def validate_predict_input(estimator: Any, X: Any) -> NDArray[np.float64]:
    """Validate finite dense prediction input."""
    check_is_fitted(estimator, ("coef_", "intercept_", "n_features_in_"))
    array = np.ascontiguousarray(
        validate_data(
            estimator,
            X,
            reset=False,
            dtype=np.float64,
            order="C",
            ensure_all_finite=False,
        ),
        dtype=np.float64,
    )
    if not _core.linear_all_finite(array):
        raise ValueError("input contains NaN or infinity")
    return array


def raw_linear_prediction(
    estimator: Any, X: Any, *, coefficients: Any = None, intercepts: Any = None
) -> NDArray[np.float64]:
    """Predict through NumPy's optimized dense BLAS path."""
    array = validate_predict_input(estimator, X)
    coef = estimator.coef_ if coefficients is None else coefficients
    intercept = estimator.intercept_ if intercepts is None else intercepts
    coef = np.asarray(coef, dtype=np.float64)
    intercept = np.asarray(intercept, dtype=np.float64)
    if coef.ndim == 1:
        coef = coef[None, :]
    if intercept.ndim == 0:
        intercept = intercept[None]
    return np.asarray(array @ coef.T + intercept)


class LinearModel:
    """Mixin exposing coefficient-based prediction."""

    def predict(self, X: Any) -> NDArray[np.float64]:
        """Predict using the learned linear model."""
        output = raw_linear_prediction(self, X)
        return output[:, 0] if getattr(self, "_single_output", False) else output
