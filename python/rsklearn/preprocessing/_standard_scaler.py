"""StandardScaler public estimator."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from rsklearn import _core
from rsklearn._validation import (
    check_feature_count,
    validate_numeric_2d,
    validate_numeric_2d_with_dtype,
)

from ._base import EstimatorMixin


class StandardScaler(EstimatorMixin):
    """Standardize each feature using population mean and variance.

    NaNs are ignored while fitting and preserved while transforming. Infinity
    is rejected. Float32 transform input produces float32 output; other numeric
    input produces float64 output.
    """

    _parameter_names = ("with_mean", "with_std")

    def __init__(self, *, with_mean: bool = True, with_std: bool = True) -> None:
        self.with_mean = with_mean
        self.with_std = with_std
        self._validate_params()

    def _validate_params(self) -> None:
        if not isinstance(self.with_mean, bool) or not isinstance(self.with_std, bool):
            raise TypeError("with_mean and with_std must be bool")

    def fit(self, X: Any, y: Any = None) -> StandardScaler:
        """Learn feature statistics and return self."""
        del y
        for attribute in (
            "mean_",
            "var_",
            "scale_",
            "n_features_in_",
            "n_samples_seen_",
            "_mean_state",
            "_variance_state",
            "_counts",
        ):
            if hasattr(self, attribute):
                delattr(self, attribute)
        return self.partial_fit(X)

    def partial_fit(self, X: Any, y: Any = None) -> StandardScaler:
        """Update feature statistics from a batch and return self."""
        del y
        array = validate_numeric_2d(X, estimator=type(self).__name__)
        if hasattr(self, "n_features_in_"):
            check_feature_count(
                array, self.n_features_in_, estimator=type(self).__name__
            )
            mean, variance, scale, counts = _core.standard_merge(
                self._mean_state, self._variance_state, self._counts, array
            )
        else:
            mean, variance, scale, counts = _core.standard_fit(array)
            self.n_features_in_ = array.shape[1]
        self._mean_state = mean
        self._variance_state = variance
        self._counts = counts
        self.mean_ = mean if self.with_mean or self.with_std else None
        self.var_ = variance if self.with_std else None
        self.scale_ = scale if self.with_std else None
        self.n_samples_seen_ = (
            int(counts[0]) if np.all(counts == counts[0]) else counts.copy()
        )
        return self

    def transform(self, X: Any) -> NDArray[np.float64]:
        """Standardize X using fitted statistics."""
        self._check_fitted("n_features_in_")
        array, output_dtype = validate_numeric_2d_with_dtype(
            X, estimator=type(self).__name__
        )
        check_feature_count(array, self.n_features_in_, estimator=type(self).__name__)
        mean = self.mean_ if self.mean_ is not None else np.zeros(self.n_features_in_)
        scale = self.scale_ if self.scale_ is not None else np.ones(self.n_features_in_)
        output = _core.standard_transform(
            array, mean, scale, self.with_mean, self.with_std
        )
        return output.astype(output_dtype, copy=False)

    def fit_transform(self, X: Any, y: Any = None) -> NDArray[np.float64]:
        """Fit to X and return its standardized representation."""
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: Any) -> NDArray[np.float64]:
        """Undo standardization."""
        self._check_fitted("n_features_in_")
        array, output_dtype = validate_numeric_2d_with_dtype(
            X, estimator=type(self).__name__
        )
        check_feature_count(array, self.n_features_in_, estimator=type(self).__name__)
        mean = self.mean_ if self.mean_ is not None else np.zeros(self.n_features_in_)
        scale = self.scale_ if self.scale_ is not None else np.ones(self.n_features_in_)
        output = _core.standard_transform(
            array, mean, scale, self.with_mean, self.with_std, inverse=True
        )
        return output.astype(output_dtype, copy=False)
