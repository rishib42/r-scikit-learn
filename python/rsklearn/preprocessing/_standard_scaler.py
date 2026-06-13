"""StandardScaler public estimator."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from rsklearn import _core
from rsklearn._validation import check_feature_count, validate_numeric_2d

from ._base import EstimatorMixin


class StandardScaler(EstimatorMixin):
    """Standardize each feature using population mean and variance.

    NaN and infinity are rejected in this MVP. Inputs are copied to contiguous
    float64 arrays before entering Rust.
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
        array = validate_numeric_2d(X, estimator=type(self).__name__)
        mean, variance, scale = _core.standard_fit(array)
        self.mean_ = mean if self.with_mean or self.with_std else None
        self.var_ = variance if self.with_std else None
        self.scale_ = scale if self.with_std else None
        self.n_features_in_ = array.shape[1]
        self.n_samples_seen_ = array.shape[0]
        return self

    def transform(self, X: Any) -> NDArray[np.float64]:
        """Standardize X using fitted statistics."""
        self._check_fitted("n_features_in_")
        array = validate_numeric_2d(X, estimator=type(self).__name__)
        check_feature_count(array, self.n_features_in_, estimator=type(self).__name__)
        mean = self.mean_ if self.mean_ is not None else np.zeros(self.n_features_in_)
        scale = self.scale_ if self.scale_ is not None else np.ones(self.n_features_in_)
        return _core.standard_transform(
            array, mean, scale, self.with_mean, self.with_std
        )

    def fit_transform(self, X: Any, y: Any = None) -> NDArray[np.float64]:
        """Fit to X and return its standardized representation."""
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: Any) -> NDArray[np.float64]:
        """Undo standardization."""
        self._check_fitted("n_features_in_")
        array = validate_numeric_2d(X, estimator=type(self).__name__)
        check_feature_count(array, self.n_features_in_, estimator=type(self).__name__)
        mean = self.mean_ if self.mean_ is not None else np.zeros(self.n_features_in_)
        scale = self.scale_ if self.scale_ is not None else np.ones(self.n_features_in_)
        return _core.standard_transform(
            array, mean, scale, self.with_mean, self.with_std, inverse=True
        )
