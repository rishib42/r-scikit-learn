"""StandardScaler public estimator."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from rsklearn import _core
from rsklearn._validation import (
    one_to_one_feature_names,
    validate_numeric_2d,
    validate_numeric_2d_with_dtype,
)
from rsklearn.base import BaseEstimator, TransformerMixin
from rsklearn.utils.validation import check_is_fitted


class StandardScaler(TransformerMixin, BaseEstimator):
    """Standardize each feature using population mean and variance.

    NaNs are ignored while fitting and preserved while transforming. Infinity
    is rejected. Float32 transform input produces float32 output; other numeric
    input produces float64 output.
    """

    _parameter_names = ("with_mean", "with_std")
    _rsklearn_input_tags = {"allow_nan": True}
    _rsklearn_preserves_dtype = ["float64", "float32"]

    def __init__(self, *, with_mean: bool = True, with_std: bool = True) -> None:
        self.with_mean = with_mean
        self.with_std = with_std

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
        self._validate_params()
        already_fitted = hasattr(self, "n_features_in_")
        array = validate_numeric_2d(X, estimator=self, reset=not already_fitted)
        if already_fitted:
            mean, variance, scale, counts = _core.standard_merge(
                self._mean_state, self._variance_state, self._counts, array
            )
        else:
            mean, variance, scale, counts = _core.standard_fit(array)
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
        self._validate_params()
        check_is_fitted(self, "n_features_in_")
        array, output_dtype = validate_numeric_2d_with_dtype(
            X, estimator=self, reset=False
        )
        mean = self.mean_ if self.mean_ is not None else np.zeros(self.n_features_in_)
        scale = self.scale_ if self.scale_ is not None else np.ones(self.n_features_in_)
        output = _core.standard_transform(
            array, mean, scale, self.with_mean, self.with_std
        )
        return output.astype(output_dtype, copy=False)

    def inverse_transform(self, X: Any) -> NDArray[np.float64]:
        """Undo standardization."""
        self._validate_params()
        check_is_fitted(self, "n_features_in_")
        array, output_dtype = validate_numeric_2d_with_dtype(
            X, estimator=self, reset=False
        )
        mean = self.mean_ if self.mean_ is not None else np.zeros(self.n_features_in_)
        scale = self.scale_ if self.scale_ is not None else np.ones(self.n_features_in_)
        output = _core.standard_transform(
            array, mean, scale, self.with_mean, self.with_std, inverse=True
        )
        return output.astype(output_dtype, copy=False)

    def get_feature_names_out(self, input_features: Any = None) -> NDArray[Any]:
        """Return unchanged output feature names."""
        return one_to_one_feature_names(self, input_features)
