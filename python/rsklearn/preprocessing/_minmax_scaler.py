"""MinMaxScaler public estimator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rsklearn import _core
from rsklearn._validation import validate_numeric_2d, validate_numeric_2d_with_dtype
from rsklearn.base import BaseEstimator, TransformerMixin
from rsklearn.utils.validation import check_is_fitted


class MinMaxScaler(TransformerMixin, BaseEstimator):
    """Scale each feature into a requested range."""

    _parameter_names = ("feature_range", "clip")
    _rsklearn_input_tags = {"allow_nan": True}
    _rsklearn_preserves_dtype = ["float64", "float32"]

    def __init__(
        self, *, feature_range: tuple[float, float] = (0.0, 1.0), clip: bool = False
    ) -> None:
        self.feature_range = feature_range
        self.clip = clip

    def _validate_params(self) -> None:
        if (
            not isinstance(self.feature_range, Sequence)
            or isinstance(self.feature_range, (str, bytes))
            or len(self.feature_range) != 2
        ):
            raise TypeError("feature_range must be a pair of finite numbers")
        low, high = self.feature_range
        if (
            isinstance(low, bool)
            or isinstance(high, bool)
            or not np.isfinite(low)
            or not np.isfinite(high)
        ):
            raise ValueError("feature_range must contain finite numbers")
        if low >= high:
            raise ValueError("feature_range minimum must be smaller than maximum")
        if not isinstance(self.clip, bool):
            raise TypeError("clip must be bool")

    def fit(self, X: Any, y: Any = None) -> MinMaxScaler:
        """Learn per-feature minimum and maximum and return self."""
        del y
        for attribute in (
            "min_",
            "scale_",
            "data_min_",
            "data_max_",
            "data_range_",
            "n_features_in_",
            "n_samples_seen_",
        ):
            if hasattr(self, attribute):
                delattr(self, attribute)
        return self.partial_fit(X)

    def partial_fit(self, X: Any, y: Any = None) -> MinMaxScaler:
        """Update per-feature minimum and maximum from a batch."""
        del y
        self._validate_params()
        already_fitted = hasattr(self, "n_features_in_")
        array = validate_numeric_2d(X, estimator=self, reset=not already_fitted)
        batch_min, batch_max, _ = _core.minmax_fit(array)
        if already_fitted:
            self.data_min_ = np.minimum(self.data_min_, batch_min)
            self.data_max_ = np.maximum(self.data_max_, batch_max)
            self.n_samples_seen_ += array.shape[0]
        else:
            self.data_min_ = batch_min
            self.data_max_ = batch_max
            self.n_samples_seen_ = array.shape[0]
        self.data_range_ = self.data_max_ - self.data_min_
        low, high = self.feature_range
        safe_range = np.where(self.data_range_ == 0.0, 1.0, self.data_range_)
        self.scale_ = (high - low) / safe_range
        self.min_ = low - self.data_min_ * self.scale_
        return self

    def transform(self, X: Any) -> NDArray[np.float64]:
        """Scale X using fitted feature ranges."""
        self._validate_params()
        check_is_fitted(self, ("n_features_in_", "scale_", "min_"))
        array, output_dtype = validate_numeric_2d_with_dtype(
            X, estimator=self, reset=False
        )
        low, high = self.feature_range
        output = _core.minmax_transform(
            array, self.scale_, self.min_, low, high, self.clip
        )
        return output.astype(output_dtype, copy=False)

    def inverse_transform(self, X: Any) -> NDArray[np.float64]:
        """Undo min-max scaling."""
        self._validate_params()
        check_is_fitted(self, ("n_features_in_", "scale_", "min_"))
        array, output_dtype = validate_numeric_2d_with_dtype(
            X, estimator=self, reset=False
        )
        low, high = self.feature_range
        output = _core.minmax_transform(
            array, self.scale_, self.min_, low, high, False, inverse=True
        )
        return output.astype(output_dtype, copy=False)
