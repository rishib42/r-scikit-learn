"""RobustScaler public estimator."""

from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rsklearn import _core
from rsklearn._validation import one_to_one_feature_names, validate_preserving_float_2d
from rsklearn.base import BaseEstimator, TransformerMixin
from rsklearn.utils.validation import check_is_fitted


def _normal_quantile(probability: float) -> float:
    if probability == 0.0:
        return -np.inf
    if probability == 1.0:
        return np.inf
    return NormalDist().inv_cdf(probability)


class RobustScaler(TransformerMixin, BaseEstimator):
    """Scale features using statistics robust to outliers.

    Medians and linear-interpolated quantiles are computed by the Rust core.
    NaNs are ignored while fitting and preserved while transforming.
    """

    _rsklearn_input_tags = {"allow_nan": True}
    _rsklearn_preserves_dtype = ["float64", "float32"]

    def __init__(
        self,
        *,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: tuple[float, float] = (25.0, 75.0),
        copy: bool = True,
        unit_variance: bool = False,
    ) -> None:
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.copy = copy
        self.unit_variance = unit_variance

    def _validate_params(self) -> None:
        for name in ("with_centering", "with_scaling", "copy", "unit_variance"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise TypeError(f"{name} must be bool")
        if not isinstance(self.quantile_range, tuple) or len(self.quantile_range) != 2:
            raise TypeError("quantile_range must be a pair")
        low, high = self.quantile_range
        if isinstance(low, (bool, np.bool_)) or isinstance(high, (bool, np.bool_)):
            raise ValueError(f"Invalid quantile range: {self.quantile_range}")
        try:
            valid = np.isfinite(low) and np.isfinite(high) and 0 <= low <= high <= 100
        except TypeError as error:
            raise ValueError(
                f"Invalid quantile range: {self.quantile_range}"
            ) from error
        if not valid:
            raise ValueError(f"Invalid quantile range: {self.quantile_range}")

    def fit(self, X: Any, y: Any = None) -> RobustScaler:
        """Compute per-feature median and quantile range."""
        del y
        self._validate_params()
        array = validate_preserving_float_2d(
            X, estimator=self, reset=True, copy=False, allow_nan=True
        )
        if not self.with_centering and not self.with_scaling:
            self.center_ = None
            self.scale_ = None
            return self
        center, scale = _core.robust_fit(
            np.ascontiguousarray(array, dtype=np.float64),
            *self.quantile_range,
            self.with_centering,
            self.with_scaling,
        )
        self.center_ = (
            center.astype(array.dtype, copy=False) if self.with_centering else None
        )
        if self.with_scaling:
            if self.unit_variance:
                low, high = self.quantile_range
                adjustment = _normal_quantile(high / 100.0) - _normal_quantile(
                    low / 100.0
                )
                scale = scale / adjustment
            self.scale_ = scale
        else:
            self.scale_ = None
        return self

    def _transform_validated(
        self,
        array: NDArray[np.float32] | NDArray[np.float64],
        *,
        inverse: bool,
    ) -> NDArray[np.float32] | NDArray[np.float64]:
        center = self.center_ if self.center_ is not None else np.empty(0)
        scale = self.scale_ if self.scale_ is not None else np.empty(0)
        function = (
            _core.robust_transform_f32
            if array.dtype == np.dtype(np.float32)
            else _core.robust_transform_f64
        )
        output = function(
            array,
            np.asarray(center, dtype=np.float64),
            np.asarray(scale, dtype=np.float64),
            self.with_centering,
            self.with_scaling,
            inverse,
        )
        if not self.copy and array.flags.writeable:
            array[...] = output
            return array
        return output

    def transform(self, X: Any) -> NDArray[np.float32] | NDArray[np.float64]:
        """Center and scale X using fitted robust statistics."""
        self._validate_params()
        check_is_fitted(self, ("n_features_in_", "center_", "scale_"))
        array = validate_preserving_float_2d(
            X, estimator=self, reset=False, copy=False, allow_nan=True
        )
        return self._transform_validated(array, inverse=False)

    def inverse_transform(self, X: Any) -> NDArray[np.float32] | NDArray[np.float64]:
        """Undo robust centering and scaling."""
        self._validate_params()
        check_is_fitted(self, ("n_features_in_", "center_", "scale_"))
        array = validate_preserving_float_2d(
            X, estimator=self, reset=False, copy=False, allow_nan=True
        )
        return self._transform_validated(array, inverse=True)

    def get_feature_names_out(self, input_features: Any = None) -> NDArray[Any]:
        """Return unchanged output feature names."""
        return one_to_one_feature_names(self, input_features)
