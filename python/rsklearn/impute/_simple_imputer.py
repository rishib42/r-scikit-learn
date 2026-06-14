"""SimpleImputer public estimator."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rsklearn import _core
from rsklearn.base import BaseEstimator, TransformerMixin
from rsklearn.utils.validation import check_is_fitted, validate_data


def _is_nan(value: Any) -> bool:
    if (
        type(value).__module__.startswith("pandas.")
        and type(value).__name__ == "NAType"
    ):
        return True
    try:
        return bool(np.isscalar(value) and np.isnan(value))
    except TypeError:
        return False


def _missing_mask(array: NDArray[Any], missing_value: Any) -> NDArray[np.bool_]:
    if _is_nan(missing_value):
        if array.dtype.kind in "fc":
            return np.isnan(array)
        if array.dtype.kind in "biuUS":
            return np.zeros(array.shape, dtype=bool)
        return np.frompyfunc(_is_nan, 1, 1)(array).astype(bool)
    if array.dtype.kind != "O":
        try:
            return np.asarray(array == missing_value, dtype=bool)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "missing_values must be comparable with input values"
            ) from error

    def matches(value: Any) -> bool:
        if value is missing_value:
            return True
        try:
            return bool(value == missing_value)
        except (TypeError, ValueError):
            return False

    return np.frompyfunc(matches, 1, 1)(array).astype(bool)


@dataclass(frozen=True)
class _MissingIndicator:
    features_: NDArray[np.int64]
    missing_values: Any

    def transform(self, X: Any) -> NDArray[np.bool_]:
        """Return fitted missing-feature indicators for dense input."""
        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError("MissingIndicator expected a 2-dimensional array")
        if self.features_.size and self.features_[-1] >= array.shape[1]:
            raise ValueError("MissingIndicator input has too few features")
        return _missing_mask(array, self.missing_values)[:, self.features_]


class SimpleImputer(TransformerMixin, BaseEstimator):
    """Impute missing values using per-feature univariate statistics.

    Dense numeric statistics and replacement use safe-Rust kernels. Dense
    string and object inputs use dtype-preserving Python orchestration.
    """

    _rsklearn_input_tags = {"allow_nan": True, "categorical": True, "string": True}

    def __init__(
        self,
        *,
        missing_values: Any = np.nan,
        strategy: str | Any = "mean",
        fill_value: Any = None,
        copy: bool = True,
        add_indicator: bool = False,
        keep_empty_features: bool = False,
    ) -> None:
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
        self.copy = copy
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features

    def _validate_params(self) -> None:
        if not (
            callable(self.strategy)
            or self.strategy
            in {
                "mean",
                "median",
                "most_frequent",
                "constant",
            }
        ):
            raise ValueError(
                "strategy must be 'mean', 'median', 'most_frequent', 'constant', "
                "or a callable"
            )
        for name in ("copy", "add_indicator", "keep_empty_features"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise TypeError(f"{name} must be bool")

    def _validate_input(
        self, X: Any, *, reset: bool, defer_numeric_finite_check: bool = False
    ) -> NDArray[Any]:
        array = validate_data(
            self,
            X,
            reset=reset,
            dtype=None,
            copy=False,
            ensure_all_finite=False,
        )
        if array.dtype.kind == "c":
            raise ValueError("Complex data not supported")
        if array.dtype.kind in "f" and not defer_numeric_finite_check:
            if np.isinf(array).any():
                raise ValueError("SimpleImputer input contains infinity")
            if not _is_nan(self.missing_values) and np.isnan(array).any():
                raise ValueError(
                    "SimpleImputer input contains NaN, but missing_values is not NaN"
                )
        if array.dtype.kind == "O":
            for value in array.flat:
                if _is_nan(value):
                    if not _is_nan(self.missing_values):
                        raise ValueError(
                            "SimpleImputer input contains NaN, but missing_values is "
                            "not NaN"
                        )
                    continue
                if value is not None and not isinstance(
                    value, (str, int, float, bool, np.generic)
                ):
                    raise TypeError("SimpleImputer supports string or numeric values")
                if isinstance(value, (float, np.floating)):
                    if np.isinf(value):
                        raise ValueError("SimpleImputer input contains infinity")
        return array

    def _numeric_missing_value(self) -> tuple[float, bool]:
        if _is_nan(self.missing_values):
            return np.nan, True
        if isinstance(self.missing_values, (bool, int, float, np.number)):
            return float(self.missing_values), False
        raise ValueError("missing_values must be numeric for numeric input")

    def _constant_value(self, array: NDArray[Any]) -> Any:
        if self.fill_value is not None:
            return self.fill_value
        return "missing_value" if array.dtype.kind in "OUS" else 0

    @staticmethod
    def _most_frequent(values: NDArray[Any]) -> Any:
        uniques, counts = np.unique(values, return_counts=True)
        return uniques[np.argmax(counts)]

    def _fit_python(self, array: NDArray[Any], mask: NDArray[np.bool_]) -> NDArray[Any]:
        statistics: list[Any] = []
        for column in range(array.shape[1]):
            values = array[~mask[:, column], column]
            if values.size == 0:
                statistics.append(np.nan)
            elif self.strategy == "most_frequent":
                statistics.append(self._most_frequent(values))
            elif callable(self.strategy):
                statistics.append(self.strategy(np.asarray(values)))
            else:
                statistics.append(self._constant_value(array))
        dtype = object if array.dtype.kind in "OUS" else np.float64
        return np.asarray(statistics, dtype=dtype)

    def fit(self, X: Any, y: Any = None) -> SimpleImputer:
        """Learn one replacement statistic per feature."""
        del y
        self._validate_params()
        array = self._validate_input(
            X,
            reset=True,
            defer_numeric_finite_check=self.strategy == "mean",
        )
        self._input_dtype = array.dtype
        numeric_strategy = isinstance(self.strategy, str) and self.strategy in {
            "mean",
            "median",
            "most_frequent",
        }
        fused_mean = self.strategy == "mean" and array.dtype.kind in "biuf"
        numeric_array = array
        if isinstance(self.strategy, str) and self.strategy in {"mean", "median"}:
            try:
                numeric_array = np.asarray(array, dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"strategy={self.strategy!r} requires numeric input"
                ) from error
            if not fused_mean and np.isinf(numeric_array).any():
                raise ValueError("SimpleImputer input contains infinity")
            if (
                not fused_mean
                and not _is_nan(self.missing_values)
                and np.isnan(numeric_array).any()
            ):
                raise ValueError(
                    "SimpleImputer input contains NaN, but missing_values is not NaN"
                )
        self._numeric_path = bool(
            numeric_strategy
            and (array.dtype.kind in "biuf" or self.strategy in {"mean", "median"})
        )
        if self._numeric_path:
            missing_value, missing_is_nan = self._numeric_missing_value()
            numeric = np.ascontiguousarray(numeric_array, dtype=np.float64)
            if fused_mean:
                (
                    self.statistics_,
                    self._indicator_features,
                    empty_features,
                ) = _core.simple_imputer_mean_fit(
                    numeric, missing_value, missing_is_nan
                )
                empty = np.zeros(array.shape[1], dtype=bool)
                empty[empty_features] = True
            else:
                mask = _missing_mask(array, self.missing_values)
                self._indicator_features = np.flatnonzero(np.any(mask, axis=0)).astype(
                    np.int64
                )
                empty = np.all(mask, axis=0)
                self.statistics_ = _core.simple_imputer_fit(
                    numeric, self.strategy, missing_value, missing_is_nan
                )
        else:
            mask = _missing_mask(array, self.missing_values)
            self._indicator_features = np.flatnonzero(np.any(mask, axis=0)).astype(
                np.int64
            )
            empty = np.all(mask, axis=0)
            self.statistics_ = self._fit_python(array, mask)
        self.indicator_ = (
            _MissingIndicator(self._indicator_features, self.missing_values)
            if self.add_indicator
            else None
        )
        if self.strategy == "constant":
            self.statistics_[:] = self._constant_value(array)
        if self.keep_empty_features:
            self.statistics_[empty] = (
                self._constant_value(array) if self.strategy == "constant" else 0
            )
        self._retained_features = np.flatnonzero(
            self.keep_empty_features | ~empty
        ).astype(np.int64)
        return self

    def _warn_dropped_features(self) -> None:
        dropped = np.setdiff1d(
            np.arange(self.n_features_in_), self._retained_features, assume_unique=True
        )
        if dropped.size:
            warnings.warn(
                "Skipping features without any observed values: "
                f"{dropped.tolist()}. At least one non-missing value is needed for "
                f"imputation with strategy={self.strategy!r}.",
                UserWarning,
                stacklevel=3,
            )

    def transform(self, X: Any) -> NDArray[Any]:
        """Replace missing values and optionally append missing indicators."""
        self._validate_params()
        check_is_fitted(
            self, ("statistics_", "_retained_features", "_indicator_features")
        )
        array = self._validate_input(X, reset=False)
        mask = _missing_mask(array, self.missing_values)
        self._warn_dropped_features()
        if self._numeric_path:
            missing_value, missing_is_nan = self._numeric_missing_value()
            output = _core.simple_imputer_transform(
                np.ascontiguousarray(array, dtype=np.float64),
                np.asarray(self.statistics_, dtype=np.float64),
                self._retained_features,
                missing_value,
                missing_is_nan,
            )
            if array.dtype == np.dtype(np.float32):
                output = output.astype(np.float32)
            if (
                not self.copy
                and not self.add_indicator
                and output.shape == array.shape
                and array.dtype in (np.dtype(np.float32), np.dtype(np.float64))
                and array.flags.writeable
            ):
                array[...] = output
                output = array
        else:
            output_dtype = object if array.dtype.kind in "US" else array.dtype
            output = array[:, self._retained_features].astype(output_dtype, copy=True)
            retained_mask = mask[:, self._retained_features]
            for output_column, input_column in enumerate(self._retained_features):
                output[retained_mask[:, output_column], output_column] = (
                    self.statistics_[input_column]
                )
            if (
                not self.copy
                and output.shape[1] == array.shape[1]
                and array.flags.writeable
            ):
                array[...] = output
                output = array
        if self.add_indicator:
            indicator = mask[:, self._indicator_features].astype(output.dtype)
            output = np.concatenate((output, indicator), axis=1)
        return output

    def inverse_transform(self, X: Any) -> NDArray[Any]:
        """Restore imputed entries to the configured missing-value sentinel."""
        self._validate_params()
        check_is_fitted(self, ("statistics_", "_indicator_features"))
        if not self.add_indicator:
            raise ValueError("inverse_transform requires add_indicator=True")
        array = np.asarray(X)
        output_width = len(self._retained_features)
        expected = output_width + len(self._indicator_features)
        if array.ndim != 2 or array.shape[1] != expected:
            raise ValueError(f"inverse_transform expected {expected} features")
        output = np.full(
            (array.shape[0], self.n_features_in_),
            self.missing_values,
            dtype=object if self._input_dtype.kind in "OUS" else array.dtype,
        )
        output[:, self._retained_features] = array[:, :output_width]
        indicators = array[:, output_width:].astype(bool)
        for index, feature in enumerate(self._indicator_features):
            output[indicators[:, index], feature] = self.missing_values
        return output

    def get_feature_names_out(self, input_features: Any = None) -> NDArray[np.object_]:
        """Return output feature names, including optional indicator names."""
        check_is_fitted(self, ("statistics_", "_retained_features"))
        if input_features is None:
            features = getattr(
                self,
                "feature_names_in_",
                np.asarray([f"x{index}" for index in range(self.n_features_in_)]),
            )
        else:
            features = np.asarray(input_features, dtype=object)
            if features.ndim != 1 or features.size != self.n_features_in_:
                raise ValueError(
                    "input_features must contain one name per input feature"
                )
            if hasattr(self, "feature_names_in_") and not np.array_equal(
                features, self.feature_names_in_
            ):
                raise ValueError("input_features must match feature_names_in_")
        output = features[self._retained_features].astype(object)
        if self.add_indicator:
            indicators = np.asarray(
                [
                    f"missingindicator_{features[index]}"
                    for index in self._indicator_features
                ],
                dtype=object,
            )
            output = np.concatenate((output, indicators))
        return output
