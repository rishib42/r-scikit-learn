"""OrdinalEncoder public estimator."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from rsklearn.base import BaseEstimator, TransformerMixin
from rsklearn.preprocessing._categorical import (
    CategoricalState,
    discover_categories,
    encode_categories,
    state_from_categories,
)
from rsklearn.utils.validation import check_is_fitted, validate_data


def _is_nan(value: Any) -> bool:
    return isinstance(value, (float, np.floating)) and np.isnan(value)


class OrdinalEncoder(TransformerMixin, BaseEstimator):
    """Encode dense categorical features as ordinal integer codes.

    Category discovery and lookup use the shared safe-Rust categorical kernels.
    Sparse input is not currently supported.
    """

    _rsklearn_input_tags = {
        "allow_nan": True,
        "categorical": True,
        "string": True,
    }

    def __init__(
        self,
        *,
        categories: str | list[Any] = "auto",
        dtype: Any = np.float64,
        handle_unknown: str = "error",
        unknown_value: int | float | None = None,
        encoded_missing_value: int | float = np.nan,
        min_frequency: int | float | None = None,
        max_categories: int | None = None,
    ) -> None:
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.encoded_missing_value = encoded_missing_value
        self.min_frequency = min_frequency
        self.max_categories = max_categories

    def _validate_params(self) -> np.dtype[Any]:
        if not (
            self.categories == "auto"
            if isinstance(self.categories, str)
            else isinstance(self.categories, (list, tuple))
        ):
            raise TypeError("categories must be 'auto' or a list of array-like values")
        try:
            output_dtype = np.dtype(self.dtype)
        except TypeError as error:
            raise TypeError("dtype must be a numeric NumPy dtype") from error
        if not np.issubdtype(output_dtype, np.number):
            raise TypeError("dtype must be a numeric NumPy dtype")
        if self.handle_unknown not in ("error", "use_encoded_value"):
            raise ValueError("handle_unknown must be 'error' or 'use_encoded_value'")
        if self.handle_unknown == "error" and self.unknown_value is not None:
            raise TypeError("unknown_value must be None when handle_unknown='error'")
        if self.handle_unknown == "use_encoded_value":
            if self.unknown_value is None or not self._is_valid_code(
                self.unknown_value
            ):
                raise TypeError(
                    "unknown_value must be an integer or NaN when "
                    "handle_unknown='use_encoded_value'"
                )
        if not self._is_valid_code(self.encoded_missing_value):
            raise TypeError("encoded_missing_value must be an integer or NaN")
        for name in ("unknown_value", "encoded_missing_value"):
            value = getattr(self, name)
            if _is_nan(value) and not np.issubdtype(output_dtype, np.floating):
                raise ValueError(f"{name}=np.nan requires a floating-point dtype")
        if self.min_frequency is not None:
            if isinstance(self.min_frequency, (bool, np.bool_)):
                raise TypeError("min_frequency must be a positive int or float")
            if isinstance(self.min_frequency, (int, np.integer)):
                if self.min_frequency < 1:
                    raise ValueError("integer min_frequency must be at least 1")
            elif isinstance(self.min_frequency, (float, np.floating)):
                if not 0 < self.min_frequency < 1:
                    raise ValueError("float min_frequency must be in (0, 1)")
            else:
                raise TypeError("min_frequency must be a positive int or float")
        if self.max_categories is not None and (
            isinstance(self.max_categories, (bool, np.bool_))
            or not isinstance(self.max_categories, (int, np.integer))
            or self.max_categories < 1
        ):
            raise ValueError("max_categories must be an integer of at least 1")
        return output_dtype

    @staticmethod
    def _is_valid_code(value: Any) -> bool:
        return (
            isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_))
        ) or _is_nan(value)

    def _fit_state(self, X: Any) -> tuple[CategoricalState, NDArray[np.int64]]:
        if isinstance(self.categories, str) and self.categories == "auto":
            return discover_categories(X, estimator=self)
        return state_from_categories(X, self.categories, estimator=self)

    def _build_code_maps(self, encoded: NDArray[np.int64]) -> None:
        self._missing_indices: dict[int, int] = {}
        self._code_maps: list[NDArray[np.int64]] = []
        self._inverse_values: list[NDArray[Any]] = []
        infrequent_categories: list[NDArray[Any] | None] = []
        for feature, categories in enumerate(self.categories_):
            missing_index = next(
                (index for index, value in enumerate(categories) if _is_nan(value)),
                None,
            )
            if missing_index is not None:
                self._missing_indices[feature] = missing_index
            counts = np.bincount(encoded[:, feature], minlength=len(categories))
            candidates = np.asarray(
                [index for index in range(len(categories)) if index != missing_index],
                dtype=np.int64,
            )
            infrequent = np.zeros(len(categories), dtype=bool)
            if self.min_frequency is not None:
                threshold = (
                    self.min_frequency
                    if isinstance(self.min_frequency, (int, np.integer))
                    else self.min_frequency * encoded.shape[0]
                )
                infrequent[candidates[counts[candidates] < threshold]] = True
            if self.max_categories is not None:
                frequent = candidates[~infrequent[candidates]]
                remove = max(0, frequent.size - max(self.max_categories - 1, 0))
                if remove:
                    order = np.lexsort((frequent, counts[frequent]))
                    infrequent[frequent[order[:remove]]] = True
            infrequent_indices = candidates[infrequent[candidates]]
            infrequent_categories.append(
                categories[infrequent_indices] if infrequent_indices.size else None
            )
            frequent_indices = candidates[~infrequent[candidates]]
            mapping = np.full(len(categories), -1, dtype=np.int64)
            mapping[frequent_indices] = np.arange(frequent_indices.size)
            inverse_values: list[Any] = categories[frequent_indices].tolist()
            if infrequent_indices.size:
                mapping[infrequent_indices] = frequent_indices.size
                inverse_values.append("infrequent_sklearn")
            self._code_maps.append(mapping)
            self._inverse_values.append(np.asarray(inverse_values, dtype=object))
        frequency_grouping = (
            self.min_frequency is not None or self.max_categories is not None
        )
        if frequency_grouping:
            self.infrequent_categories_ = infrequent_categories
        elif hasattr(self, "infrequent_categories_"):
            del self.infrequent_categories_
        has_infrequent = any(values is not None for values in infrequent_categories)
        self._inverse_dtype = (
            np.dtype(object)
            if self.handle_unknown == "use_encoded_value" or has_infrequent
            else np.result_type(*(categories.dtype for categories in self.categories_))
        )

    def _validate_sentinels(self) -> None:
        for feature, inverse in enumerate(self._inverse_values):
            used = set(range(len(inverse)))
            if (
                self.handle_unknown == "use_encoded_value"
                and not _is_nan(self.unknown_value)
                and self.unknown_value in used
            ):
                raise ValueError(
                    "unknown_value must be distinct from category and missing codes"
                )
            if (
                feature in self._missing_indices
                and not _is_nan(self.encoded_missing_value)
                and self.encoded_missing_value in set(range(len(inverse)))
            ):
                raise ValueError(
                    "encoded_missing_value must be distinct from category codes"
                )

    def fit(self, X: Any, y: Any = None) -> OrdinalEncoder:
        """Learn categories and return self."""
        del y
        self._output_dtype = self._validate_params()
        self._category_state, encoded = self._fit_state(X)
        self.categories_ = list(self._category_state.categories)
        self._build_code_maps(encoded)
        self._validate_sentinels()
        return self

    def _transform_codes(self, X: Any) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
        encoded, known = encode_categories(X, self._category_state, estimator=self)
        if self.handle_unknown == "error" and not np.all(known):
            flat = int(np.flatnonzero(~known)[0])
            row, column = np.unravel_index(flat, known.shape)
            value = np.asarray(X, dtype=object)[row, column]
            raise ValueError(
                f"Found unknown category {value!r} in column {column} during transform"
            )
        return encoded, known

    def transform(self, X: Any) -> NDArray[Any]:
        """Encode X using learned ordinal category codes."""
        output_dtype = self._validate_params()
        check_is_fitted(self, ("categories_", "_category_state", "_code_maps"))
        self._validate_sentinels()
        encoded, known = self._transform_codes(X)
        output = np.empty(encoded.shape, dtype=output_dtype)
        for feature, mapping in enumerate(self._code_maps):
            valid = known[:, feature]
            output[valid, feature] = mapping[encoded[valid, feature]]
            if feature in self._missing_indices:
                missing = encoded[:, feature] == self._missing_indices[feature]
                output[missing, feature] = self.encoded_missing_value
            if not np.all(valid):
                output[~valid, feature] = self.unknown_value
        return output

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> NDArray[Any]:
        """Learn categories and encode X in one validation pass."""
        del y
        if fit_params:
            names = ", ".join(sorted(fit_params))
            raise TypeError(
                f"OrdinalEncoder.fit_transform got unexpected arguments: {names}"
            )
        self._output_dtype = self._validate_params()
        self._category_state, encoded = self._fit_state(X)
        self.categories_ = list(self._category_state.categories)
        self._build_code_maps(encoded)
        self._validate_sentinels()
        output = np.empty(encoded.shape, dtype=self._output_dtype)
        for feature, mapping in enumerate(self._code_maps):
            output[:, feature] = mapping[encoded[:, feature]]
            if feature in self._missing_indices:
                missing = encoded[:, feature] == self._missing_indices[feature]
                output[missing, feature] = self.encoded_missing_value
        return output

    def inverse_transform(self, X: Any) -> NDArray[Any]:
        """Convert ordinal codes back to categories."""
        self._validate_params()
        check_is_fitted(self, ("categories_", "_inverse_values"))
        values = validate_data(
            self,
            X,
            reset=False,
            dtype="numeric",
            ensure_all_finite=False,
        )
        output = np.empty(values.shape, dtype=self._inverse_dtype)
        for feature, inverse in enumerate(self._inverse_values):
            column = values[:, feature]
            unknown = (
                np.isnan(column)
                if _is_nan(self.unknown_value)
                else column == self.unknown_value
            )
            missing = np.zeros(column.shape, dtype=bool)
            if feature in self._missing_indices:
                missing = (
                    np.isnan(column)
                    if _is_nan(self.encoded_missing_value)
                    else column == self.encoded_missing_value
                )
            unknown &= ~missing
            if np.any(unknown):
                output[unknown, feature] = None
            if np.any(missing):
                output[missing, feature] = np.nan
            regular = ~(unknown | missing)
            regular_values = column[regular]
            if np.any(regular_values != np.floor(regular_values)):
                raise ValueError("encoded values must be integers")
            indices = regular_values.astype(np.int64)
            if np.any(indices < 0) or np.any(indices >= inverse.size):
                raise ValueError("encoded value is outside the valid range")
            output[regular, feature] = inverse[indices]
        return output

    def get_feature_names_out(self, input_features: Any = None) -> NDArray[Any]:
        """Return output feature names, which are unchanged by ordinal encoding."""
        check_is_fitted(self, "n_features_in_")
        if input_features is None:
            if hasattr(self, "feature_names_in_"):
                return self.feature_names_in_.copy()
            return np.asarray(
                [f"x{index}" for index in range(self.n_features_in_)], dtype=object
            )
        names = np.asarray(input_features, dtype=object)
        if names.ndim != 1 or names.size != self.n_features_in_:
            raise ValueError(
                f"input_features must contain {self.n_features_in_} feature names"
            )
        if not all(isinstance(name, str) for name in names):
            raise TypeError("input_features must contain only strings")
        if hasattr(self, "feature_names_in_") and not np.array_equal(
            names, self.feature_names_in_
        ):
            raise ValueError("input_features must match feature_names_in_")
        return names
