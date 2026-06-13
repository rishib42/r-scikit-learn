"""OneHotEncoder public estimator."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rsklearn import _core
from rsklearn.base import BaseEstimator, TransformerMixin
from rsklearn.preprocessing._categorical import (
    CategoricalState,
    discover_categories,
    encode_categories,
    state_from_categories,
)
from rsklearn.utils import SparseComponents, check_array, sparse_from_components
from rsklearn.utils.validation import check_is_fitted


def _is_nan(value: Any) -> bool:
    return isinstance(value, (float, np.floating)) and np.isnan(value)


def _category_equal(left: Any, right: Any) -> bool:
    return (_is_nan(left) and _is_nan(right)) or left == right


class OneHotEncoder(TransformerMixin, BaseEstimator):
    """Encode dense categorical input as sparse or dense one-hot features."""

    _rsklearn_input_tags = {
        "allow_nan": True,
        "categorical": True,
        "string": True,
    }

    def __init__(
        self,
        *,
        categories: str | list[Any] = "auto",
        drop: str | list[Any] | None = None,
        sparse_output: bool = True,
        dtype: Any = np.float64,
        handle_unknown: str = "error",
        min_frequency: int | float | None = None,
        max_categories: int | None = None,
        feature_name_combiner: str | Callable[[str, Any], str] = "concat",
    ) -> None:
        self.categories = categories
        self.drop = drop
        self.sparse_output = sparse_output
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.feature_name_combiner = feature_name_combiner

    def _validate_params(self) -> np.dtype[Any]:
        if not (
            self.categories == "auto"
            if isinstance(self.categories, str)
            else isinstance(self.categories, (list, tuple))
        ):
            raise TypeError("categories must be 'auto' or a list of array-like values")
        valid_drop_string = isinstance(self.drop, str) and self.drop in (
            "first",
            "if_binary",
        )
        if (
            self.drop is not None
            and not valid_drop_string
            and not isinstance(self.drop, (list, tuple, np.ndarray))
        ):
            raise TypeError("drop must be None, 'first', 'if_binary', or array-like")
        if not isinstance(self.sparse_output, (bool, np.bool_)):
            raise TypeError("sparse_output must be bool")
        try:
            output_dtype = np.dtype(self.dtype)
        except TypeError as error:
            raise TypeError("dtype must be a numeric NumPy dtype") from error
        if not np.issubdtype(output_dtype, np.number):
            raise TypeError("dtype must be a numeric NumPy dtype")
        if self.handle_unknown not in (
            "error",
            "ignore",
            "infrequent_if_exist",
            "warn",
        ):
            raise ValueError(
                "handle_unknown must be 'error', 'ignore', "
                "'infrequent_if_exist', or 'warn'"
            )
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
        if self.feature_name_combiner != "concat" and not callable(
            self.feature_name_combiner
        ):
            raise TypeError("feature_name_combiner must be 'concat' or callable")
        return output_dtype

    def _fit_state(self, X: Any) -> tuple[CategoricalState, NDArray[np.int64]]:
        if isinstance(self.categories, str) and self.categories == "auto":
            return discover_categories(X, estimator=self)
        return state_from_categories(X, self.categories, estimator=self)

    def _build_effective_categories(self, encoded: NDArray[np.int64]) -> None:
        self._code_maps: list[NDArray[np.int64]] = []
        self._effective_categories: list[NDArray[Any]] = []
        infrequent_categories: list[NDArray[Any] | None] = []
        for feature, categories in enumerate(self.categories_):
            counts = np.bincount(encoded[:, feature], minlength=len(categories))
            infrequent = np.zeros(len(categories), dtype=bool)
            if self.min_frequency is not None:
                threshold = (
                    self.min_frequency
                    if isinstance(self.min_frequency, (int, np.integer))
                    else self.min_frequency * encoded.shape[0]
                )
                infrequent[counts < threshold] = True
            if self.max_categories is not None:
                frequent = np.flatnonzero(~infrequent)
                remove = max(0, frequent.size - max(self.max_categories - 1, 0))
                if remove:
                    order = np.lexsort((frequent, counts[frequent]))
                    infrequent[frequent[order[:remove]]] = True
            infrequent_indices = np.flatnonzero(infrequent)
            infrequent_categories.append(
                categories[infrequent_indices] if infrequent_indices.size else None
            )
            frequent_indices = np.flatnonzero(~infrequent)
            mapping = np.empty(len(categories), dtype=np.int64)
            mapping[frequent_indices] = np.arange(frequent_indices.size)
            effective = categories[frequent_indices].astype(object, copy=False).tolist()
            if infrequent_indices.size:
                mapping[infrequent_indices] = frequent_indices.size
                effective.append("infrequent_sklearn")
            self._code_maps.append(mapping)
            self._effective_categories.append(np.asarray(effective, dtype=object))
        frequency_grouping = (
            self.min_frequency is not None or self.max_categories is not None
        )
        if frequency_grouping:
            self.infrequent_categories_ = infrequent_categories
        elif hasattr(self, "infrequent_categories_"):
            del self.infrequent_categories_

    def _build_drop_indices(self) -> None:
        feature_count = len(self.categories_)
        original_drop = np.full(feature_count, -1, dtype=np.int64)
        effective_drop = np.full(feature_count, -1, dtype=np.int64)
        if self.drop is None:
            self.drop_idx_ = None
            self._drop_codes = effective_drop
            return
        if isinstance(self.drop, str):
            for feature, effective in enumerate(self._effective_categories):
                if self.drop == "first" or effective.size == 2:
                    effective_drop[feature] = 0
                    original_drop[feature] = int(
                        np.flatnonzero(self._code_maps[feature] == 0)[0]
                    )
        else:
            if len(self.drop) != feature_count:
                raise ValueError("drop must provide one category per input feature")
            for feature, value in enumerate(self.drop):
                matches = [
                    index
                    for index, category in enumerate(self.categories_[feature])
                    if _category_equal(category, value)
                ]
                if not matches:
                    raise ValueError(
                        f"drop category {value!r} was not found in feature {feature}"
                    )
                original_drop[feature] = matches[0]
                effective_drop[feature] = self._code_maps[feature][matches[0]]
        self.drop_idx_ = np.asarray(
            [None if index < 0 else index for index in original_drop], dtype=object
        )
        self._drop_codes = effective_drop

    def fit(self, X: Any, y: Any = None) -> OneHotEncoder:
        """Learn categories and output-column mappings."""
        del y
        self._output_dtype = self._validate_params()
        self._category_state, encoded = self._fit_state(X)
        self.categories_ = list(self._category_state.categories)
        self._build_effective_categories(encoded)
        self._build_drop_indices()
        self._widths = np.asarray(
            [categories.size for categories in self._effective_categories],
            dtype=np.int64,
        )
        self._n_features_outs = self._widths - (self._drop_codes >= 0)
        self._inverse_dtype = (
            np.dtype(object)
            if self.handle_unknown != "error"
            or any(
                category == "infrequent_sklearn"
                for categories in self._effective_categories
                for category in categories
            )
            else np.result_type(*(categories.dtype for categories in self.categories_))
        )
        return self

    def _map_known_codes(self, encoded: NDArray[np.int64]) -> NDArray[np.int64]:
        output = np.empty(encoded.shape, dtype=np.int64)
        for feature, mapping in enumerate(self._code_maps):
            output[:, feature] = mapping[encoded[:, feature]]
        return output

    def _effective_codes(self, X: Any) -> NDArray[np.int64]:
        encoded, known = encode_categories(X, self._category_state, estimator=self)
        unknown_features = np.flatnonzero(~np.all(known, axis=0))
        if unknown_features.size and self.handle_unknown == "error":
            raise ValueError(
                f"Found unknown categories in columns {unknown_features.tolist()} "
                "during transform"
            )
        if unknown_features.size and self.handle_unknown == "warn":
            warnings.warn(
                f"Found unknown categories in columns {unknown_features.tolist()} "
                "during transform. These unknown categories will be encoded as "
                "the infrequent category.",
                UserWarning,
                stacklevel=2,
            )
        elif (
            unknown_features.size
            and self.handle_unknown == "ignore"
            and self.drop is not None
        ):
            warnings.warn(
                f"Found unknown categories in columns {unknown_features.tolist()} "
                "during transform. These unknown categories will be encoded as "
                "all zeros",
                UserWarning,
                stacklevel=2,
            )
        output = np.full(encoded.shape, -1, dtype=np.int64)
        for feature, mapping in enumerate(self._code_maps):
            valid = known[:, feature]
            output[valid, feature] = mapping[encoded[valid, feature]]
            if self.handle_unknown in ("infrequent_if_exist", "warn"):
                infrequent = getattr(
                    self, "infrequent_categories_", [None] * len(self.categories_)
                )[feature]
                if infrequent is not None:
                    output[~valid, feature] = (
                        self._effective_categories[feature].size - 1
                    )
        return output

    def _encode_output(self, codes: NDArray[np.int64]) -> Any:
        indices, indptr = _core.one_hot_csr(
            np.ascontiguousarray(codes), self._widths, self._drop_codes
        )
        columns = int(np.sum(self._n_features_outs))
        output = sparse_from_components(
            SparseComponents(
                "csr",
                (codes.shape[0], columns),
                np.ones(indices.size, dtype=self._output_dtype),
                indices,
                indptr,
            ),
            canonicalize=False,
            validate=False,
        )
        return output if self.sparse_output else output.toarray()

    def transform(self, X: Any) -> Any:
        """Encode X into one-hot output."""
        self._output_dtype = self._validate_params()
        check_is_fitted(self, ("categories_", "_category_state", "_widths"))
        return self._encode_output(self._effective_codes(X))

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> Any:
        """Learn categories and encode X."""
        del y
        if fit_params:
            names = ", ".join(sorted(fit_params))
            raise TypeError(
                f"OneHotEncoder.fit_transform got unexpected arguments: {names}"
            )
        self._output_dtype = self._validate_params()
        self._category_state, encoded = self._fit_state(X)
        self.categories_ = list(self._category_state.categories)
        self._build_effective_categories(encoded)
        self._build_drop_indices()
        self._widths = np.asarray(
            [categories.size for categories in self._effective_categories],
            dtype=np.int64,
        )
        self._n_features_outs = self._widths - (self._drop_codes >= 0)
        self._inverse_dtype = (
            np.dtype(object)
            if self.handle_unknown != "error"
            or any(
                category == "infrequent_sklearn"
                for categories in self._effective_categories
                for category in categories
            )
            else np.result_type(*(categories.dtype for categories in self.categories_))
        )
        return self._encode_output(self._map_known_codes(encoded))

    def inverse_transform(self, X: Any) -> NDArray[Any]:
        """Convert one-hot output back to categorical values."""
        self._validate_params()
        check_is_fitted(
            self, ("categories_", "_effective_categories", "_n_features_outs")
        )
        values = check_array(
            X,
            accept_sparse=("csr", "csc"),
            dtype="numeric",
            ensure_all_finite=True,
        )
        if values.shape[1] != int(np.sum(self._n_features_outs)):
            raise ValueError(
                "encoded feature count does not match fitted OneHotEncoder"
            )
        dense = values.toarray() if hasattr(values, "toarray") else np.asarray(values)
        output = np.empty(
            (dense.shape[0], len(self.categories_)), dtype=self._inverse_dtype
        )
        start = 0
        for feature, (effective, width) in enumerate(
            zip(self._effective_categories, self._n_features_outs, strict=True)
        ):
            block = dense[:, start : start + width]
            start += width
            drop = self._drop_codes[feature]
            for row, encoded_row in enumerate(block):
                active = np.flatnonzero(encoded_row)
                if active.size > 1:
                    raise ValueError(
                        "one-hot rows may contain at most one active category"
                    )
                if active.size == 0:
                    if drop >= 0:
                        output[row, feature] = effective[drop]
                    elif self.handle_unknown in (
                        "ignore",
                        "infrequent_if_exist",
                        "warn",
                    ):
                        output[row, feature] = None
                    else:
                        raise ValueError(
                            f"Samples contain all zeros in encoded feature {feature}"
                        )
                    continue
                code = int(active[0])
                if drop >= 0 and code >= drop:
                    code += 1
                output[row, feature] = effective[code]
        return output

    def get_feature_names_out(self, input_features: Any = None) -> NDArray[Any]:
        """Return one generated feature name per output column."""
        check_is_fitted(self, ("categories_", "_effective_categories"))
        if input_features is None:
            names = (
                self.feature_names_in_
                if hasattr(self, "feature_names_in_")
                else np.asarray(
                    [f"x{i}" for i in range(self.n_features_in_)], dtype=object
                )
            )
        else:
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
        combiner = (
            (lambda feature, category: f"{feature}_{category}")
            if self.feature_name_combiner == "concat"
            else self.feature_name_combiner
        )
        output: list[str] = []
        for feature, (name, categories) in enumerate(
            zip(names, self._effective_categories, strict=True)
        ):
            for code, category in enumerate(categories):
                if code == self._drop_codes[feature]:
                    continue
                value = combiner(str(name), category)
                if not isinstance(value, str):
                    raise TypeError("feature_name_combiner must return strings")
                output.append(value)
        return np.asarray(output, dtype=object)
