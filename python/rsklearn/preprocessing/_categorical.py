"""Shared dense categorical discovery and lookup infrastructure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rsklearn import _core
from rsklearn.utils.validation import validate_data


@dataclass(frozen=True)
class CategoricalState:
    """Learned per-feature categories and their internal lookup pathways."""

    categories: tuple[NDArray[Any], ...]
    kinds: tuple[str, ...]


def _is_nan(value: Any) -> bool:
    return isinstance(value, (float, np.floating)) and np.isnan(value)


def _object_column_kind(values: list[Any]) -> str:
    has_nan = any(_is_nan(value) for value in values)
    non_missing = [value for value in values if not _is_nan(value)]
    types = {type(value).__name__ for value in non_missing}
    if all(isinstance(value, (bool, np.bool_)) for value in non_missing):
        return "float" if has_nan else "bool"
    if all(
        isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
        for value in non_missing
    ):
        if has_nan:
            return "float"
        minimum = min(non_missing, default=0)
        maximum = max(non_missing, default=0)
        if minimum < np.iinfo(np.int64).min or maximum > np.iinfo(np.uint64).max:
            raise ValueError("categorical integer values must fit in int64 or uint64")
        return (
            "unsigned"
            if minimum >= 0 and maximum > np.iinfo(np.int64).max
            else "signed"
        )
    if all(
        isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating))
        for value in non_missing
    ):
        return "float"
    if all(isinstance(value, (str, type(None))) for value in non_missing):
        return "object_string"
    raise TypeError(
        "Encoders require each feature to contain uniformly strings or numbers. "
        f"Got {sorted(types)}"
    )


def _column_kind(column: NDArray[Any]) -> str:
    if np.issubdtype(column.dtype, np.bool_):
        return "bool"
    if np.issubdtype(column.dtype, np.signedinteger):
        return "signed"
    if np.issubdtype(column.dtype, np.unsignedinteger):
        return "unsigned"
    if np.issubdtype(column.dtype, np.floating):
        return "float"
    if np.issubdtype(column.dtype, np.str_):
        return "string"
    if column.dtype == object:
        return _object_column_kind(column.tolist())
    raise TypeError(f"unsupported categorical dtype: {column.dtype}")


def _coerce_column(column: NDArray[Any], kind: str) -> NDArray[Any]:
    dtypes = {
        "bool": np.bool_,
        "signed": np.int64,
        "unsigned": np.uint64,
        "float": np.float64,
        "string": str,
        "object_string": object,
    }
    return np.ascontiguousarray(column, dtype=dtypes[kind])


def _unicode_codepoints(values: NDArray[Any]) -> NDArray[np.uint32]:
    width = values.dtype.itemsize // np.dtype(np.uint32).itemsize
    return values.view(np.uint32).reshape(*values.shape, width)


def _discover_homogeneous_matrix(
    array: NDArray[Any],
) -> tuple[tuple[NDArray[Any], ...], tuple[str, ...], NDArray[np.int64]] | None:
    kind = _column_kind(array[:, 0])
    if kind == "signed":
        raw_categories, encoded = _core.category_discover_matrix_i64(
            np.ascontiguousarray(array, dtype=np.int64)
        )
    elif kind == "unsigned":
        raw_categories, encoded = _core.category_discover_matrix_u64(
            np.ascontiguousarray(array, dtype=np.uint64)
        )
    elif kind == "float":
        raw_categories, encoded = _core.category_discover_matrix_f64(
            np.ascontiguousarray(array, dtype=np.float64)
        )
    elif kind == "bool":
        raw_categories, encoded = _core.category_discover_matrix_bool(
            np.ascontiguousarray(array, dtype=np.bool_)
        )
    elif kind == "string":
        values = np.ascontiguousarray(array, dtype=str)
        raw_categories, encoded = _core.category_discover_matrix_unicode(
            _unicode_codepoints(values)
        )
    else:
        return None
    categories = tuple(
        np.asarray(feature_categories, dtype=array.dtype)
        for feature_categories in raw_categories
    )
    return categories, (kind,) * array.shape[1], encoded


def _discover_column(
    column: NDArray[Any], kind: str
) -> tuple[NDArray[Any], NDArray[np.int64]]:
    values = _coerce_column(column, kind)
    if kind == "float" and np.isinf(values).any():
        raise ValueError("categorical input contains infinity")
    if kind == "bool":
        return _core.category_discover_bool(values)
    if kind == "signed":
        categories, encoded = _core.category_discover_i64(values)
        return categories.astype(values.dtype, copy=False), encoded
    if kind == "unsigned":
        categories, encoded = _core.category_discover_u64(values)
        return categories.astype(values.dtype, copy=False), encoded
    if kind == "float":
        categories, encoded = _core.category_discover_f64(values)
        return categories.astype(values.dtype, copy=False), encoded
    if kind == "string":
        categories, encoded = _core.category_discover_unicode(
            _unicode_codepoints(values)
        )
        return np.asarray(categories, dtype=column.dtype), encoded
    raw_values = values.tolist()
    strings = sorted({value for value in raw_values if isinstance(value, str)})
    category_values: list[Any] = list(strings)
    if any(value is None for value in raw_values):
        category_values.append(None)
    if any(_is_nan(value) for value in raw_values):
        category_values.append(np.nan)
    categories = np.asarray(category_values, dtype=object)
    mapping = {
        value: index for index, value in enumerate(categories) if not _is_nan(value)
    }
    nan_index = (
        len(categories) - 1 if categories.size and _is_nan(categories[-1]) else -1
    )
    encoded = np.asarray(
        [nan_index if _is_nan(value) else mapping[value] for value in values],
        dtype=np.int64,
    )
    return categories, encoded


def _encode_column(
    column: NDArray[Any], kind: str, categories: NDArray[Any]
) -> NDArray[np.int64]:
    values = _coerce_column(column, kind)
    if kind == "float" and np.isinf(values).any():
        raise ValueError("categorical input contains infinity")
    if kind == "bool":
        return _core.category_encode_bool(
            values, categories.astype(np.bool_, copy=False)
        )
    if kind == "signed":
        return _core.category_encode_i64(
            values, categories.astype(np.int64, copy=False)
        )
    if kind == "unsigned":
        return _core.category_encode_u64(
            values, categories.astype(np.uint64, copy=False)
        )
    if kind == "float":
        return _core.category_encode_f64(
            values, categories.astype(np.float64, copy=False)
        )
    if kind == "string":
        return _core.category_encode_unicode(
            _unicode_codepoints(values), categories.tolist()
        )
    mapping = {
        value: index for index, value in enumerate(categories) if not _is_nan(value)
    }
    nan_index = (
        len(categories) - 1 if categories.size and _is_nan(categories[-1]) else -1
    )
    return np.asarray(
        [nan_index if _is_nan(value) else mapping.get(value, -1) for value in values],
        dtype=np.int64,
    )


def validate_categorical_input(X: Any, *, estimator: Any, reset: bool) -> NDArray[Any]:
    """Validate dense two-dimensional categorical input and feature metadata."""
    values = np.asarray(X, dtype=object) if isinstance(X, (list, tuple)) else X
    array = validate_data(
        estimator,
        values,
        reset=reset,
        dtype=None,
        ensure_all_finite="allow-nan",
    )
    if np.iscomplexobj(array) or (
        array.dtype == object
        and any(
            isinstance(value, (complex, np.complexfloating)) for value in array.flat
        )
    ):
        raise ValueError("Complex data not supported")
    return np.asarray(array)


def discover_categories(
    X: Any, *, estimator: Any
) -> tuple[CategoricalState, NDArray[np.int64]]:
    """Discover sorted categories independently for every feature."""
    array = validate_categorical_input(X, estimator=estimator, reset=True)
    if array.dtype != object:
        homogeneous = _discover_homogeneous_matrix(array)
        if homogeneous is not None:
            categories, kinds, encoded = homogeneous
            return CategoricalState(categories, kinds), encoded
    categories: list[NDArray[Any]] = []
    kinds: list[str] = []
    encoded = np.empty(array.shape, dtype=np.int64)
    for index in range(array.shape[1]):
        column = array[:, index]
        kind = _column_kind(column)
        feature_categories, encoded[:, index] = _discover_column(column, kind)
        categories.append(feature_categories)
        kinds.append(kind)
    return CategoricalState(tuple(categories), tuple(kinds)), encoded


def encode_categories(
    X: Any, state: CategoricalState, *, estimator: Any
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """Encode using learned categories and return codes plus a known-value mask."""
    array = validate_categorical_input(X, estimator=estimator, reset=False)
    if array.shape[1] != len(state.categories):
        raise ValueError("categorical state feature count does not match input")
    encoded = np.empty(array.shape, dtype=np.int64)
    for index, (kind, categories) in enumerate(
        zip(state.kinds, state.categories, strict=True)
    ):
        actual_kind = _column_kind(array[:, index])
        if actual_kind != kind:
            raise TypeError(
                f"feature {index} was fitted on {kind} values, got {actual_kind}"
            )
        encoded[:, index] = _encode_column(array[:, index], kind, categories)
    return encoded, encoded >= 0


def state_from_categories(
    X: Any, categories: Any, *, estimator: Any
) -> tuple[CategoricalState, NDArray[np.int64]]:
    """Validate explicit per-feature categories and encode fitting input."""
    array = validate_categorical_input(X, estimator=estimator, reset=True)
    if not isinstance(categories, (list, tuple)):
        raise TypeError("categories must be 'auto' or a list of array-like values")
    if len(categories) != array.shape[1]:
        raise ValueError(
            "categories must provide exactly one category array per input feature"
        )
    learned: list[NDArray[Any]] = []
    kinds: list[str] = []
    for index, supplied in enumerate(categories):
        feature_categories = np.asarray(supplied)
        if feature_categories.ndim != 1 or feature_categories.size == 0:
            raise ValueError(f"categories[{index}] must be a non-empty 1D array")
        kind = _column_kind(array[:, index])
        supplied_kind = _column_kind(feature_categories)
        numeric_kinds = {"bool", "signed", "unsigned", "float"}
        compatible = (
            kind == supplied_kind
            or {kind, supplied_kind} <= {"string", "object_string"}
            or {kind, supplied_kind} <= numeric_kinds
        )
        if not compatible:
            raise TypeError(
                f"categories[{index}] has {supplied_kind} values, but feature "
                f"{index} contains {kind} values"
            )
        values = _coerce_column(feature_categories, kind)
        if kind == "float" and np.isinf(values).any():
            raise ValueError(f"categories[{index}] contains infinity")
        if kind in {"bool", "signed", "unsigned", "float"}:
            discovered, _ = _discover_column(values, kind)
            if discovered.size != values.size or not np.array_equal(
                discovered, values, equal_nan=True
            ):
                raise ValueError(
                    f"categories[{index}] must contain unique values sorted in "
                    "ascending order, with NaN last"
                )
        else:
            _, codes = _discover_column(values, kind)
            if np.unique(codes).size != values.size:
                raise ValueError(f"categories[{index}] contains duplicate values")
        learned.append(values)
        kinds.append(kind)
    state = CategoricalState(tuple(learned), tuple(kinds))
    encoded, known = encode_categories(array, state, estimator=estimator)
    if not np.all(known):
        index = int(np.flatnonzero(~known)[0])
        row, column = np.unravel_index(index, known.shape)
        raise ValueError(
            f"Found unknown category {array[row, column]!r} in column "
            f"{column} during fit"
        )
    return state, encoded


__all__ = [
    "CategoricalState",
    "discover_categories",
    "encode_categories",
    "state_from_categories",
    "validate_categorical_input",
]
