"""LabelEncoder public estimator."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from rsklearn import _core
from rsklearn._validation import validate_codes, validate_labels

from ._base import EstimatorMixin


def _unicode_codepoints(values: NDArray[Any]) -> NDArray[np.uint32]:
    """Expose a contiguous NumPy Unicode array as fixed-width codepoint rows."""
    width = values.dtype.itemsize // np.dtype(np.uint32).itemsize
    return values.view(np.uint32).reshape(values.size, width)


class LabelEncoder(EstimatorMixin):
    """Encode numeric, boolean, or Unicode labels as consecutive integers."""

    _parameter_names: tuple[str, ...] = ()

    def fit(self, y: Any) -> LabelEncoder:
        """Learn sorted unique classes and return self."""
        values, kind, input_dtype = validate_labels(y)
        self._input_dtype = input_dtype
        if kind == "signed":
            classes, _ = _core.label_fit_transform_i64(values)
            self.classes_ = classes.astype(self._input_dtype, copy=False)
        elif kind == "unsigned":
            classes, _ = _core.label_fit_transform_u64(values)
            self.classes_ = classes.astype(self._input_dtype, copy=False)
        elif kind == "float":
            self.classes_, _ = _core.label_fit_transform_numeric(values.tolist())
        elif kind == "bool":
            self.classes_ = np.unique(values)
        else:
            classes, _ = _core.label_fit_transform_unicode(_unicode_codepoints(values))
            self.classes_ = np.asarray(classes, dtype=values.dtype)
        self._label_kind = kind
        return self

    def transform(self, y: Any) -> NDArray[np.int64]:
        """Encode labels using fitted classes."""
        self._check_fitted("classes_", "_label_kind")
        values, kind, _ = validate_labels(y)
        if kind != self._label_kind:
            raise TypeError(f"LabelEncoder was fitted on {self._label_kind} labels")
        if kind == "signed":
            return _core.label_transform_i64(
                values, np.asarray(self.classes_, dtype=np.int64)
            )
        if kind == "unsigned":
            return _core.label_transform_u64(
                values, np.asarray(self.classes_, dtype=np.uint64)
            )
        if kind == "float":
            return _core.label_transform_numeric(
                values.tolist(), self.classes_.tolist()
            )
        if kind == "bool":
            indices = np.searchsorted(self.classes_, values)
            valid = indices < self.classes_.size
            if np.any(valid):
                valid[valid] &= self.classes_[indices[valid]] == values[valid]
            if not np.all(valid):
                raise ValueError(f"unknown label: {values[~valid][0]}")
            return indices.astype(np.int64, copy=False)
        return _core.label_transform_unicode(
            _unicode_codepoints(values), self.classes_.tolist()
        )

    def fit_transform(self, y: Any) -> NDArray[np.int64]:
        """Learn classes and encode labels."""
        values, kind, input_dtype = validate_labels(y)
        self._input_dtype = input_dtype
        if kind == "signed":
            classes, encoded = _core.label_fit_transform_i64(values)
            self.classes_ = classes.astype(self._input_dtype, copy=False)
        elif kind == "unsigned":
            classes, encoded = _core.label_fit_transform_u64(values)
            self.classes_ = classes.astype(self._input_dtype, copy=False)
        elif kind == "float":
            self.classes_, encoded = _core.label_fit_transform_numeric(values.tolist())
        elif kind == "bool":
            self.classes_, encoded = np.unique(values, return_inverse=True)
            encoded = encoded.astype(np.int64, copy=False)
        else:
            classes, encoded = _core.label_fit_transform_unicode(
                _unicode_codepoints(values)
            )
            self.classes_ = np.asarray(classes, dtype=values.dtype)
        self._label_kind = kind
        return encoded

    def inverse_transform(self, y: Any) -> NDArray[Any]:
        """Decode integer labels using fitted classes."""
        self._check_fitted("classes_", "_label_kind")
        codes = validate_codes(y)
        if self._label_kind == "signed":
            return _core.label_inverse_i64(
                codes, np.asarray(self.classes_, dtype=np.int64)
            ).astype(self.classes_.dtype, copy=False)
        if self._label_kind == "unsigned":
            return _core.label_inverse_u64(
                codes, np.asarray(self.classes_, dtype=np.uint64)
            ).astype(self.classes_.dtype, copy=False)
        if self._label_kind == "float":
            return _core.label_inverse_numeric(codes.tolist(), self.classes_.tolist())
        if self._label_kind == "bool":
            if np.any(codes < 0) or np.any(codes >= self.classes_.size):
                invalid = codes[(codes < 0) | (codes >= self.classes_.size)][0]
                raise ValueError(f"encoded label {invalid} is outside the valid range")
            return self.classes_[codes]
        return np.asarray(
            _core.label_inverse_strings(codes.tolist(), self.classes_.tolist()),
            dtype=self.classes_.dtype,
        )
