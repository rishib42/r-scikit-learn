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
    """Encode homogeneous numeric or Unicode labels as consecutive integers."""

    _parameter_names: tuple[str, ...] = ()

    def fit(self, y: Any) -> LabelEncoder:
        """Learn sorted unique classes and return self."""
        values, kind = validate_labels(y)
        if kind == "numeric":
            self.classes_, _ = _core.label_fit_transform_numeric(values.tolist())
        else:
            classes, _ = _core.label_fit_transform_unicode(_unicode_codepoints(values))
            self.classes_ = np.asarray(classes, dtype=values.dtype)
        self._label_kind = kind
        return self

    def transform(self, y: Any) -> NDArray[np.int64]:
        """Encode labels using fitted classes."""
        self._check_fitted("classes_", "_label_kind")
        values, kind = validate_labels(y)
        if kind != self._label_kind:
            raise TypeError(f"LabelEncoder was fitted on {self._label_kind} labels")
        if kind == "numeric":
            return _core.label_transform_numeric(
                values.tolist(), self.classes_.tolist()
            )
        return _core.label_transform_unicode(
            _unicode_codepoints(values), self.classes_.tolist()
        )

    def fit_transform(self, y: Any) -> NDArray[np.int64]:
        """Learn classes and encode labels."""
        values, kind = validate_labels(y)
        if kind == "numeric":
            self.classes_, encoded = _core.label_fit_transform_numeric(values.tolist())
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
        if self._label_kind == "numeric":
            return _core.label_inverse_numeric(codes.tolist(), self.classes_.tolist())
        return np.asarray(
            _core.label_inverse_strings(codes.tolist(), self.classes_.tolist()),
            dtype=self.classes_.dtype,
        )
