"""Normalizer public estimator."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from rsklearn import _core
from rsklearn._validation import one_to_one_feature_names, validate_normalizer_2d
from rsklearn.base import BaseEstimator, TransformerMixin
from rsklearn.utils.validation import check_is_fitted


class Normalizer(TransformerMixin, BaseEstimator):
    """Normalize each sample independently to unit norm.

    Dense float32 input is processed by a native float32 Rust kernel. Other
    numeric input is processed as float64. Sparse input is not yet supported.
    """

    _rsklearn_preserves_dtype = ["float64", "float32"]

    def __init__(self, norm: str = "l2", *, copy: bool = True) -> None:
        self.norm = norm
        self.copy = copy

    def _validate_params(self) -> None:
        if self.norm not in ("l1", "l2", "max"):
            raise ValueError("norm must be 'l1', 'l2', or 'max'")
        if not isinstance(self.copy, bool):
            raise TypeError("copy must be bool")

    def fit(self, X: Any, y: Any = None) -> Normalizer:
        """Validate X, record feature metadata, and return self."""
        del y
        self._validate_params()
        validate_normalizer_2d(X, estimator=self, reset=True, copy=False)
        return self

    def _transform_validated(
        self, array: NDArray[np.float32] | NDArray[np.float64]
    ) -> NDArray[np.float32] | NDArray[np.float64]:
        if array.dtype == np.dtype(np.float32):
            output = _core.normalize_f32(array, self.norm)
        else:
            output = _core.normalize_f64(array, self.norm)
        if not self.copy and array.flags.writeable:
            array[...] = output
            return array
        return output

    def transform(self, X: Any) -> NDArray[np.float32] | NDArray[np.float64]:
        """Normalize each row of X using the configured norm."""
        self._validate_params()
        check_is_fitted(self, "n_features_in_")
        array = validate_normalizer_2d(X, estimator=self, reset=False, copy=False)
        return self._transform_validated(array)

    def fit_transform(
        self, X: Any, y: Any = None, **fit_params: Any
    ) -> NDArray[np.float32] | NDArray[np.float64]:
        """Validate and normalize X in a single public API pass."""
        del y
        if fit_params:
            names = ", ".join(sorted(fit_params))
            raise TypeError(
                f"Normalizer.fit_transform got unexpected arguments: {names}"
            )
        self._validate_params()
        array = validate_normalizer_2d(X, estimator=self, reset=True, copy=False)
        return self._transform_validated(array)

    def get_feature_names_out(self, input_features: Any = None) -> NDArray[Any]:
        """Return unchanged output feature names."""
        return one_to_one_feature_names(self, input_features)
