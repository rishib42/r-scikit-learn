"""MaxAbsScaler public estimator."""

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
from rsklearn.utils import scale_sparse_columns, sparse_max_abs_checked
from rsklearn.utils.validation import check_is_fitted, validate_data


class MaxAbsScaler(TransformerMixin, BaseEstimator):
    """Scale each feature by its maximum absolute value.

    Dense and CSR/CSC sparse inputs are supported. Sparse inputs are never
    densified; stored values are scaled in Rust while implicit zeros remain
    implicit.
    """

    _parameter_names: tuple[str, ...] = ()
    _rsklearn_input_tags = {"allow_nan": True, "sparse": True}
    _rsklearn_preserves_dtype = ["float64", "float32"]

    def fit(self, X: Any, y: Any = None) -> MaxAbsScaler:
        """Learn per-feature maximum absolute values and return self."""
        del y
        for attribute in (
            "scale_",
            "max_abs_",
            "n_features_in_",
            "n_samples_seen_",
        ):
            if hasattr(self, attribute):
                delattr(self, attribute)
        return self.partial_fit(X)

    def partial_fit(self, X: Any, y: Any = None) -> MaxAbsScaler:
        """Update per-feature maximum absolute values from a batch."""
        del y
        already_fitted = hasattr(self, "n_features_in_")
        if _is_sparse(X):
            matrix = validate_data(
                self,
                X,
                reset=not already_fitted,
                accept_sparse=("csr", "csc"),
                dtype=np.float64,
                ensure_all_finite="allow-nan",
            )
            batch_max_abs = sparse_max_abs_checked(matrix)
            batch_rows = matrix.shape[0]
        else:
            array = validate_numeric_2d(X, estimator=self, reset=not already_fitted)
            batch_max_abs = _core.maxabs_fit(array)
            batch_rows = array.shape[0]
        if already_fitted:
            self.max_abs_ = np.fmax(self.max_abs_, batch_max_abs)
            self.n_samples_seen_ += batch_rows
        else:
            self.max_abs_ = batch_max_abs
            self.n_samples_seen_ = batch_rows
        self.scale_ = _handle_zeros_in_scale(self.max_abs_)
        return self

    def transform(self, X: Any) -> Any:
        """Scale X using fitted maximum absolute values."""
        check_is_fitted(self, ("n_features_in_", "scale_"))
        if _is_sparse(X):
            matrix = validate_data(
                self,
                X,
                reset=False,
                accept_sparse=("csr", "csc"),
                dtype="numeric",
                ensure_all_finite="allow-nan",
            )
            return scale_sparse_columns(matrix, self.scale_, copy=True)
        array, output_dtype = validate_numeric_2d_with_dtype(
            X, estimator=self, reset=False
        )
        output = (
            _core.maxabs_transform_f32(
                array.astype(np.float32, copy=False), self.scale_
            )
            if output_dtype == np.dtype(np.float32)
            else _core.maxabs_transform_f64(array, self.scale_)
        )
        return output.astype(output_dtype, copy=False)

    def inverse_transform(self, X: Any) -> Any:
        """Undo max-abs scaling."""
        check_is_fitted(self, ("n_features_in_", "scale_"))
        if _is_sparse(X):
            matrix = validate_data(
                self,
                X,
                reset=False,
                accept_sparse=("csr", "csc"),
                dtype="numeric",
                ensure_all_finite="allow-nan",
            )
            return scale_sparse_columns(matrix, self.scale_, inverse=True, copy=True)
        array, output_dtype = validate_numeric_2d_with_dtype(
            X, estimator=self, reset=False
        )
        output = (
            _core.maxabs_transform_f32(
                array.astype(np.float32, copy=False), self.scale_, inverse=True
            )
            if output_dtype == np.dtype(np.float32)
            else _core.maxabs_transform_f64(array, self.scale_, inverse=True)
        )
        return output.astype(output_dtype, copy=False)

    def get_feature_names_out(self, input_features: Any = None) -> NDArray[Any]:
        """Return unchanged output feature names."""
        return one_to_one_feature_names(self, input_features)


def _handle_zeros_in_scale(scale: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.where(scale == 0.0, 1.0, scale)


def _is_sparse(value: Any) -> bool:
    try:
        from scipy import sparse
    except ImportError:
        return False
    return bool(sparse.issparse(value))
