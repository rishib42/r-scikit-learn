"""K-nearest-neighbors regression."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from rsklearn import _core
from rsklearn.base import BaseEstimator, RegressorMixin
from rsklearn.neighbors._base import (
    KNeighborsMixin,
    _NeighborNormCache,
    _NeighborTransposeCache,
)
from rsklearn.utils.validation import check_is_fitted, validate_data


class KNeighborsRegressor(KNeighborsMixin, RegressorMixin, BaseEstimator):
    """Regressor implementing dense brute-force k-nearest-neighbor averaging."""

    _rsklearn_target_tags = {
        "required": True,
        "one_d_labels": True,
        "two_d_labels": True,
        "multi_output": True,
        "single_output": True,
    }

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        weights: str = "uniform",
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: int = 2,
        metric: str = "minkowski",
        metric_params: dict[str, Any] | None = None,
        n_jobs: int | None = None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def _validate_params(self) -> None:
        self._validate_neighbor_params()

    def fit(self, X: Any, y: Any) -> KNeighborsRegressor:
        """Store the training set and numeric regression targets."""
        self._validate_params()
        if y is None:
            raise ValueError(
                "KNeighborsRegressor requires y to be passed, but the target y is None"
            )
        X_array = validate_data(
            self,
            X,
            reset=True,
            dtype=np.float64,
            order="C",
            ensure_all_finite=True,
        )
        try:
            y_array = np.asarray(y, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise TypeError("regression targets must be numeric") from error
        if y_array.ndim not in (1, 2) or y_array.shape[0] != X_array.shape[0]:
            raise ValueError(
                "y must be one- or two-dimensional with one target row per X sample"
            )
        if y_array.shape[0] == 0 or not np.isfinite(y_array).all():
            raise ValueError("regression targets must contain finite values")
        self._single_output = y_array.ndim == 1
        if y_array.ndim == 1:
            y_array = y_array.reshape(-1, 1)
        metric_name, metric_code = self._resolve_metric()
        self._fit_X = np.ascontiguousarray(X_array, dtype=np.float64)
        self._fit_X_transposed = (
            _NeighborTransposeCache()
            if metric_code == 0
            else np.empty((0, 0), dtype=np.float64)
        )
        self._y = np.ascontiguousarray(y_array, dtype=np.float64)
        self._fit_norms = (
            _NeighborNormCache()
            if metric_code == 0
            else np.asarray([], dtype=np.float64)
        )
        self.n_samples_fit_ = self._fit_X.shape[0]
        self.n_outputs_ = self._y.shape[1]
        self.effective_metric_ = metric_name
        self.effective_metric_params_ = (
            {} if self.metric_params is None else dict(self.metric_params)
        )
        self._metric_code = metric_code
        return self

    def kneighbors(
        self,
        X: Any = None,
        n_neighbors: int | None = None,
        return_distance: bool = True,
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]] | NDArray[np.int64]:
        """Return nearest-neighbor distances and indices."""
        training_query = X is None
        k = self._validate_neighbor_count(n_neighbors, training=training_query)
        query = self._fit_X if training_query else self._validate_X(X)
        distances, indices = _core.knn_kneighbors(
            query,
            self._fit_X,
            self._fit_transposed_array(),
            self._fit_norm_array(),
            k,
            self._metric_code,
            training_query,
        )
        if return_distance:
            return distances, indices
        return indices

    def predict(self, X: Any) -> NDArray[np.float64]:
        """Predict numeric targets for query samples."""
        check_is_fitted(self, ("_fit_X", "_fit_norms", "_y", "n_outputs_"))
        k = self._validate_neighbor_count(None, training=False)
        query = self._validate_X(X)
        output = _core.knn_predict_regression(
            query,
            self._fit_X,
            self._fit_transposed_array(),
            self._fit_norm_array(),
            self._y,
            k,
            self._metric_code,
            self._weights_code(),
        )
        return output[:, 0] if self._single_output else output
