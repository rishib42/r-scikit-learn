"""K-nearest-neighbors classification."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rsklearn import _core
from rsklearn._validation import validate_labels
from rsklearn.base import BaseEstimator, ClassifierMixin
from rsklearn.preprocessing import LabelEncoder
from rsklearn.utils.validation import check_is_fitted, validate_data

try:
    from sklearn.exceptions import DataConversionWarning
except ImportError:
    DataConversionWarning = UserWarning


class KNeighborsClassifier(ClassifierMixin, BaseEstimator):
    """Classifier implementing dense brute-force k-nearest-neighbor voting."""

    _rsklearn_target_tags = {"required": True}

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
        if (
            isinstance(self.n_neighbors, (bool, np.bool_))
            or not isinstance(self.n_neighbors, (int, np.integer))
            or self.n_neighbors <= 0
        ):
            raise ValueError("n_neighbors must be a positive integer")
        if self.weights not in ("uniform", "distance"):
            raise NotImplementedError(
                "KNeighborsClassifier currently supports weights='uniform' "
                "or weights='distance'"
            )
        if self.algorithm not in ("auto", "brute"):
            raise NotImplementedError(
                "KNeighborsClassifier currently supports algorithm='auto' or 'brute'"
            )
        if (
            isinstance(self.leaf_size, (bool, np.bool_))
            or not isinstance(self.leaf_size, (int, np.integer))
            or self.leaf_size <= 0
        ):
            raise ValueError("leaf_size must be a positive integer")
        if self.metric_params not in (None, {}):
            raise NotImplementedError("metric_params are not implemented")
        if self.n_jobs not in (None, 1):
            raise NotImplementedError(
                "n_jobs parallel execution is not implemented at the Python API level"
            )
        self._resolve_metric()

    def _resolve_metric(self) -> tuple[str, int]:
        if self.metric == "euclidean":
            if self.p not in (2, 2.0):
                raise ValueError("p is only used with metric='minkowski'")
            return "euclidean", 0
        if self.metric == "manhattan":
            if self.p not in (1, 1.0):
                raise ValueError("p is only used with metric='minkowski'")
            return "manhattan", 1
        if self.metric == "minkowski":
            if self.p in (2, 2.0):
                return "euclidean", 0
            if self.p in (1, 1.0):
                return "manhattan", 1
            raise NotImplementedError(
                "KNeighborsClassifier currently supports Minkowski p=1 or p=2"
            )
        raise NotImplementedError(
            "KNeighborsClassifier currently supports metric='minkowski', "
            "'euclidean', or 'manhattan'"
        )

    def _weights_code(self) -> int:
        return 0 if self.weights == "uniform" else 1

    def _validate_neighbor_count(
        self, n_neighbors: int | None, *, training: bool
    ) -> int:
        check_is_fitted(self, ("_fit_X", "_fit_norms", "_y_encoded", "classes_"))
        k = self.n_neighbors if n_neighbors is None else n_neighbors
        if (
            isinstance(k, (bool, np.bool_))
            or not isinstance(k, (int, np.integer))
            or k <= 0
        ):
            raise ValueError("n_neighbors must be a positive integer")
        maximum = self.n_samples_fit_ - int(training)
        if int(k) > maximum:
            raise ValueError(
                f"Expected n_neighbors <= n_samples_fit, but n_neighbors = {int(k)}, "
                f"n_samples_fit = {maximum}"
            )
        return int(k)

    def _validate_X(self, X: Any) -> NDArray[np.float64]:
        array = validate_data(
            self,
            X,
            reset=False,
            dtype=np.float64,
            order="C",
            ensure_all_finite=True,
        )
        return np.ascontiguousarray(array, dtype=np.float64)

    def fit(self, X: Any, y: Any) -> KNeighborsClassifier:
        """Store the training set and encoded target labels."""
        self._validate_params()
        if y is None:
            raise ValueError(
                "KNeighborsClassifier requires y to be passed, but the target y is None"
            )
        target = np.asarray(y)
        if target.ndim == 2 and target.shape[1] == 1:
            warnings.warn(
                "A column-vector y was passed when a 1d array was expected.",
                DataConversionWarning,
                stacklevel=2,
            )
            y = target.ravel()
        X_array, y_array = validate_data(
            self,
            X,
            y,
            reset=True,
            dtype=np.float64,
            order="C",
            ensure_all_finite=True,
        )
        if y_array.dtype.kind in "fc" and np.any(y_array != np.floor(y_array)):
            raise ValueError("Unknown label type: continuous")
        validate_labels(y_array)
        encoder = LabelEncoder()
        labels = encoder.fit_transform(y_array)
        self.classes_ = encoder.classes_
        if self.classes_.size < 2:
            raise ValueError(
                "KNeighborsClassifier requires at least two classes; got 1 class"
            )
        metric_name, metric_code = self._resolve_metric()
        self._fit_X = np.ascontiguousarray(X_array, dtype=np.float64)
        self._y_encoded = np.ascontiguousarray(labels, dtype=np.int64)
        self._fit_norms = (
            _core.knn_row_norms(self._fit_X)
            if metric_code == 0
            else np.asarray([], dtype=np.float64)
        )
        self.n_samples_fit_ = self._fit_X.shape[0]
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
            self._fit_norms,
            k,
            self._metric_code,
            training_query,
        )
        if return_distance:
            return distances, indices
        return indices

    def predict_proba(self, X: Any) -> NDArray[np.float64]:
        """Return class probabilities for query samples."""
        check_is_fitted(self, ("_fit_X", "_fit_norms", "_y_encoded", "classes_"))
        k = self._validate_neighbor_count(None, training=False)
        query = self._validate_X(X)
        return _core.knn_predict_proba(
            query,
            self._fit_X,
            self._fit_norms,
            self._y_encoded,
            k,
            self.classes_.size,
            self._metric_code,
            self._weights_code(),
        )

    def predict(self, X: Any) -> NDArray[Any]:
        """Predict class labels for query samples."""
        check_is_fitted(self, ("_fit_X", "_fit_norms", "_y_encoded", "classes_"))
        k = self._validate_neighbor_count(None, training=False)
        query = self._validate_X(X)
        indices = _core.knn_predict(
            query,
            self._fit_X,
            self._fit_norms,
            self._y_encoded,
            k,
            self.classes_.size,
            self._metric_code,
            self._weights_code(),
        )
        return self.classes_[indices]
