"""Shared nearest-neighbor estimator infrastructure."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from rsklearn.utils.validation import check_is_fitted, validate_data


class _NeighborNormCache:
    """Mutable holder so lazy norm computation does not change estimator __dict__."""

    def __init__(self) -> None:
        self.values: NDArray[np.float64] | None = None


class _NeighborTransposeCache:
    """Mutable holder for the lazily materialized transposed fit matrix."""

    def __init__(self) -> None:
        self.values: NDArray[np.float64] | None = None


class KNeighborsMixin:
    """Shared validation and neighbor-query logic for dense KNN estimators."""

    def _validate_neighbor_params(self) -> None:
        if (
            isinstance(self.n_neighbors, (bool, np.bool_))
            or not isinstance(self.n_neighbors, (int, np.integer))
            or self.n_neighbors <= 0
        ):
            raise ValueError("n_neighbors must be a positive integer")
        if self.weights not in ("uniform", "distance"):
            raise NotImplementedError(
                f"{type(self).__name__} currently supports weights='uniform' "
                "or weights='distance'"
            )
        if self.algorithm not in ("auto", "brute"):
            raise NotImplementedError(
                f"{type(self).__name__} currently supports algorithm='auto' or 'brute'"
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
                f"{type(self).__name__} currently supports Minkowski p=1 or p=2"
            )
        raise NotImplementedError(
            f"{type(self).__name__} currently supports metric='minkowski', "
            "'euclidean', or 'manhattan'"
        )

    def _weights_code(self) -> int:
        return 0 if self.weights == "uniform" else 1

    def _fit_norm_array(self) -> NDArray[np.float64]:
        if self._metric_code != 0:
            return self._fit_norms
        if self._fit_norms.values is None:
            from rsklearn import _core

            self._fit_norms.values = _core.knn_row_norms(self._fit_X)
        return self._fit_norms.values

    def _fit_transposed_array(self) -> NDArray[np.float64]:
        if self._metric_code != 0:
            return self._fit_X_transposed
        if self._fit_X_transposed.values is None:
            self._fit_X_transposed.values = np.ascontiguousarray(
                self._fit_X.T, dtype=np.float64
            )
        return self._fit_X_transposed.values

    def _validate_neighbor_count(
        self, n_neighbors: int | None, *, training: bool
    ) -> int:
        check_is_fitted(self, ("_fit_X", "_fit_norms"))
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
