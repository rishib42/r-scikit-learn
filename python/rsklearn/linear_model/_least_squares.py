"""Ordinary and L2-regularized least-squares estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import linalg

from rsklearn import _core
from rsklearn.base import BaseEstimator, RegressorMixin

from ._base import LinearModel, validate_regression_fit

# Normal equations square the condition number. This cutoff limits the
# resulting float64 error amplification before selecting the fast Gram path.
_GRAM_MIN_SINGULAR_RATIO = np.finfo(np.float64).eps ** 0.25
_GRAM_RANK_RESOLUTION = np.sqrt(np.finfo(np.float64).eps)


def _tall_solution_is_stable(singular: np.ndarray, rank: int, tolerance: float) -> bool:
    """Return whether normal-equation accuracy is reliable for this spectrum."""
    if rank == 0 or singular.size == 0 or not np.isfinite(singular).all():
        return False
    if rank < singular.size and tolerance < _GRAM_RANK_RESOLUTION:
        return False
    largest = singular[0]
    smallest_retained = singular[rank - 1]
    return (
        largest > 0
        and smallest_retained > 0
        and smallest_retained / largest >= _GRAM_MIN_SINGULAR_RATIO
    )


def _fit_lstsq(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    fit_intercept: bool,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Solve unregularized least squares through a shape-aware dense backend."""
    if X.shape[0] >= 4 * X.shape[1]:
        tall_fit = _core.linear_fit_tall(X, y, weights, fit_intercept, tolerance)
        if _tall_solution_is_stable(tall_fit[3], tall_fit[2], tolerance):
            return tall_fit
    uniform_weights = np.all(weights == weights[0])
    if fit_intercept:
        if uniform_weights:
            x_mean = X.mean(axis=0)
            y_mean = y.mean(axis=0)
        else:
            x_mean = np.average(X, axis=0, weights=weights)
            y_mean = np.average(y, axis=0, weights=weights)
    else:
        x_mean = np.zeros(X.shape[1], dtype=np.float64)
        y_mean = np.zeros(y.shape[1], dtype=np.float64)
    centered_X = X.copy()
    centered_y = y.copy()
    centered_X -= x_mean
    centered_y -= y_mean
    if not uniform_weights:
        root_weights = np.sqrt(weights)[:, None]
        centered_X *= root_weights
        centered_y *= root_weights
    coefficients, _, rank, singular = linalg.lstsq(
        centered_X,
        centered_y,
        cond=tolerance,
        check_finite=False,
        lapack_driver="gelsd",
    )
    coefficients = np.asarray(coefficients.T, dtype=np.float64)
    intercepts = y_mean - coefficients @ x_mean
    return coefficients, np.asarray(intercepts, dtype=np.float64), rank, singular


class LinearRegression(RegressorMixin, LinearModel, BaseEstimator):
    """Ordinary least squares using Rust tall-matrix and LAPACK fallback solvers."""

    _rsklearn_target_tags = {
        "required": True,
        "two_d_labels": True,
        "multi_output": True,
    }

    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        copy_X: bool = True,
        tol: float = 1e-6,
        n_jobs: int | None = None,
        positive: bool = False,
    ) -> None:
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.tol = tol
        self.n_jobs = n_jobs
        self.positive = positive

    def _validate_params(self) -> None:
        for name in ("fit_intercept", "copy_X", "positive"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise TypeError(f"{name} must be bool")
        if not isinstance(self.tol, (int, float, np.number)) or self.tol < 0:
            raise ValueError("tol must be a non-negative number")
        if self.n_jobs not in (None, 1):
            raise NotImplementedError("parallel LinearRegression is not implemented")
        if self.positive:
            raise NotImplementedError(
                "positive-constrained LinearRegression is not implemented"
            )

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> LinearRegression:
        """Fit an ordinary least-squares model."""
        self._validate_params()
        X_array, y_array, weights, self._single_output = validate_regression_fit(
            self, X, y, sample_weight
        )
        coefficients, intercepts, self.rank_, self.singular_ = _fit_lstsq(
            X_array, y_array, weights, self.fit_intercept, float(self.tol)
        )
        self.coef_ = coefficients[0] if self._single_output else coefficients
        self.intercept_ = float(intercepts[0]) if self._single_output else intercepts
        return self


class Ridge(RegressorMixin, LinearModel, BaseEstimator):
    """L2-regularized least squares using a safe-Rust SVD solver."""

    _rsklearn_target_tags = {
        "required": True,
        "two_d_labels": True,
        "multi_output": True,
    }

    def __init__(
        self,
        alpha: float = 1.0,
        *,
        fit_intercept: bool = True,
        copy_X: bool = True,
        max_iter: int | None = None,
        tol: float = 1e-4,
        solver: str = "auto",
        positive: bool = False,
        random_state: Any = None,
    ) -> None:
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.positive = positive
        self.random_state = random_state

    def _validate_params(self) -> None:
        if (
            isinstance(self.alpha, (bool, np.bool_))
            or not np.isscalar(self.alpha)
            or not np.isfinite(self.alpha)
            or self.alpha < 0
        ):
            raise ValueError("alpha must be a finite non-negative scalar")
        for name in ("fit_intercept", "copy_X", "positive"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise TypeError(f"{name} must be bool")
        if self.solver not in ("auto", "svd"):
            raise NotImplementedError("Ridge currently supports solver='auto' or 'svd'")
        if self.positive:
            raise NotImplementedError("positive-constrained Ridge is not implemented")
        if self.max_iter is not None and (
            isinstance(self.max_iter, (bool, np.bool_))
            or not isinstance(self.max_iter, (int, np.integer))
            or self.max_iter <= 0
        ):
            raise ValueError("max_iter must be a positive integer or None")
        if not isinstance(self.tol, (int, float, np.number)) or self.tol < 0:
            raise ValueError("tol must be a non-negative number")

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Ridge:
        """Fit an L2-regularized least-squares model."""
        self._validate_params()
        X_array, y_array, weights, self._single_output = validate_regression_fit(
            self, X, y, sample_weight
        )
        coefficients, intercepts = _core.linear_fit(
            X_array, y_array, weights, float(self.alpha), self.fit_intercept
        )
        self.coef_ = coefficients[0] if self._single_output else coefficients
        self.intercept_ = float(intercepts[0]) if self._single_output else intercepts
        self.n_iter_ = np.asarray([1], dtype=np.int32)
        return self
