"""L1 and elastic-net regularized linear regression."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from rsklearn import _core
from rsklearn.base import BaseEstimator, RegressorMixin

from ._base import LinearModel, validate_regression_fit
from ._warnings import ConvergenceWarning


class _CoordinateDescent(RegressorMixin, LinearModel, BaseEstimator):
    _rsklearn_target_tags = {
        "required": True,
        "two_d_labels": True,
        "multi_output": True,
    }

    def _validate_common_params(self) -> None:
        if (
            isinstance(self.alpha, (bool, np.bool_))
            or not np.isscalar(self.alpha)
            or not np.isfinite(self.alpha)
            or self.alpha < 0
        ):
            raise ValueError("alpha must be a finite non-negative scalar")
        for name in ("fit_intercept", "copy_X", "warm_start", "positive"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise TypeError(f"{name} must be bool")
        if not isinstance(self.precompute, (bool, np.bool_)) or self.precompute:
            raise NotImplementedError("precompute is not implemented")
        if self.warm_start:
            raise NotImplementedError("warm_start is not implemented")
        if self.selection != "cyclic":
            raise NotImplementedError("only selection='cyclic' is implemented")
        if (
            isinstance(self.max_iter, (bool, np.bool_))
            or not isinstance(self.max_iter, (int, np.integer))
            or self.max_iter <= 0
        ):
            raise ValueError("max_iter must be a positive integer")
        if (
            isinstance(self.tol, (bool, np.bool_))
            or not np.isscalar(self.tol)
            or not np.isfinite(self.tol)
            or self.tol <= 0
        ):
            raise ValueError("tol must be a positive finite scalar")

    def _fit_coordinate(
        self, X: Any, y: Any, sample_weight: Any, l1_ratio: float
    ) -> _CoordinateDescent:
        self._validate_common_params()
        X_array, y_array, weights, self._single_output = validate_regression_fit(
            self, X, y, sample_weight
        )
        coefficients, intercepts, iterations, gaps, converged = (
            _core.linear_coordinate_fit_validated(
                X_array,
                y_array,
                weights,
                float(self.alpha),
                l1_ratio,
                self.fit_intercept,
                float(self.tol),
                int(self.max_iter),
                self.positive,
            )
        )
        self.coef_ = coefficients[0] if self._single_output else coefficients
        self.intercept_ = float(intercepts[0]) if self._single_output else intercepts
        self.n_iter_ = int(iterations)
        self.dual_gap_ = float(gaps[0]) if self._single_output else gaps
        if not converged:
            warnings.warn(
                f"{type(self).__name__} reached max_iter before convergence",
                ConvergenceWarning,
                stacklevel=2,
            )
        return self


class Lasso(_CoordinateDescent):
    """L1-regularized least squares using safe-Rust coordinate descent."""

    def __init__(
        self,
        alpha: float = 1.0,
        *,
        fit_intercept: bool = True,
        precompute: bool = False,
        copy_X: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        warm_start: bool = False,
        positive: bool = False,
        random_state: Any = None,
        selection: str = "cyclic",
    ) -> None:
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.random_state = random_state
        self.selection = selection

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Lasso:
        """Fit an L1-regularized linear model."""
        return self._fit_coordinate(X, y, sample_weight, 1.0)


class ElasticNet(_CoordinateDescent):
    """Elastic-net regression using safe-Rust coordinate descent."""

    def __init__(
        self,
        alpha: float = 1.0,
        *,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        precompute: bool = False,
        max_iter: int = 1000,
        copy_X: bool = True,
        tol: float = 1e-4,
        warm_start: bool = False,
        positive: bool = False,
        random_state: Any = None,
        selection: str = "cyclic",
    ) -> None:
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.random_state = random_state
        self.selection = selection

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> ElasticNet:
        """Fit an elastic-net regularized linear model."""
        if (
            isinstance(self.l1_ratio, (bool, np.bool_))
            or not np.isscalar(self.l1_ratio)
            or not np.isfinite(self.l1_ratio)
            or not 0 <= self.l1_ratio <= 1
        ):
            raise ValueError("l1_ratio must be a finite scalar in [0, 1]")
        return self._fit_coordinate(X, y, sample_weight, float(self.l1_ratio))
