"""Multinomial logistic regression."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rsklearn import _core
from rsklearn._validation import validate_labels
from rsklearn.base import BaseEstimator, ClassifierMixin
from rsklearn.metrics._validation import validate_sample_weight
from rsklearn.preprocessing import LabelEncoder
from rsklearn.utils.validation import validate_data

from ._base import raw_linear_prediction
from ._warnings import ConvergenceWarning

try:
    from sklearn.exceptions import DataConversionWarning
except ImportError:
    DataConversionWarning = UserWarning


class LogisticRegression(ClassifierMixin, BaseEstimator):
    """Regularized logistic regression backed by safe Rust."""

    _rsklearn_target_tags = {"required": True}

    def __init__(
        self,
        penalty: str | None = "l2",
        *,
        dual: bool = False,
        tol: float = 1e-4,
        C: float = 1.0,
        fit_intercept: bool = True,
        intercept_scaling: float = 1,
        class_weight: dict[Any, float] | str | None = None,
        random_state: Any = None,
        solver: str = "lbfgs",
        max_iter: int = 100,
        multi_class: str = "auto",
        verbose: int = 0,
        warm_start: bool = False,
        n_jobs: int | None = None,
        l1_ratio: float | None = None,
    ) -> None:
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio

    def _validate_params(self) -> None:
        if self.penalty not in ("l1", "l2", "elasticnet", None):
            raise ValueError("penalty must be 'l1', 'l2', 'elasticnet', or None")
        if self.dual:
            raise NotImplementedError("dual LogisticRegression is not implemented")
        if self.solver not in ("lbfgs", "rust", "saga"):
            raise NotImplementedError(
                "LogisticRegression currently supports solver='lbfgs', "
                "'rust', or 'saga'"
            )
        if self.penalty in ("l1", "elasticnet") and self.solver not in ("saga", "rust"):
            raise ValueError(
                "penalty='l1' and penalty='elasticnet' require solver='saga' or 'rust'"
            )
        if self.penalty == "elasticnet":
            if (
                isinstance(self.l1_ratio, (bool, np.bool_))
                or not np.isscalar(self.l1_ratio)
                or not np.isfinite(self.l1_ratio)
                or not 0 <= self.l1_ratio <= 1
            ):
                raise ValueError(
                    "l1_ratio must be a finite scalar in [0, 1] for elastic-net"
                )
        elif self.l1_ratio is not None:
            raise ValueError("l1_ratio is only used with elastic-net penalty")
        if self.multi_class not in ("auto", "multinomial"):
            raise NotImplementedError(
                "one-vs-rest LogisticRegression is not implemented"
            )
        if self.warm_start:
            raise NotImplementedError(
                "warm_start LogisticRegression is not implemented"
            )
        if self.n_jobs not in (None, 1):
            raise NotImplementedError("parallel LogisticRegression is not implemented")
        for name in ("dual", "fit_intercept", "warm_start"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise TypeError(f"{name} must be bool")
        if not np.isfinite(self.C) or self.C <= 0:
            raise ValueError("C must be a positive finite number")
        if not np.isfinite(self.tol) or self.tol <= 0:
            raise ValueError("tol must be a positive finite number")
        if (
            isinstance(self.max_iter, (bool, np.bool_))
            or not isinstance(self.max_iter, (int, np.integer))
            or self.max_iter <= 0
        ):
            raise ValueError("max_iter must be a positive integer")
        if (
            isinstance(self.verbose, (bool, np.bool_))
            or not isinstance(self.verbose, (int, np.integer))
            or self.verbose < 0
        ):
            raise ValueError("verbose must be a non-negative integer")

    def _class_weights(
        self, labels: NDArray[np.int64], sample_weight: Any
    ) -> NDArray[np.float64]:
        weights = validate_sample_weight(sample_weight, labels.size)
        if self.class_weight is None:
            return weights
        if self.class_weight == "balanced":
            counts = np.bincount(labels, minlength=self.classes_.size)
            factors = labels.size / (self.classes_.size * counts)
        elif isinstance(self.class_weight, dict):
            factors = np.asarray(
                [float(self.class_weight.get(value, 1.0)) for value in self.classes_],
                dtype=np.float64,
            )
        else:
            raise ValueError("class_weight must be None, 'balanced', or a dict")
        if not np.isfinite(factors).all() or np.any(factors < 0):
            raise ValueError("class_weight values must be finite and non-negative")
        return np.ascontiguousarray(weights * factors[labels])

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> LogisticRegression:
        """Fit a multinomial logistic classifier."""
        self._validate_params()
        if y is None:
            raise ValueError(
                "LogisticRegression requires y to be passed, but the target y is None"
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
                "LogisticRegression requires at least two classes; got 1 class"
            )
        weights = self._class_weights(labels, sample_weight)
        X_array = np.ascontiguousarray(X_array)
        labels = np.ascontiguousarray(labels, dtype=np.int64)
        if self.penalty in ("l1", "elasticnet"):
            if self.classes_.size != 2:
                raise NotImplementedError(
                    "L1 and elastic-net LogisticRegression currently support "
                    "binary classification only"
                )
            ratio = 1.0 if self.penalty == "l1" else float(self.l1_ratio)
            regularization = 1.0 / (float(self.C) * float(weights.sum()))
            coefficients, intercepts, iterations, converged = (
                _core.logistic_fit_proximal(
                    X_array,
                    labels,
                    weights,
                    regularization * ratio,
                    regularization * (1.0 - ratio),
                    self.fit_intercept,
                    float(self.tol),
                    int(self.max_iter),
                )
            )
        else:
            coefficients, intercepts, iterations, converged = _core.logistic_fit(
                X_array,
                labels,
                weights,
                self.classes_.size,
                0.0 if self.penalty is None else 1.0 / float(self.C),
                self.fit_intercept,
                float(self.tol),
                int(self.max_iter),
            )
        if self.classes_.size == 2:
            self.coef_ = (coefficients[1] - coefficients[0])[None, :]
            self.intercept_ = np.asarray([intercepts[1] - intercepts[0]])
        else:
            self.coef_ = coefficients
            self.intercept_ = intercepts
        self.n_iter_ = np.asarray([iterations], dtype=np.int32)
        if not converged:
            warnings.warn(
                "LogisticRegression reached max_iter before convergence",
                ConvergenceWarning,
                stacklevel=2,
            )
        return self

    def decision_function(self, X: Any) -> NDArray[np.float64]:
        """Return signed binary or per-class decision scores."""
        scores = raw_linear_prediction(self, X)
        return scores[:, 0] if self.classes_.size == 2 else scores

    def predict_proba(self, X: Any) -> NDArray[np.float64]:
        """Return class probabilities."""
        scores = self.decision_function(X)
        if scores.ndim == 1:
            positive = 1.0 / (1.0 + np.exp(-scores))
            return np.column_stack((1.0 - positive, positive))
        return np.asarray(_core.logistic_softmax(np.ascontiguousarray(scores)))

    def predict_log_proba(self, X: Any) -> NDArray[np.float64]:
        """Return logarithmic class probabilities."""
        return np.log(self.predict_proba(X))

    def predict(self, X: Any) -> NDArray[Any]:
        """Predict class labels."""
        scores = self.decision_function(X)
        indices = (
            (scores > 0).astype(np.intp)
            if scores.ndim == 1
            else np.argmax(scores, axis=1)
        )
        return self.classes_[indices]
