"""Cross-validation scoring utilities."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rsklearn.base import clone
from rsklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from ._split import KFold, StratifiedKFold
from ._utils import check_consistent_length, is_arraylike, safe_indexing


class FitFailedWarning(RuntimeWarning):
    """Warning used when fitting an estimator on a fold fails."""


def _is_classifier(estimator: Any) -> bool:
    if getattr(estimator, "_estimator_type", None) == "classifier":
        return True
    final = getattr(estimator, "_final_estimator", None)
    return getattr(final, "_estimator_type", None) == "classifier"


def _metric_scorer(
    metric: Callable[..., float], *, negate: bool = False, **kwargs: Any
) -> Callable[..., float]:
    def scorer(estimator: Any, X: Any, y: Any) -> float:
        value = float(metric(y, estimator.predict(X), **kwargs))
        return -value if negate else value

    return scorer


_SCORERS: dict[str, Callable[..., float]] = {
    "accuracy": _metric_scorer(accuracy_score),
    "precision": _metric_scorer(precision_score, zero_division=0),
    "precision_macro": _metric_scorer(
        precision_score, average="macro", zero_division=0
    ),
    "recall": _metric_scorer(recall_score, zero_division=0),
    "recall_macro": _metric_scorer(recall_score, average="macro", zero_division=0),
    "f1": _metric_scorer(f1_score, zero_division=0),
    "f1_macro": _metric_scorer(f1_score, average="macro", zero_division=0),
    "neg_mean_squared_error": _metric_scorer(mean_squared_error, negate=True),
    "neg_mean_absolute_error": _metric_scorer(mean_absolute_error, negate=True),
    "r2": _metric_scorer(r2_score),
}


def _resolve_scorer(estimator: Any, scoring: Any) -> Callable[..., float]:
    if scoring is None:
        if not hasattr(estimator, "score"):
            raise TypeError("estimator must implement score when scoring=None")
        return lambda fitted, X, y: float(
            fitted.score(X) if y is None else fitted.score(X, y)
        )
    if isinstance(scoring, str):
        try:
            return _SCORERS[scoring]
        except KeyError as error:
            raise ValueError(
                f"Unknown scoring value {scoring!r}. Supported values are: "
                f"{', '.join(sorted(_SCORERS))}"
            ) from error
    if callable(scoring):
        return scoring
    raise TypeError("scoring must be None, a supported string, or a callable")


def _resolve_cv(cv: Any, estimator: Any, y: Any) -> Any:
    if cv is None:
        cv = 5
    if isinstance(cv, (bool, np.bool_)):
        raise TypeError("cv must be an integer, splitter, or iterable")
    if isinstance(cv, (int, np.integer)):
        splitter = (
            StratifiedKFold(int(cv)) if _is_classifier(estimator) else KFold(int(cv))
        )
        return splitter.split
    if hasattr(cv, "split"):
        return cv.split
    if isinstance(cv, Iterable):
        return lambda X, y, groups: iter(cv)
    raise TypeError("cv must be an integer, splitter, or iterable")


def _slice_fit_params(
    params: dict[str, Any], train_indices: NDArray[np.intp], samples: int
) -> dict[str, Any]:
    return {
        name: safe_indexing(value, train_indices)
        if is_arraylike(value, samples)
        else value
        for name, value in params.items()
    }


def cross_val_score(
    estimator: Any,
    X: Any,
    y: Any = None,
    *,
    groups: Any = None,
    scoring: Any = None,
    cv: Any = None,
    n_jobs: int | None = None,
    verbose: int = 0,
    params: dict[str, Any] | None = None,
    pre_dispatch: str | int = "2*n_jobs",
    error_score: str | float = np.nan,
) -> NDArray[np.float64]:
    """Evaluate a score by cross-validation."""
    if n_jobs not in (None, 1):
        raise NotImplementedError(
            "parallel cross-validation through n_jobs is not implemented"
        )
    if (
        isinstance(verbose, (bool, np.bool_))
        or not isinstance(verbose, (int, np.integer))
        or verbose < 0
    ):
        raise ValueError("verbose must be a non-negative integer")
    del pre_dispatch
    if params is not None and not isinstance(params, dict):
        raise TypeError("params must be a dict or None")
    if error_score != "raise":
        try:
            error_score = float(error_score)
        except (TypeError, ValueError) as error:
            raise ValueError("error_score must be 'raise' or numeric") from error
    samples = check_consistent_length(X, y, groups)
    scorer = _resolve_scorer(estimator, scoring)
    split = _resolve_cv(cv, estimator, y)
    fit_params = {} if params is None else params
    scores: list[float] = []
    failures = 0
    for fold, (train_indices, test_indices) in enumerate(split(X, y, groups)):
        train_indices = np.asarray(train_indices, dtype=np.intp)
        test_indices = np.asarray(test_indices, dtype=np.intp)
        fitted = clone(estimator)
        try:
            X_train = safe_indexing(X, train_indices)
            sliced_params = _slice_fit_params(fit_params, train_indices, samples)
            if y is None:
                fitted.fit(X_train, **sliced_params)
            else:
                fitted.fit(
                    X_train,
                    safe_indexing(y, train_indices),
                    **sliced_params,
                )
            score = scorer(
                fitted,
                safe_indexing(X, test_indices),
                None if y is None else safe_indexing(y, test_indices),
            )
            score = float(score)
            if not np.ndim(score) == 0:
                raise ValueError("scoring must return a scalar")
        except Exception as error:
            if error_score == "raise":
                raise
            failures += 1
            score = float(error_score)
            warnings.warn(
                f"Estimator fit or scoring failed on fold {fold}; score set to "
                f"{score}. Details: {error}",
                FitFailedWarning,
                stacklevel=2,
            )
        scores.append(score)
        if verbose:
            print(f"[CV] fold={fold + 1} score={score:.6f}")
    if not scores:
        raise ValueError("cv produced no train/test splits")
    if failures == len(scores):
        raise ValueError("All cross-validation fits failed")
    return np.asarray(scores, dtype=np.float64)


__all__ = ["FitFailedWarning", "cross_val_score"]
