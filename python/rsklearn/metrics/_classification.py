"""Classification metrics backed by Rust reductions."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rsklearn import _core
from rsklearn.metrics._validation import UndefinedMetricWarning, validate_sample_weight
from rsklearn.preprocessing import LabelEncoder


def _validate_targets(
    y_true: Any, y_pred: Any, *, allow_multilabel: bool = False
) -> tuple[NDArray[Any], NDArray[Any]]:
    expected = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    allowed = (1, 2) if allow_multilabel else (1,)
    if expected.ndim not in allowed or predicted.ndim not in allowed:
        raise ValueError("classification targets have an unsupported shape")
    if expected.shape != predicted.shape:
        raise ValueError(
            "y_true and y_pred have different shapes: "
            f"{expected.shape} and {predicted.shape}"
        )
    if expected.dtype.kind in "fc" and (
        not np.isfinite(expected).all() or np.any(expected != np.floor(expected))
    ):
        raise ValueError("classification metrics do not support continuous targets")
    if predicted.dtype.kind in "fc" and (
        not np.isfinite(predicted).all() or np.any(predicted != np.floor(predicted))
    ):
        raise ValueError("classification metrics do not support continuous targets")
    return expected, predicted


def _classes_and_codes(
    expected: NDArray[Any], predicted: NDArray[Any], labels: Any = None
) -> tuple[NDArray[Any], NDArray[np.int64], NDArray[np.int64], NDArray[np.bool_]]:
    if labels is None:
        try:
            encoder = LabelEncoder()
            encoded = encoder.fit_transform(np.concatenate((expected, predicted)))
        except (TypeError, ValueError) as error:
            raise TypeError(
                "classification labels must have a consistent type"
            ) from error
        classes = encoder.classes_
        expected_codes = np.ascontiguousarray(encoded[: expected.size])
        predicted_codes = np.ascontiguousarray(encoded[expected.size :])
        included = np.ones(expected.size, dtype=bool)
        return classes, expected_codes, predicted_codes, included
    else:
        classes = np.asarray(labels)
        if classes.ndim != 1 or classes.size == 0:
            raise ValueError("labels must contain at least one value")
        if np.unique(classes).size != classes.size:
            raise ValueError("labels must contain unique values")
    mapping = {
        value.item() if isinstance(value, np.generic) else value: index
        for index, value in enumerate(classes)
    }
    expected_codes = np.asarray(
        [
            mapping.get(value.item() if isinstance(value, np.generic) else value, -1)
            for value in expected
        ],
        dtype=np.int64,
    )
    predicted_codes = np.asarray(
        [
            mapping.get(value.item() if isinstance(value, np.generic) else value, -1)
            for value in predicted
        ],
        dtype=np.int64,
    )
    included = (expected_codes >= 0) & (predicted_codes >= 0)
    return classes, expected_codes, predicted_codes, included


def accuracy_score(
    y_true: Any,
    y_pred: Any,
    *,
    normalize: bool = True,
    sample_weight: Any = None,
) -> float:
    """Return subset accuracy or the weighted number of correct predictions."""
    expected, predicted = _validate_targets(y_true, y_pred, allow_multilabel=True)
    if not isinstance(normalize, (bool, np.bool_)):
        raise TypeError("normalize must be bool")
    samples = expected.shape[0]
    if samples == 0:
        return float("nan") if normalize else 0.0
    matches = (
        np.all(expected == predicted, axis=1)
        if expected.ndim == 2
        else expected == predicted
    )
    weights = validate_sample_weight(sample_weight, samples)
    correct, total = _core.metric_accuracy(
        np.ascontiguousarray(matches, dtype=np.int64),
        np.ones(samples, dtype=np.int64),
        weights,
    )
    return (
        float(correct / total)
        if normalize and total
        else (float("nan") if normalize else float(correct))
    )


def confusion_matrix(
    y_true: Any,
    y_pred: Any,
    *,
    labels: Any = None,
    sample_weight: Any = None,
    normalize: str | None = None,
) -> NDArray[Any]:
    """Compute a matrix whose rows are true classes and columns are predictions."""
    expected, predicted = _validate_targets(y_true, y_pred)
    if normalize not in (None, "true", "pred", "all"):
        raise ValueError("normalize must be None, 'true', 'pred', or 'all'")
    if expected.size == 0 and labels is None:
        return np.empty((0, 0), dtype=np.int64)
    weights = validate_sample_weight(
        sample_weight, expected.size, allow_zero_total=True
    )
    if labels is None and expected.dtype.kind in "bi" and predicted.dtype.kind in "bi":
        classes, matrix = _core.metric_confusion_i64(
            np.ascontiguousarray(expected, dtype=np.int64),
            np.ascontiguousarray(predicted, dtype=np.int64),
            weights,
        )
        classes = np.asarray(classes)
        matrix = np.asarray(matrix)
        if classes.size == 1:
            warnings.warn(
                "A single label was found in y_true and y_pred. Pass labels to "
                "request a larger confusion matrix.",
                UserWarning,
                stacklevel=2,
            )
        return _normalize_confusion(matrix, normalize, sample_weight is not None)
    classes, expected_codes, predicted_codes, included = _classes_and_codes(
        expected, predicted, labels
    )
    if labels is not None and expected.size and not np.any(expected_codes >= 0):
        raise ValueError("At least one label specified must be in y_true")
    if labels is None and classes.size == 1 and expected.size:
        warnings.warn(
            "A single label was found in y_true and y_pred. Pass labels to request "
            "a larger confusion matrix.",
            UserWarning,
            stacklevel=2,
        )
    matrix = (
        _core.metric_confusion_matrix(
            np.ascontiguousarray(expected_codes[included]),
            np.ascontiguousarray(predicted_codes[included]),
            np.ascontiguousarray(weights[included]),
            classes.size,
        )
        if np.any(included)
        else np.zeros((classes.size, classes.size), dtype=np.float64)
    )
    return _normalize_confusion(
        np.asarray(matrix), normalize, sample_weight is not None
    )


def _normalize_confusion(
    matrix: NDArray[Any], normalize: str | None, weighted: bool
) -> NDArray[Any]:
    if normalize is None:
        return matrix if weighted else matrix.astype(np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        if normalize == "true":
            matrix = matrix / matrix.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            matrix = matrix / matrix.sum(axis=0, keepdims=True)
        else:
            matrix = matrix / matrix.sum()
    return np.nan_to_num(matrix)


def _zero_division_value(
    zero_division: Any, metric: str, count: int, *, warn: bool = True
) -> float:
    if zero_division == "warn":
        if not warn:
            return 0.0
        warnings.warn(
            f"{metric.capitalize()} is ill-defined and being set to 0.0 due to "
            "zero division.",
            UndefinedMetricWarning,
            stacklevel=3,
        )
        return 0.0
    if zero_division in (0, 1) or (
        isinstance(zero_division, (float, np.floating)) and np.isnan(zero_division)
    ):
        return float(zero_division)
    raise ValueError("zero_division must be 'warn', 0, 1, or np.nan")


def _prf_score(
    metric: str,
    y_true: Any,
    y_pred: Any,
    *,
    labels: Any,
    pos_label: Any,
    average: str | None,
    sample_weight: Any,
    zero_division: Any,
) -> Any:
    if average not in (None, "binary", "micro", "macro", "weighted"):
        raise ValueError(
            "average must be None, 'binary', 'micro', 'macro', or 'weighted'"
        )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="A single label was found in y_true and y_pred"
        )
        matrix = confusion_matrix(
            y_true, y_pred, labels=labels, sample_weight=sample_weight
        ).astype(np.float64)
    classes, _, _, _ = _classes_and_codes(
        *_validate_targets(y_true, y_pred), labels=labels
    )
    true_positive = np.diag(matrix)
    predicted = matrix.sum(axis=0)
    actual = matrix.sum(axis=1)
    if metric == "precision":
        numerator, denominator = true_positive, predicted
    elif metric == "recall":
        numerator, denominator = true_positive, actual
    else:
        numerator, denominator = 2 * true_positive, actual + predicted
    scores = np.empty(classes.size, dtype=np.float64)
    valid = denominator != 0
    scores[valid] = numerator[valid] / denominator[valid]
    scores[~valid] = _zero_division_value(
        zero_division,
        metric,
        int(np.sum(~valid)),
        warn=average in (None, "macro", "weighted"),
    )
    if average is None:
        return scores
    if average == "binary":
        if classes.size > 2:
            raise ValueError("average='binary' is not supported for multiclass targets")
        matches = np.flatnonzero(classes == pos_label)
        if matches.size != 1:
            if classes.size == 1:
                return _zero_division_value(zero_division, metric, 1)
            raise ValueError(f"pos_label={pos_label!r} is not a valid label")
        if not valid[matches[0]]:
            return _zero_division_value(zero_division, metric, 1)
        return float(scores[matches[0]])
    if average == "micro":
        numerator_sum = float(np.sum(numerator))
        denominator_sum = float(np.sum(denominator))
        return (
            numerator_sum / denominator_sum
            if denominator_sum
            else _zero_division_value(zero_division, metric, 1)
        )
    if average == "weighted":
        return (
            float(np.average(scores, weights=actual))
            if np.sum(actual)
            else float(np.mean(scores))
        )
    return float(np.mean(scores))


def precision_score(
    y_true: Any,
    y_pred: Any,
    *,
    labels: Any = None,
    pos_label: Any = 1,
    average: str | None = "binary",
    sample_weight: Any = None,
    zero_division: Any = "warn",
) -> Any:
    """Compute precision."""
    return _prf_score(
        "precision",
        y_true,
        y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )


def recall_score(
    y_true: Any,
    y_pred: Any,
    *,
    labels: Any = None,
    pos_label: Any = 1,
    average: str | None = "binary",
    sample_weight: Any = None,
    zero_division: Any = "warn",
) -> Any:
    """Compute recall."""
    return _prf_score(
        "recall",
        y_true,
        y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )


def f1_score(
    y_true: Any,
    y_pred: Any,
    *,
    labels: Any = None,
    pos_label: Any = 1,
    average: str | None = "binary",
    sample_weight: Any = None,
    zero_division: Any = "warn",
) -> Any:
    """Compute the harmonic mean of precision and recall."""
    return _prf_score(
        "f1",
        y_true,
        y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )
