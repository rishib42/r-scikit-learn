"""Base classes and cloning utilities for rsklearn estimators."""

from __future__ import annotations

import copy
import inspect
from collections import defaultdict
from typing import Any

import numpy as np


def _clone_parameter(value: Any, *, safe: bool) -> Any:
    if isinstance(value, dict):
        return type(value)((key, clone(item, safe=safe)) for key, item in value.items())
    if isinstance(value, (list, tuple, set, frozenset)):
        return type(value)(clone(item, safe=safe) for item in value)
    return clone(value, safe=safe)


def clone(estimator: Any, *, safe: bool = True) -> Any:
    """Construct a new unfitted estimator with the same parameters.

    Parameters are recursively cloned. Learned attributes are intentionally not
    copied.
    """
    if estimator is None or isinstance(
        estimator, (str, bytes, int, float, complex, bool, np.generic)
    ):
        if safe:
            raise TypeError(
                f"Cannot clone object {estimator!r}; it does not implement get_params"
            )
        return copy.deepcopy(estimator)
    if hasattr(estimator, "__sklearn_clone__") and not isinstance(estimator, type):
        return estimator.__sklearn_clone__()
    if isinstance(estimator, dict):
        return type(estimator)(
            (key, clone(value, safe=False)) for key, value in estimator.items()
        )
    if isinstance(estimator, (list, tuple, set, frozenset)):
        return type(estimator)(clone(value, safe=False) for value in estimator)
    if not hasattr(estimator, "get_params") or isinstance(estimator, type):
        if safe:
            raise TypeError(
                f"Cannot clone object {estimator!r}; it does not implement get_params"
            )
        return copy.deepcopy(estimator)

    estimator_type = type(estimator)
    parameters = estimator.get_params(deep=False)
    cloned_parameters = {
        name: _clone_parameter(value, safe=False) for name, value in parameters.items()
    }
    try:
        result = estimator_type(**cloned_parameters)
    except TypeError as error:
        raise TypeError(
            f"Cannot clone {estimator_type.__name__}; its constructor could not be "
            "called with get_params(deep=False)"
        ) from error
    result_parameters = result.get_params(deep=False)
    for name, value in cloned_parameters.items():
        if name not in result_parameters:
            raise RuntimeError(
                f"Cannot clone {estimator_type.__name__}; constructor does not set "
                f"parameter {name!r}"
            )
        if result_parameters[name] is not value:
            raise RuntimeError(
                f"Cannot clone {estimator_type.__name__}; constructor modifies "
                f"parameter {name!r}"
            )
    return result


class BaseEstimator:
    """Base class implementing parameter inspection and mutation."""

    _rsklearn_input_tags: dict[str, Any] = {}
    _rsklearn_target_tags: dict[str, Any] = {"required": False}

    @classmethod
    def _get_param_names(cls) -> list[str]:
        constructor = cls.__init__
        if constructor is object.__init__:
            return []
        signature = inspect.signature(constructor)
        parameters = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.name != "self" and parameter.kind != parameter.VAR_KEYWORD
        ]
        if any(parameter.kind == parameter.VAR_POSITIONAL for parameter in parameters):
            raise RuntimeError(
                f"{cls.__name__} estimators must declare constructor parameters "
                "explicitly and may not use *args"
            )
        return sorted(parameter.name for parameter in parameters)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor parameters, optionally including nested parameters."""
        output: dict[str, Any] = {}
        for name in self._get_param_names():
            value = getattr(self, name)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                output.update(
                    (f"{name}__{key}", item) for key, item in value.get_params().items()
                )
            output[name] = value
        return output

    def set_params(self, **params: Any) -> BaseEstimator:
        """Set constructor parameters, including ``name__nested`` parameters."""
        if not params:
            return self
        valid = self.get_params(deep=True)
        nested: defaultdict[str, dict[str, Any]] = defaultdict(dict)
        for key, value in params.items():
            name, delimiter, sub_name = key.partition("__")
            if name not in valid:
                valid_names = ", ".join(sorted(self._get_param_names()))
                raise ValueError(
                    f"Invalid parameter {name!r} for estimator {type(self).__name__}. "
                    f"Valid parameters are: {valid_names}."
                )
            if delimiter:
                nested[name][sub_name] = value
            else:
                setattr(self, name, value)
        for name, sub_params in nested.items():
            getattr(self, name).set_params(**sub_params)
        return self

    def _validate_params(self) -> None:
        """Validate constructor parameters after mutation."""

    def __sklearn_tags__(self) -> Any:
        """Return modern scikit-learn tags when scikit-learn is installed."""
        try:
            from sklearn.utils import InputTags, Tags, TargetTags
        except ImportError as error:
            raise ImportError(
                "__sklearn_tags__ requires the optional scikit-learn dependency"
            ) from error
        return Tags(
            estimator_type=getattr(self, "_estimator_type", None),
            target_tags=TargetTags(**self._rsklearn_target_tags),
            input_tags=InputTags(**self._rsklearn_input_tags),
        )

    def __repr__(self) -> str:
        arguments = ", ".join(
            f"{name}={value!r}" for name, value in self.get_params(deep=False).items()
        )
        return f"{type(self).__name__}({arguments})"


class TransformerMixin:
    """Mixin implementing ``fit_transform`` through ``fit`` and ``transform``."""

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> Any:
        if y is None:
            return self.fit(X, **fit_params).transform(X)
        return self.fit(X, y, **fit_params).transform(X)

    def __sklearn_tags__(self) -> Any:
        from sklearn.utils import TransformerTags

        tags = super().__sklearn_tags__()
        tags.transformer_tags = TransformerTags(
            preserves_dtype=getattr(self, "_rsklearn_preserves_dtype", [])
        )
        return tags


class ClassifierMixin:
    """Mixin providing a basic classification accuracy score."""

    _estimator_type = "classifier"
    _rsklearn_target_tags = {"required": True}

    def score(self, X: Any, y: Any) -> float:
        predicted = np.asarray(self.predict(X))
        expected = np.asarray(y)
        if predicted.shape != expected.shape:
            raise ValueError("predicted and expected labels have different shapes")
        matches = predicted == expected
        if matches.ndim > 1:
            matches = np.all(matches, axis=1)
        return float(np.mean(matches))

    def __sklearn_tags__(self) -> Any:
        from sklearn.utils import ClassifierTags

        tags = super().__sklearn_tags__()
        tags.classifier_tags = ClassifierTags()
        return tags


class RegressorMixin:
    """Mixin providing the coefficient of determination regression score."""

    _estimator_type = "regressor"
    _rsklearn_target_tags = {"required": True, "two_d_labels": True}

    def score(self, X: Any, y: Any) -> float:
        predicted = np.asarray(self.predict(X), dtype=np.float64)
        expected = np.asarray(y, dtype=np.float64)
        if predicted.shape != expected.shape:
            raise ValueError("predicted and expected targets have different shapes")
        if expected.ndim == 1:
            expected = expected[:, None]
            predicted = predicted[:, None]
        numerator = np.sum((expected - predicted) ** 2, axis=0)
        denominator = np.sum((expected - np.mean(expected, axis=0)) ** 2, axis=0)
        scores = np.empty_like(numerator)
        constant = denominator == 0
        scores[constant] = np.where(numerator[constant] == 0, 1.0, 0.0)
        scores[~constant] = 1.0 - numerator[~constant] / denominator[~constant]
        return float(np.mean(scores))

    def __sklearn_tags__(self) -> Any:
        from sklearn.utils import RegressorTags

        tags = super().__sklearn_tags__()
        tags.regressor_tags = RegressorTags()
        return tags


__all__ = [
    "BaseEstimator",
    "ClassifierMixin",
    "RegressorMixin",
    "TransformerMixin",
    "clone",
]
