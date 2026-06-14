"""Estimator composition utilities."""

from __future__ import annotations

import copy
import time
from collections import Counter
from collections.abc import Iterator
from typing import Any

import numpy as np

from rsklearn.base import BaseEstimator


def _is_passthrough(estimator: Any) -> bool:
    return estimator is None or (
        isinstance(estimator, str) and estimator == "passthrough"
    )


def _name_estimators(estimators: tuple[Any, ...]) -> list[tuple[str, Any]]:
    names = [
        estimator if isinstance(estimator, str) else type(estimator).__name__.lower()
        for estimator in estimators
    ]
    counts = Counter(names)
    remaining = counts.copy()
    for index in range(len(names) - 1, -1, -1):
        name = names[index]
        if counts[name] > 1:
            names[index] = f"{name}-{remaining[name]}"
            remaining[name] -= 1
    return list(zip(names, estimators, strict=True))


class _NamedSteps(dict[str, Any]):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


class Pipeline(BaseEstimator):
    """Apply transformers sequentially and optionally fit a final estimator.

    Intermediate steps must implement ``fit`` and ``transform``. The final
    estimator only needs to implement ``fit`` and may expose prediction,
    transformation, scoring, or inverse-transformation methods.
    """

    def __init__(
        self,
        steps: list[tuple[str, Any]],
        *,
        transform_input: list[str] | None = None,
        memory: Any = None,
        verbose: bool = False,
    ) -> None:
        self.steps = steps
        self.transform_input = transform_input
        self.memory = memory
        self.verbose = verbose

    def __getattribute__(self, name: str) -> Any:
        delegated = {
            "decision_function",
            "predict",
            "predict_proba",
            "score",
            "score_samples",
        }
        if name in delegated:
            steps = object.__getattribute__(self, "steps")
            final = steps[-1][1] if steps else None
            if _is_passthrough(final) or not hasattr(final, name):
                raise AttributeError(
                    f"final pipeline estimator does not implement {name}"
                )
        elif name in {"transform", "get_feature_names_out"}:
            steps = object.__getattribute__(self, "steps")
            if steps:
                method = name
                if any(
                    not _is_passthrough(estimator) and not hasattr(estimator, method)
                    for _, estimator in steps
                ):
                    raise AttributeError(
                        f"not every pipeline estimator implements {method}"
                    )
        elif name == "fit_transform":
            steps = object.__getattribute__(self, "steps")
            transformers = steps[:-1]
            final = steps[-1][1] if steps else None
            if any(
                not _is_passthrough(estimator) and not hasattr(estimator, "transform")
                for _, estimator in transformers
            ) or (
                not _is_passthrough(final)
                and not hasattr(final, "fit_transform")
                and not hasattr(final, "transform")
            ):
                raise AttributeError("pipeline estimators do not support fit_transform")
        elif name == "inverse_transform":
            steps = object.__getattribute__(self, "steps")
            if any(
                not _is_passthrough(estimator)
                and not hasattr(estimator, "inverse_transform")
                for _, estimator in steps
            ):
                raise AttributeError(
                    "not every pipeline estimator implements inverse_transform"
                )
        return object.__getattribute__(self, name)

    @property
    def named_steps(self) -> _NamedSteps:
        """Return a mapping supporting key and attribute access to steps."""
        return _NamedSteps(self.steps)

    @property
    def _final_estimator(self) -> Any:
        return self.steps[-1][1] if self.steps else None

    @property
    def n_features_in_(self) -> int:
        return next(
            estimator.n_features_in_
            for _, estimator in self.steps
            if not _is_passthrough(estimator)
        )

    @property
    def feature_names_in_(self) -> Any:
        return next(
            estimator.feature_names_in_
            for _, estimator in self.steps
            if not _is_passthrough(estimator)
        )

    @property
    def classes_(self) -> Any:
        return self._final_estimator.classes_

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, index: int | slice | str) -> Any:
        if isinstance(index, str):
            return self.named_steps[index]
        if isinstance(index, slice):
            return type(self)(
                self.steps[index],
                transform_input=self.transform_input,
                memory=self.memory,
                verbose=self.verbose,
            )
        return self.steps[index][1]

    def __iter__(self) -> Iterator[Any]:
        return (estimator for _, estimator in self.steps)

    def _validate_steps(self) -> None:
        if not isinstance(self.steps, list) or not self.steps:
            raise ValueError(
                "steps must be a non-empty list of (name, estimator) pairs"
            )
        if not isinstance(self.verbose, (bool, np.bool_)):
            raise TypeError("verbose must be bool")
        if self.memory is not None:
            raise NotImplementedError("Pipeline memory caching is not implemented")
        if self.transform_input is not None:
            raise NotImplementedError("Pipeline metadata routing is not implemented")
        names: list[str] = []
        for item in self.steps:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("each pipeline step must be a (name, estimator) pair")
            name, estimator = item
            if not isinstance(name, str) or not name:
                raise TypeError("pipeline step names must be non-empty strings")
            if "__" in name:
                raise ValueError("pipeline step names must not contain '__'")
            names.append(name)
            if _is_passthrough(estimator):
                continue
            if not hasattr(estimator, "fit"):
                raise TypeError(f"pipeline step {name!r} must implement fit")
        if len(set(names)) != len(names):
            raise ValueError("pipeline step names must be unique")
        conflicts = set(names) & {"steps", "memory", "verbose", "transform_input"}
        if conflicts:
            raise ValueError(
                "pipeline step names conflict with constructor arguments: "
                f"{sorted(conflicts)}"
            )
        for name, estimator in self.steps[:-1]:
            if _is_passthrough(estimator):
                continue
            if not hasattr(estimator, "transform"):
                raise TypeError(
                    f"intermediate pipeline step {name!r} must implement transform"
                )

    def _iter_transformers(self, *, reverse: bool = False) -> Iterator[tuple[str, Any]]:
        steps = self.steps[:-1]
        values = reversed(steps) if reverse else iter(steps)
        return (
            (name, estimator)
            for name, estimator in values
            if not _is_passthrough(estimator)
        )

    def _route_params(self, params: dict[str, Any]) -> dict[str, dict[str, Any]]:
        routed = {name: {} for name, _ in self.steps}
        names = set(routed)
        for key, value in params.items():
            name, separator, parameter = key.partition("__")
            if not separator:
                raise ValueError(
                    f"Pipeline.fit does not accept {key!r}; use step__parameter"
                )
            if name not in names:
                raise ValueError(f"unknown pipeline step {name!r}")
            routed[name][parameter] = value
        return routed

    @staticmethod
    def _fit_transform_step(
        estimator: Any, X: Any, y: Any, params: dict[str, Any]
    ) -> Any:
        if hasattr(estimator, "fit_transform"):
            return estimator.fit_transform(X, y, **params)
        return estimator.fit(X, y, **params).transform(X)

    def _fit_transformers(
        self, X: Any, y: Any, routed: dict[str, dict[str, Any]]
    ) -> Any:
        transformed = X
        for name, estimator in self._iter_transformers():
            started_at = time.perf_counter()
            transformed = self._fit_transform_step(
                estimator, transformed, y, routed[name]
            )
            self._log_step(name, started_at)
        return transformed

    def _log_step(self, name: str, started_at: float) -> None:
        if self.verbose:
            elapsed = time.perf_counter() - started_at
            print(f"[Pipeline] {name} completed in {elapsed:.3f}s")

    def fit(self, X: Any, y: Any = None, **params: Any) -> Pipeline:
        """Fit all transformers sequentially and fit the final estimator."""
        self._validate_steps()
        routed = self._route_params(params)
        transformed = self._fit_transformers(X, y, routed)
        final_name, final = self.steps[-1]
        if not _is_passthrough(final):
            started_at = time.perf_counter()
            final.fit(transformed, y, **routed[final_name])
            self._log_step(final_name, started_at)
        self._is_fitted = True
        return self

    def fit_transform(self, X: Any, y: Any = None, **params: Any) -> Any:
        """Fit every step and transform X through the complete pipeline."""
        self._validate_steps()
        routed = self._route_params(params)
        transformed = self._fit_transformers(X, y, routed)
        final_name, final = self.steps[-1]
        if _is_passthrough(final):
            self._is_fitted = True
            return transformed
        if not hasattr(final, "fit_transform") and not hasattr(final, "transform"):
            raise AttributeError(
                "final pipeline estimator does not implement fit_transform or transform"
            )
        started_at = time.perf_counter()
        transformed = self._fit_transform_step(
            final, transformed, y, routed[final_name]
        )
        self._log_step(final_name, started_at)
        self._is_fitted = True
        return transformed

    def _transform(self, X: Any) -> Any:
        self._require_fitted()
        transformed = X
        for _, estimator in self.steps:
            if _is_passthrough(estimator):
                continue
            if not hasattr(estimator, "transform"):
                raise AttributeError("pipeline estimator does not implement transform")
            transformed = estimator.transform(transformed)
        return transformed

    def transform(self, X: Any, **params: Any) -> Any:
        """Transform X through every step."""
        if params:
            raise NotImplementedError("transform metadata routing is not implemented")
        return self._transform(X)

    def _transform_before_final(self, X: Any) -> Any:
        self._require_fitted()
        transformed = X
        for _, estimator in self._iter_transformers():
            transformed = estimator.transform(transformed)
        return transformed

    def predict(self, X: Any, **params: Any) -> Any:
        """Transform X and predict using the final estimator."""
        final = self._require_final_method("predict")
        return final.predict(self._transform_before_final(X), **params)

    def predict_proba(self, X: Any, **params: Any) -> Any:
        """Transform X and return final-estimator class probabilities."""
        final = self._require_final_method("predict_proba")
        return final.predict_proba(self._transform_before_final(X), **params)

    def decision_function(self, X: Any, **params: Any) -> Any:
        """Transform X and return final-estimator decision values."""
        final = self._require_final_method("decision_function")
        return final.decision_function(self._transform_before_final(X), **params)

    def score_samples(self, X: Any) -> Any:
        """Transform X and return final-estimator per-sample scores."""
        final = self._require_final_method("score_samples")
        return final.score_samples(self._transform_before_final(X))

    def score(
        self, X: Any, y: Any = None, sample_weight: Any = None, **params: Any
    ) -> float:
        """Transform X and score using the final estimator."""
        final = self._require_final_method("score")
        if sample_weight is not None:
            params["sample_weight"] = sample_weight
        return float(final.score(self._transform_before_final(X), y, **params))

    def inverse_transform(self, X: Any, **params: Any) -> Any:
        """Apply inverse transforms in reverse step order."""
        if params:
            raise NotImplementedError(
                "inverse_transform metadata routing is not implemented"
            )
        self._require_fitted()
        transformed = X
        for name, estimator in reversed(self.steps):
            if _is_passthrough(estimator):
                continue
            if not hasattr(estimator, "inverse_transform"):
                raise AttributeError(
                    f"pipeline step {name!r} does not implement inverse_transform"
                )
            transformed = estimator.inverse_transform(transformed)
        return transformed

    def get_feature_names_out(self, input_features: Any = None) -> Any:
        """Propagate feature names through every transformer."""
        self._require_fitted()
        names = input_features
        for name, estimator in self.steps:
            if _is_passthrough(estimator):
                continue
            if not hasattr(estimator, "get_feature_names_out"):
                raise AttributeError(
                    f"pipeline step {name!r} does not provide get_feature_names_out"
                )
            names = estimator.get_feature_names_out(names)
        return names

    def _require_fitted(self) -> None:
        if not getattr(self, "_is_fitted", False):
            raise ValueError(
                "This Pipeline instance is not fitted yet. Call 'fit' with appropriate "
                "arguments before using this estimator."
            )

    def _require_final_method(self, method: str) -> Any:
        self._require_fitted()
        final = self._final_estimator
        if _is_passthrough(final) or not hasattr(final, method):
            raise AttributeError(
                f"final pipeline estimator does not implement {method}"
            )
        return final

    def __sklearn_is_fitted__(self) -> bool:
        return bool(getattr(self, "_is_fitted", False))

    def __sklearn_tags__(self) -> Any:
        """Delegate estimator-type tags to the final estimator."""
        final = self._final_estimator
        if not _is_passthrough(final) and hasattr(final, "__sklearn_tags__"):
            return copy.deepcopy(final.__sklearn_tags__())
        return super().__sklearn_tags__()

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor, named-step, and nested estimator parameters."""
        output = {
            "memory": self.memory,
            "steps": self.steps,
            "transform_input": self.transform_input,
            "verbose": self.verbose,
        }
        if not deep:
            return output
        for name, estimator in self.steps:
            output[name] = estimator
            if not _is_passthrough(estimator) and hasattr(estimator, "get_params"):
                output.update(
                    (f"{name}__{key}", value)
                    for key, value in estimator.get_params(deep=True).items()
                )
        return output

    def set_params(self, **params: Any) -> Pipeline:
        """Set constructor, named-step, and nested estimator parameters."""
        if not params:
            return self
        constructor_names = {"memory", "steps", "transform_input", "verbose"}
        for name in constructor_names & params.keys():
            setattr(self, name, params.pop(name))
        step_names = {name for name, _ in self.steps}
        replacements = step_names & params.keys()
        if replacements:
            self.steps = [
                (name, params.pop(name) if name in replacements else estimator)
                for name, estimator in self.steps
            ]
        nested: dict[str, dict[str, Any]] = {}
        for key, value in params.items():
            name, separator, parameter = key.partition("__")
            if not separator or name not in step_names:
                raise ValueError(f"Invalid parameter {key!r} for Pipeline")
            nested.setdefault(name, {})[parameter] = value
        for name, values in nested.items():
            estimator = self.named_steps[name]
            if _is_passthrough(estimator) or not hasattr(estimator, "set_params"):
                raise ValueError(f"pipeline step {name!r} does not accept parameters")
            estimator.set_params(**values)
        return self


def make_pipeline(
    *steps: Any,
    memory: Any = None,
    transform_input: list[str] | None = None,
    verbose: bool = False,
) -> Pipeline:
    """Construct a Pipeline with automatically generated step names."""
    return Pipeline(
        _name_estimators(steps),
        transform_input=transform_input,
        memory=memory,
        verbose=verbose,
    )


__all__ = ["Pipeline", "make_pipeline"]
