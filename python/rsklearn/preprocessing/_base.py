"""Small shared scikit-learn-style estimator helpers."""

from __future__ import annotations

from typing import Any


class EstimatorMixin:
    _parameter_names: tuple[str, ...] = ()

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor parameters. ``deep`` is accepted for compatibility."""
        del deep
        return {name: getattr(self, name) for name in self._parameter_names}

    def set_params(self, **params: Any) -> EstimatorMixin:
        """Set constructor parameters and return self."""
        invalid = sorted(set(params) - set(self._parameter_names))
        if invalid:
            raise ValueError(f"Invalid parameter(s): {', '.join(invalid)}")
        for name, value in params.items():
            setattr(self, name, value)
        self._validate_params()
        return self

    def _validate_params(self) -> None:
        pass

    def _check_fitted(self, *attributes: str) -> None:
        if not all(hasattr(self, attribute) for attribute in attributes):
            raise ValueError(f"{type(self).__name__} is not fitted yet; call fit first")

    def __repr__(self) -> str:
        arguments = ", ".join(
            f"{name}={getattr(self, name)!r}" for name in self._parameter_names
        )
        return f"{type(self).__name__}({arguments})"
