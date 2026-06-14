"""Column-wise estimator composition."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rsklearn.base import BaseEstimator, TransformerMixin, clone
from rsklearn.pipeline import _NamedSteps
from rsklearn.utils.validation import check_is_fitted


def _is_sparse(value: Any) -> bool:
    from scipy import sparse

    return bool(sparse.issparse(value))


def _is_drop(value: Any) -> bool:
    return isinstance(value, str) and value == "drop"


def _is_passthrough(value: Any) -> bool:
    return isinstance(value, str) and value == "passthrough"


def _feature_names(X: Any) -> NDArray[Any] | None:
    columns = getattr(X, "columns", None)
    if columns is None:
        return None
    names = np.asarray(columns, dtype=object)
    return names if all(isinstance(name, str) for name in names) else None


def _shape(X: Any) -> tuple[int, int]:
    shape = getattr(X, "shape", None)
    if shape is None:
        shape = np.asarray(X).shape
    if len(shape) != 2:
        raise ValueError("ColumnTransformer expected a 2-dimensional input")
    if shape[0] < 1 or shape[1] < 1:
        raise ValueError("ColumnTransformer requires non-empty input")
    return int(shape[0]), int(shape[1])


def _name_estimators(
    estimators: tuple[tuple[Any, Any], ...],
) -> list[tuple[str, Any, Any]]:
    names = [
        estimator if isinstance(estimator, str) else type(estimator).__name__.lower()
        for estimator, _ in estimators
    ]
    counts = Counter(names)
    remaining = counts.copy()
    for index in range(len(names) - 1, -1, -1):
        name = names[index]
        if counts[name] > 1:
            names[index] = f"{name}-{remaining[name]}"
            remaining[name] -= 1
    return [
        (name, estimator, columns)
        for name, (estimator, columns) in zip(names, estimators, strict=True)
    ]


def _validate_transformer(transformer: Any, *, name: str) -> None:
    if isinstance(transformer, str):
        if transformer not in {"drop", "passthrough"}:
            raise ValueError(
                f"transformer {name!r} must be an estimator, 'drop', or 'passthrough'"
            )
        return
    if not hasattr(transformer, "transform") or not (
        hasattr(transformer, "fit") or hasattr(transformer, "fit_transform")
    ):
        raise TypeError(
            f"transformer {name!r} must implement transform and fit or fit_transform"
        )


class ColumnTransformer(TransformerMixin, BaseEstimator):
    """Apply transformers to selected columns and concatenate their outputs."""

    def __init__(
        self,
        transformers: list[tuple[str, Any, Any]],
        *,
        remainder: str | Any = "drop",
        sparse_threshold: float = 0.3,
        n_jobs: int | None = None,
        transformer_weights: dict[str, float] | None = None,
        verbose: bool = False,
        verbose_feature_names_out: bool | str | Callable[[str, str], str] = True,
        force_int_remainder_cols: Any = "deprecated",
    ) -> None:
        self.transformers = transformers
        self.remainder = remainder
        self.sparse_threshold = sparse_threshold
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose
        self.verbose_feature_names_out = verbose_feature_names_out
        self.force_int_remainder_cols = force_int_remainder_cols

    @property
    def named_transformers_(self) -> _NamedSteps:
        check_is_fitted(self, "transformers_")
        return _NamedSteps(
            (name, transformer) for name, transformer, _ in self.transformers_
        )

    def _validate_params(self) -> None:
        if not isinstance(self.transformers, list):
            raise TypeError(
                "transformers must be a list of (name, transformer, columns)"
            )
        if self.n_jobs is not None:
            raise NotImplementedError(
                "parallel ColumnTransformer execution is not implemented"
            )
        if (
            isinstance(self.sparse_threshold, (bool, np.bool_))
            or not isinstance(self.sparse_threshold, (int, float, np.number))
            or not 0 <= self.sparse_threshold <= 1
        ):
            raise ValueError("sparse_threshold must be a number in [0, 1]")
        if not isinstance(self.verbose, (bool, np.bool_)):
            raise TypeError("verbose must be bool")
        if not (
            isinstance(self.verbose_feature_names_out, (bool, str))
            or callable(self.verbose_feature_names_out)
        ):
            raise TypeError("verbose_feature_names_out must be bool, str, or callable")
        _validate_transformer(self.remainder, name="remainder")
        names: list[str] = []
        for item in self.transformers:
            if not isinstance(item, tuple) or len(item) != 3:
                raise TypeError(
                    "each transformer must be a (name, transformer, columns) tuple"
                )
            name, transformer, _ = item
            if not isinstance(name, str) or not name:
                raise TypeError("transformer names must be non-empty strings")
            if "__" in name:
                raise ValueError("transformer names must not contain '__'")
            names.append(name)
            _validate_transformer(transformer, name=name)
        if len(set(names)) != len(names):
            raise ValueError("transformer names must be unique")
        conflicts = set(names) & {
            "transformers",
            "remainder",
            "sparse_threshold",
            "n_jobs",
            "transformer_weights",
            "verbose",
            "verbose_feature_names_out",
            "force_int_remainder_cols",
        }
        if conflicts:
            raise ValueError(
                "transformer names conflict with constructor arguments: "
                f"{sorted(conflicts)}"
            )
        if self.transformer_weights is not None:
            unknown = set(self.transformer_weights) - set(names) - {"remainder"}
            if unknown:
                raise ValueError(f"unknown transformer weights: {sorted(unknown)}")

    def _resolve_columns(self, X: Any, columns: Any) -> tuple[Any, NDArray[np.int64]]:
        if callable(columns):
            columns = columns(X)
        names = _feature_names(X)
        width = self.n_features_in_
        if isinstance(columns, str):
            if names is None:
                raise ValueError("string column selectors require named input columns")
            matches = np.flatnonzero(names == columns)
            if matches.size != 1:
                raise ValueError(f"column {columns!r} was not found uniquely")
            return columns, matches.astype(np.int64)
        if isinstance(columns, (int, np.integer)) and not isinstance(columns, bool):
            index = int(columns)
            if index < 0:
                index += width
            if index < 0 or index >= width:
                raise ValueError(f"column index {columns} is out of bounds")
            return int(columns), np.asarray([index], dtype=np.int64)
        if isinstance(columns, slice):
            if isinstance(columns.start, str) or isinstance(columns.stop, str):
                if names is None:
                    raise ValueError("string slices require named input columns")
                start = (
                    0
                    if columns.start is None
                    else int(np.flatnonzero(names == columns.start)[0])
                )
                stop = (
                    width - 1
                    if columns.stop is None
                    else int(np.flatnonzero(names == columns.stop)[0])
                )
                indices = np.arange(width, dtype=np.int64)[
                    slice(start, stop + 1, columns.step)
                ]
            else:
                indices = np.arange(width, dtype=np.int64)[columns]
            return columns, indices
        values = np.asarray(columns)
        if values.ndim != 1:
            raise ValueError("column selectors must be scalar or one-dimensional")
        if values.dtype.kind == "b":
            if values.size != width:
                raise ValueError("boolean column selector must match input width")
            return values.astype(bool), np.flatnonzero(values).astype(np.int64)
        if values.size == 0:
            return values.astype(np.int64), np.asarray([], dtype=np.int64)
        if all(isinstance(value, str) for value in values.tolist()):
            if names is None:
                raise ValueError("string column selectors require named input columns")
            indices: list[int] = []
            for value in values:
                matches = np.flatnonzero(names == value)
                if matches.size != 1:
                    raise ValueError(f"column {value!r} was not found uniquely")
                indices.append(int(matches[0]))
            return values.astype(object), np.asarray(indices, dtype=np.int64)
        if not all(
            isinstance(value, (int, np.integer)) and not isinstance(value, bool)
            for value in values.tolist()
        ):
            raise TypeError(
                "column selectors must contain only integers, strings, or booleans"
            )
        indices = values.astype(np.int64)
        indices = np.where(indices < 0, indices + width, indices)
        if np.any((indices < 0) | (indices >= width)):
            raise ValueError("column selector contains an out-of-bounds index")
        return values.astype(np.int64), indices

    @staticmethod
    def _select(X: Any, selector: Any, indices: NDArray[np.int64]) -> Any:
        if isinstance(selector, str):
            return X[selector]
        values = np.asarray(selector) if not isinstance(selector, slice) else None
        if (
            values is not None
            and values.ndim == 1
            and values.size
            and all(isinstance(value, str) for value in values.tolist())
        ):
            return X[list(values)]
        scalar = isinstance(selector, (int, np.integer)) and not isinstance(
            selector, bool
        )
        if hasattr(X, "iloc"):
            return X.iloc[:, int(indices[0])] if scalar else X.iloc[:, indices]
        if _is_sparse(X):
            return X[:, int(indices[0])] if scalar else X[:, indices]
        array = np.asarray(X)
        return array[:, int(indices[0])] if scalar else array[:, indices]

    @staticmethod
    def _fit_transform(estimator: Any, X: Any, y: Any, params: dict[str, Any]) -> Any:
        if hasattr(estimator, "fit_transform"):
            return estimator.fit_transform(X, y, **params)
        return estimator.fit(X, y, **params).transform(X)

    def _route_params(self, params: dict[str, Any]) -> dict[str, dict[str, Any]]:
        names = {name for name, _, _ in self.transformers} | {"remainder"}
        routed = {name: {} for name in names}
        for key, value in params.items():
            name, separator, parameter = key.partition("__")
            if not separator or name not in names:
                raise ValueError(
                    f"ColumnTransformer.fit does not accept {key!r}; "
                    "use transformer__parameter"
                )
            routed[name][parameter] = value
        return routed

    def _prepare_fit(self, X: Any) -> list[tuple[str, Any, Any, NDArray[np.int64]]]:
        self._validate_params()
        _, self.n_features_in_ = _shape(X)
        names = _feature_names(X)
        if names is not None:
            self.feature_names_in_ = names.copy()
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        resolved = []
        used = np.zeros(self.n_features_in_, dtype=bool)
        for name, transformer, columns in self.transformers:
            selector, indices = self._resolve_columns(X, columns)
            if names is not None:
                scalar = isinstance(
                    selector, (str, int, np.integer)
                ) and not isinstance(selector, bool)
                selector = (
                    str(names[indices[0]]) if scalar else names[indices].astype(object)
                )
            resolved.append((name, transformer, selector, indices))
            used[indices] = True
        remainder_indices = np.flatnonzero(~used).astype(np.int64)
        if remainder_indices.size:
            selector: Any = remainder_indices
            if names is not None:
                selector = names[remainder_indices].astype(object)
            resolved.append(("remainder", self.remainder, selector, remainder_indices))
        return resolved

    def _validate_transform_input(self, X: Any) -> None:
        _, width = _shape(X)
        if hasattr(self, "feature_names_in_"):
            names = _feature_names(X)
            if names is None:
                raise ValueError("input feature names must match fitted feature names")
            missing = np.setdiff1d(self.feature_names_in_, names)
            if missing.size:
                raise ValueError(
                    f"input is missing fitted feature names: {missing.tolist()}"
                )
        elif width != self.n_features_in_:
            raise ValueError(
                f"X has {width} features, but ColumnTransformer expects "
                f"{self.n_features_in_}"
            )

    def _weight(self, name: str, block: Any) -> Any:
        if self.transformer_weights is None or name not in self.transformer_weights:
            return block
        return block * self.transformer_weights[name]

    def _validate_block(self, block: Any, rows: int, *, name: str) -> Any:
        shape = getattr(block, "shape", None)
        if shape is None:
            block = np.asarray(block)
            shape = block.shape
        if len(shape) != 2:
            raise ValueError(f"transformer {name!r} output must be 2-dimensional")
        if shape[0] != rows:
            raise ValueError(f"transformer {name!r} output has the wrong sample count")
        return block

    def _stack(self, blocks: list[Any], rows: int, *, fit: bool) -> Any:
        if not blocks:
            if fit:
                self.sparse_output_ = False
            return np.empty((rows, 0))
        from scipy import sparse

        if fit:
            if any(sparse.issparse(block) for block in blocks):
                total = sum(int(np.prod(block.shape)) for block in blocks)
                nonzero = sum(
                    block.nnz if sparse.issparse(block) else np.count_nonzero(block)
                    for block in blocks
                )
                self.sparse_output_ = (
                    total > 0 and nonzero / total < self.sparse_threshold
                )
            else:
                self.sparse_output_ = False
        if self.sparse_output_:
            try:
                return sparse.hstack(blocks, format="csr")
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "sparse output requires numeric transformer outputs"
                ) from error
        dense = [
            block.toarray() if sparse.issparse(block) else np.asarray(block)
            for block in blocks
        ]
        return np.hstack(dense)

    def _log_step(self, name: str, started_at: float) -> None:
        if self.verbose:
            elapsed = time.perf_counter() - started_at
            print(f"[ColumnTransformer] {name} completed in {elapsed:.3f}s")

    def fit_transform(self, X: Any, y: Any = None, **params: Any) -> Any:
        """Fit all selected transformers and concatenate their outputs."""
        rows, _ = _shape(X)
        resolved = self._prepare_fit(X)
        routed = self._route_params(params)
        fitted: list[tuple[str, Any, Any]] = []
        blocks: list[Any] = []
        self.output_indices_ = {}
        offset = 0
        for name, transformer, selector, indices in resolved:
            if _is_drop(transformer):
                fitted.append((name, "drop", selector))
                self.output_indices_[name] = slice(offset, offset)
                continue
            if indices.size == 0:
                fitted.append(
                    (
                        name,
                        transformer
                        if _is_passthrough(transformer)
                        else clone(transformer),
                        selector,
                    )
                )
                self.output_indices_[name] = slice(offset, offset)
                continue
            selected = self._select(X, selector, indices)
            if _is_passthrough(transformer):
                fitted_transformer = "passthrough"
                block = selected
            else:
                fitted_transformer = clone(transformer)
                started_at = time.perf_counter()
                block = self._fit_transform(
                    fitted_transformer, selected, y, routed[name]
                )
                self._log_step(name, started_at)
            block = self._validate_block(self._weight(name, block), rows, name=name)
            blocks.append(block)
            fitted.append((name, fitted_transformer, selector))
            width = int(block.shape[1])
            self.output_indices_[name] = slice(offset, offset + width)
            offset += width
        self.transformers_ = fitted
        self._resolved_indices = {name: indices for name, _, _, indices in resolved}
        self.output_indices_.setdefault("remainder", slice(offset, offset))
        return self._stack(blocks, rows, fit=True)

    def fit(self, X: Any, y: Any = None, **params: Any) -> ColumnTransformer:
        """Fit all selected transformers."""
        self.fit_transform(X, y, **params)
        return self

    def transform(self, X: Any) -> Any:
        """Transform selected columns and concatenate their outputs."""
        check_is_fitted(self, ("transformers_", "_resolved_indices"))
        self._validate_transform_input(X)
        rows, _ = _shape(X)
        blocks = []
        for name, transformer, selector in self.transformers_:
            if _is_drop(transformer) or self._resolved_indices[name].size == 0:
                continue
            selected = self._select(X, selector, self._resolved_indices[name])
            block = (
                selected
                if _is_passthrough(transformer)
                else transformer.transform(selected)
            )
            blocks.append(
                self._validate_block(self._weight(name, block), rows, name=name)
            )
        return self._stack(blocks, rows, fit=False)

    def get_feature_names_out(self, input_features: Any = None) -> NDArray[Any]:
        """Return output feature names in concatenation order."""
        check_is_fitted(self, ("transformers_", "_resolved_indices"))
        if input_features is None:
            names = getattr(
                self,
                "feature_names_in_",
                np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object),
            )
        else:
            names = np.asarray(input_features, dtype=object)
            if names.ndim != 1 or names.size != self.n_features_in_:
                raise ValueError(
                    "input_features must contain one name per input feature"
                )
            if hasattr(self, "feature_names_in_") and not np.array_equal(
                names, self.feature_names_in_
            ):
                raise ValueError("input_features must match feature_names_in_")
        output: list[str] = []
        for name, transformer, _ in self.transformers_:
            if _is_drop(transformer) or self._resolved_indices[name].size == 0:
                continue
            selected_names = names[self._resolved_indices[name]]
            if _is_passthrough(transformer):
                block_names = selected_names
            elif hasattr(transformer, "get_feature_names_out"):
                block_names = transformer.get_feature_names_out(selected_names)
            else:
                raise AttributeError(
                    f"transformer {name!r} does not provide get_feature_names_out"
                )
            output.extend(
                self._format_feature_name(name, str(value)) for value in block_names
            )
        if self.verbose_feature_names_out is False and len(output) != len(set(output)):
            raise ValueError(
                "output feature names are not unique; enable verbose names"
            )
        return np.asarray(output, dtype=object)

    def _format_feature_name(self, transformer: str, feature: str) -> str:
        setting = self.verbose_feature_names_out
        if setting is True:
            return f"{transformer}__{feature}"
        if setting is False:
            return feature
        if isinstance(setting, str):
            return setting.format(transformer_name=transformer, feature_name=feature)
        value = setting(transformer, feature)
        if not isinstance(value, str):
            raise TypeError("verbose_feature_names_out callable must return strings")
        return value

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor, named-transformer, and nested parameters."""
        output = {
            "force_int_remainder_cols": self.force_int_remainder_cols,
            "n_jobs": self.n_jobs,
            "remainder": self.remainder,
            "sparse_threshold": self.sparse_threshold,
            "transformer_weights": self.transformer_weights,
            "transformers": self.transformers,
            "verbose": self.verbose,
            "verbose_feature_names_out": self.verbose_feature_names_out,
        }
        if not deep:
            return output
        for name, transformer, _ in self.transformers:
            output[name] = transformer
            if not isinstance(transformer, str) and hasattr(transformer, "get_params"):
                output.update(
                    (f"{name}__{key}", value)
                    for key, value in transformer.get_params(deep=True).items()
                )
        if not isinstance(self.remainder, str) and hasattr(
            self.remainder, "get_params"
        ):
            output.update(
                (f"remainder__{key}", value)
                for key, value in self.remainder.get_params(deep=True).items()
            )
        return output

    def set_params(self, **params: Any) -> ColumnTransformer:
        """Set constructor, named-transformer, and nested parameters."""
        if not params:
            return self
        constructor_names = set(self.get_params(deep=False))
        for name in constructor_names & params.keys():
            setattr(self, name, params.pop(name))
        transformer_names = {name for name, _, _ in self.transformers}
        replacements = transformer_names & params.keys()
        if replacements:
            self.transformers = [
                (
                    name,
                    params.pop(name) if name in replacements else transformer,
                    columns,
                )
                for name, transformer, columns in self.transformers
            ]
        for key, value in params.items():
            name, separator, parameter = key.partition("__")
            if not separator or name not in transformer_names | {"remainder"}:
                raise ValueError(f"Invalid parameter {key!r} for ColumnTransformer")
            transformer = (
                self.remainder
                if name == "remainder"
                else next(
                    transformer
                    for transformer_name, transformer, _ in self.transformers
                    if transformer_name == name
                )
            )
            if isinstance(transformer, str) or not hasattr(transformer, "set_params"):
                raise ValueError(f"transformer {name!r} does not accept parameters")
            transformer.set_params(**{parameter: value})
        return self


def make_column_transformer(
    *transformers: tuple[Any, Any],
    remainder: str | Any = "drop",
    sparse_threshold: float = 0.3,
    n_jobs: int | None = None,
    verbose: bool = False,
    verbose_feature_names_out: bool | str | Callable[[str, str], str] = True,
    force_int_remainder_cols: Any = "deprecated",
) -> ColumnTransformer:
    """Construct a ColumnTransformer with automatically generated names."""
    return ColumnTransformer(
        _name_estimators(transformers),
        remainder=remainder,
        sparse_threshold=sparse_threshold,
        n_jobs=n_jobs,
        verbose=verbose,
        verbose_feature_names_out=verbose_feature_names_out,
        force_int_remainder_cols=force_int_remainder_cols,
    )


__all__ = ["ColumnTransformer", "make_column_transformer"]
