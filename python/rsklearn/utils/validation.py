"""Input and fitted-state validation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

_NO_VALIDATION = "no_validation"


try:
    from sklearn.exceptions import NotFittedError
except ImportError:

    class NotFittedError(ValueError, AttributeError):
        """Raised when an estimator method requires fitting first."""


def _estimator_name(estimator: Any) -> str:
    if estimator is None:
        return ""
    return estimator if isinstance(estimator, str) else type(estimator).__name__


def _is_sparse(value: Any) -> bool:
    try:
        from scipy import sparse
    except ImportError:
        return False
    return bool(sparse.issparse(value))


def _accepted_sparse_formats(
    accept_sparse: bool | str | list[str] | tuple[str, ...],
) -> tuple[str, ...] | None:
    if accept_sparse is True:
        return None
    if accept_sparse is False:
        return ()
    formats = (
        (accept_sparse,) if isinstance(accept_sparse, str) else tuple(accept_sparse)
    )
    valid = {"bsr", "coo", "csc", "csr", "dia", "dok", "lil"}
    if not formats or any(item not in valid for item in formats):
        raise ValueError(f"accept_sparse contains unsupported formats: {formats}")
    return formats


def _check_sparse_finite(array: Any, policy: bool | str, *, name: str) -> None:
    _check_finite(_sparse_stored_data(array), policy, name=name)


def _sparse_stored_data(array: Any) -> NDArray[Any]:
    data = getattr(array, "data", None)
    if isinstance(data, np.ndarray):
        return data
    return np.asarray(array.tocoo(copy=False).data)


def _check_sparse_array(
    array: Any,
    *,
    accept_sparse: bool | str | list[str] | tuple[str, ...],
    accept_large_sparse: bool,
    dtype: Any,
    copy: bool,
    force_writeable: bool,
    ensure_all_finite: bool | str,
    ensure_non_negative: bool,
    ensure_min_samples: int,
    ensure_min_features: int,
    prefix: str,
    name: str,
) -> Any:
    formats = _accepted_sparse_formats(accept_sparse)
    if formats == ():
        raise TypeError(
            f"{prefix}does not support sparse input; dense data is required"
        )
    result = array
    if np.iscomplexobj(result) and dtype == "numeric":
        raise ValueError("Complex data not supported")
    if formats is not None and result.format not in formats:
        result = result.asformat(formats[0])
    target_dtype = None if dtype in (None, "numeric") else dtype
    if dtype == "numeric" and result.dtype.kind in "OUSV":
        target_dtype = np.float64
    if target_dtype is not None and result.dtype != np.dtype(target_dtype):
        result = result.astype(target_dtype)
    elif copy:
        result = result.copy()
    if result.ndim != 2:
        raise ValueError(f"{prefix}expected a 2-dimensional sparse array")
    if result.shape[0] < ensure_min_samples:
        raise ValueError(
            f"Found array with {result.shape[0]} sample(s) (shape={result.shape}) "
            f"while a minimum of {ensure_min_samples} is required."
        )
    if result.shape[1] < ensure_min_features:
        raise ValueError(
            f"{result.shape[1]} feature(s) (shape={result.shape}) while a minimum "
            f"of {ensure_min_features} is required."
        )
    if not accept_large_sparse:
        for attribute in ("indices", "indptr"):
            values = getattr(result, attribute, None)
            if values is not None and values.dtype != np.dtype(np.int32):
                raise ValueError(
                    "Only sparse matrices with 32-bit integer indices are accepted"
                )
    _check_sparse_finite(result, ensure_all_finite, name=f"{prefix}{name}".strip())
    stored_data = _sparse_stored_data(result)
    if ensure_non_negative and stored_data.size and np.any(stored_data < 0):
        raise ValueError(f"{prefix}{name} contains negative values")
    writable_arrays = [
        values
        for values in (
            getattr(result, "data", None),
            getattr(result, "indices", None),
            getattr(result, "indptr", None),
        )
        if isinstance(values, np.ndarray)
    ]
    if force_writeable and any(
        not values.flags.writeable for values in writable_arrays
    ):
        result = result.copy()
    return result


def _check_finite(array: NDArray[Any], policy: bool | str, *, name: str) -> None:
    if policy is False:
        return
    if policy not in (True, "allow-nan"):
        raise ValueError("ensure_all_finite must be True, False, or 'allow-nan'")
    if array.dtype.kind not in "biufc":
        return
    try:
        if policy == "allow-nan":
            invalid = np.isinf(array).any()
        else:
            invalid = not np.isfinite(array).all()
    except TypeError as error:
        raise TypeError(f"{name} must contain numeric values") from error
    if invalid:
        allowed = "infinity" if policy == "allow-nan" else "NaN or infinity"
        raise ValueError(f"{name} contains {allowed}")


def check_array(
    array: Any,
    accept_sparse: bool | str | list[str] | tuple[str, ...] = False,
    *,
    accept_large_sparse: bool = True,
    dtype: Any = "numeric",
    order: str | None = None,
    copy: bool = False,
    force_writeable: bool = False,
    ensure_all_finite: bool | str = True,
    ensure_non_negative: bool = False,
    ensure_2d: bool = True,
    allow_nd: bool = False,
    ensure_min_samples: int = 1,
    ensure_min_features: int = 1,
    estimator: Any = None,
    input_name: str = "",
) -> Any:
    """Validate array-like input and return a NumPy array.

    SciPy sparse inputs are preserved or converted to an explicitly accepted
    sparse format. Sparse inputs are rejected unless ``accept_sparse`` is set.
    """
    name = input_name or "input"
    estimator_name = _estimator_name(estimator)
    prefix = f"{estimator_name} " if estimator_name else ""
    if _is_sparse(array):
        return _check_sparse_array(
            array,
            accept_sparse=accept_sparse,
            accept_large_sparse=accept_large_sparse,
            dtype=dtype,
            copy=copy,
            force_writeable=force_writeable,
            ensure_all_finite=ensure_all_finite,
            ensure_non_negative=ensure_non_negative,
            ensure_min_samples=ensure_min_samples,
            ensure_min_features=ensure_min_features,
            prefix=prefix,
            name=name,
        )
    original = np.asarray(array)
    allow_complex = dtype is None
    if dtype not in (None, "numeric"):
        try:
            allow_complex = np.issubdtype(np.dtype(dtype), np.complexfloating)
        except TypeError:
            allow_complex = False
    if np.iscomplexobj(original) and not allow_complex:
        raise ValueError("Complex data not supported")
    target_dtype = None if dtype is None else dtype
    if dtype == "numeric":
        target_dtype = None
    try:
        result = np.asarray(array, dtype=target_dtype, order=order)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{prefix}{name} argument must be a string or a number"
        ) from error
    if dtype == "numeric" and result.dtype.kind in "OUSV":
        try:
            result = result.astype(np.float64)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"{prefix}{name} argument must be a string or a number"
            ) from error
    if ensure_2d and result.ndim != 2:
        raise ValueError(
            f"{prefix}expected a 2-dimensional array, got {result.ndim}D. "
            "Reshape your data before passing it to the estimator."
        )
    if not allow_nd and result.ndim >= 3:
        raise ValueError(f"{prefix}found array with dimension {result.ndim} > 2")
    if result.ndim > 0 and result.shape[0] < ensure_min_samples:
        raise ValueError(
            f"Found array with {result.shape[0]} sample(s) (shape={result.shape}) "
            f"while a minimum of {ensure_min_samples} is required."
        )
    if ensure_2d and result.shape[1] < ensure_min_features:
        raise ValueError(
            f"{result.shape[1]} feature(s) (shape={result.shape}) while a minimum "
            f"of {ensure_min_features} is required."
        )
    _check_finite(result, ensure_all_finite, name=f"{prefix}{name}".strip())
    if ensure_non_negative and np.any(result < 0):
        raise ValueError(f"{prefix}{name} contains negative values")
    if copy:
        result = result.copy(order=order or "K")
    if force_writeable and not result.flags.writeable:
        result = result.copy(order=order or "K")
    return result


def check_X_y(
    X: Any,
    y: Any,
    accept_sparse: bool | str | list[str] | tuple[str, ...] = False,
    *,
    accept_large_sparse: bool = True,
    dtype: Any = "numeric",
    order: str | None = None,
    copy: bool = False,
    force_writeable: bool = False,
    ensure_all_finite: bool | str = True,
    ensure_2d: bool = True,
    allow_nd: bool = False,
    multi_output: bool = False,
    ensure_min_samples: int = 1,
    ensure_min_features: int = 1,
    y_numeric: bool = False,
    estimator: Any = None,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Validate matching feature and target arrays."""
    checked_X = check_array(
        X,
        accept_sparse=accept_sparse,
        accept_large_sparse=accept_large_sparse,
        dtype=dtype,
        order=order,
        copy=copy,
        force_writeable=force_writeable,
        ensure_all_finite=ensure_all_finite,
        ensure_2d=ensure_2d,
        allow_nd=allow_nd,
        ensure_min_samples=ensure_min_samples,
        ensure_min_features=ensure_min_features,
        estimator=estimator,
        input_name="X",
    )
    checked_y = check_array(
        y,
        dtype="numeric" if y_numeric else None,
        copy=copy,
        ensure_all_finite=ensure_all_finite,
        ensure_2d=multi_output,
        allow_nd=multi_output,
        ensure_min_samples=ensure_min_samples,
        ensure_min_features=1,
        estimator=estimator,
        input_name="y",
    )
    if not multi_output:
        checked_y = np.ravel(checked_y)
    if checked_X.shape[0] != checked_y.shape[0]:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: "
            f"[{checked_X.shape[0]}, {checked_y.shape[0]}]"
        )
    return checked_X, checked_y


def check_is_fitted(
    estimator: Any,
    attributes: str | list[str] | tuple[str, ...] | None = None,
    *,
    msg: str | None = None,
    all_or_any: Any = all,
) -> None:
    """Raise when an estimator has not learned fitted attributes."""
    if isinstance(estimator, type) or not hasattr(estimator, "fit"):
        raise TypeError("check_is_fitted requires an estimator instance with fit")
    if attributes is None and hasattr(estimator, "__sklearn_is_fitted__"):
        is_fitted = bool(estimator.__sklearn_is_fitted__())
    elif attributes is None:
        fitted = [
            name
            for name in vars(estimator)
            if name.endswith("_") and not name.startswith("__")
        ]
        is_fitted = bool(fitted)
    else:
        names = [attributes] if isinstance(attributes, str) else list(attributes)
        is_fitted = all_or_any(hasattr(estimator, name) for name in names)
    if not is_fitted:
        default = (
            "This {name} instance is not fitted yet. Call 'fit' with appropriate "
            "arguments before using this estimator."
        )
        raise NotFittedError((msg or default).format(name=type(estimator).__name__))


def _feature_names(X: Any) -> NDArray[Any] | None:
    columns = getattr(X, "columns", None)
    if columns is None:
        return None
    names = np.asarray(columns, dtype=object)
    if names.size and all(isinstance(name, str) for name in names):
        return names
    return None


def _is_no_validation(value: Any) -> bool:
    return isinstance(value, str) and value == _NO_VALIDATION


def validate_data(
    _estimator: Any,
    /,
    X: Any = _NO_VALIDATION,
    y: Any = _NO_VALIDATION,
    reset: bool = True,
    validate_separately: bool | tuple[dict[str, Any], dict[str, Any]] = False,
    skip_check_array: bool = False,
    **check_params: Any,
) -> Any:
    """Validate estimator input and manage feature-count metadata."""
    no_X = _is_no_validation(X)
    no_y = y is None or _is_no_validation(y)
    if no_X and no_y:
        raise ValueError("validate_data requires X, y, or both")
    if skip_check_array:
        checked = (X, y) if not no_y else X
    elif no_y:
        checked = check_array(X, estimator=_estimator, **check_params)
    elif no_X:
        checked = check_array(
            y, estimator=_estimator, ensure_2d=False, input_name="y", **check_params
        )
    elif validate_separately:
        if not isinstance(validate_separately, tuple) or len(validate_separately) != 2:
            raise TypeError("validate_separately must be a pair of parameter mappings")
        x_params, y_params = validate_separately
        checked = (
            check_array(X, estimator=_estimator, input_name="X", **x_params),
            check_array(y, estimator=_estimator, input_name="y", **y_params),
        )
    else:
        checked = check_X_y(X, y, estimator=_estimator, **check_params)

    checked_X = checked[0] if isinstance(checked, tuple) else checked
    if not no_X and getattr(checked_X, "ndim", 0) >= 2:
        feature_count = checked_X.shape[1]
        names = _feature_names(X)
        if reset:
            _estimator.n_features_in_ = feature_count
            if names is not None:
                _estimator.feature_names_in_ = names
            elif hasattr(_estimator, "feature_names_in_"):
                delattr(_estimator, "feature_names_in_")
        elif not hasattr(_estimator, "n_features_in_"):
            raise ValueError(
                f"{type(_estimator).__name__} has no recorded feature count; "
                "call fit before validation with reset=False"
            )
        elif feature_count != _estimator.n_features_in_:
            raise ValueError(
                f"X has {feature_count} features, but {type(_estimator).__name__} is "
                f"expecting {_estimator.n_features_in_} features as input."
            )
        elif hasattr(_estimator, "feature_names_in_"):
            if names is None:
                raise ValueError(
                    "input does not have feature names, but the estimator was fitted "
                    "with feature names"
                )
            if not np.array_equal(names, _estimator.feature_names_in_):
                raise ValueError("feature names must match those passed during fit")
    return checked


__all__ = [
    "NotFittedError",
    "check_array",
    "check_is_fitted",
    "check_X_y",
    "validate_data",
]
