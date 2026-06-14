"""Internal indexing and random-state helpers for model selection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


def num_samples(value: Any) -> int:
    """Return the number of rows in an array-like object."""
    if value is None:
        raise TypeError("Expected an array-like object, got None")
    if hasattr(value, "fit") and callable(value.fit):
        raise TypeError("Expected an array-like object, got an estimator")
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) == 0:
            raise TypeError("Singleton array cannot be considered a valid collection")
        if isinstance(shape[0], (int, np.integer)):
            return int(shape[0])
    if hasattr(value, "__len__"):
        return len(value)
    raise TypeError(f"Expected an array-like object, got {type(value).__name__}")


def check_consistent_length(*arrays: Any) -> int:
    """Validate that all non-None arrays contain the same number of rows."""
    lengths = [num_samples(array) for array in arrays if array is not None]
    if not lengths:
        return 0
    if len(set(lengths)) != 1:
        raise ValueError(
            f"Found input variables with inconsistent numbers of samples: {lengths}"
        )
    return lengths[0]


def safe_indexing(value: Any, indices: NDArray[np.integer[Any]]) -> Any:
    """Select rows while preserving common input container types."""
    indices = np.asarray(indices, dtype=np.intp)
    if hasattr(value, "iloc"):
        return value.iloc[indices]
    if getattr(value, "format", None) is not None:
        return value[indices]
    if hasattr(value, "take"):
        try:
            return value.take(indices, axis=0)
        except TypeError:
            return value.take(indices)
    if isinstance(value, tuple):
        return tuple(value[index] for index in indices)
    if isinstance(value, list):
        return [value[index] for index in indices]
    return np.asarray(value)[indices]


def check_random_state(seed: Any) -> np.random.RandomState:
    """Return a legacy RandomState compatible with scikit-learn splitters."""
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_)):
        return np.random.RandomState(int(seed))
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        f"{seed!r} cannot be used to seed a numpy.random.RandomState instance"
    )


def indexable(*arrays: Any) -> tuple[Any, ...]:
    """Convert non-indexable iterables and validate matching lengths."""
    result = tuple(
        None
        if array is None
        else array
        if hasattr(array, "__getitem__") or hasattr(array, "iloc")
        else np.asarray(list(array))
        for array in arrays
    )
    check_consistent_length(*result)
    return result


def is_arraylike(value: Any, samples: int) -> bool:
    """Return whether a fit parameter should be sliced with a training fold."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence | np.ndarray):
        return (
            hasattr(value, "shape") and getattr(value, "shape", (None,))[0] == samples
        )
    try:
        return num_samples(value) == samples
    except TypeError:
        return False
