"""Dataset splitting and cross-validation iterators."""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._utils import (
    check_consistent_length,
    check_random_state,
    indexable,
    num_samples,
    safe_indexing,
)


def _validate_size(value: Any, samples: int, name: str, *, test: bool) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an int, float, or None")
    if isinstance(value, (int, np.integer)):
        if not 0 < value < samples:
            raise ValueError(
                f"{name}={value} must be positive and smaller than {samples}"
            )
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not 0 < value < 1:
            raise ValueError(f"{name}={value} must be between 0 and 1")
        sized = math.ceil(value * samples) if test else math.floor(value * samples)
        if sized == 0:
            raise ValueError(
                f"{name}={value} results in an empty split for {samples} samples"
            )
        return sized
    raise TypeError(f"{name} must be an int, float, or None")


def _validate_shuffle_split(
    samples: int, test_size: Any, train_size: Any
) -> tuple[int, int]:
    if test_size is None and train_size is None:
        test_size = 0.25
    test = _validate_size(test_size, samples, "test_size", test=True)
    train = _validate_size(train_size, samples, "train_size", test=False)
    if test is None:
        test = samples - train
    if train is None:
        train = samples - test
    if train + test > samples:
        raise ValueError(
            f"The train_size and test_size result in {train + test} samples, "
            f"which exceeds the available {samples} samples"
        )
    if train == 0:
        raise ValueError("The resulting train set will be empty")
    return train, test


def _encode_stratification(y: Any) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    target = np.asarray(y)
    if target.ndim != 1:
        raise ValueError("stratify and StratifiedKFold require one-dimensional targets")
    if target.size == 0:
        raise ValueError("stratification requires at least one sample")
    try:
        _, encoded = np.unique(target, return_inverse=True)
    except TypeError as error:
        raise TypeError("stratification labels must have a consistent type") from error
    encoded = np.asarray(encoded, dtype=np.intp)
    return encoded, np.bincount(encoded)


def _stratified_shuffle_indices(
    y: Any, train: int, test: int, random_state: Any
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    encoded, counts = _encode_stratification(y)
    if np.min(counts) < 2:
        raise ValueError(
            "The least populated class in y has only 1 member, which is too few"
        )
    if train < counts.size:
        raise ValueError(
            "The train_size must be greater than or equal to the number of classes"
        )
    if test < counts.size:
        raise ValueError(
            "The test_size must be greater than or equal to the number of classes"
        )
    rng = check_random_state(random_state)
    train_counts = _approximate_mode(counts, train, rng)
    remaining = counts - train_counts
    test_counts = _approximate_mode(remaining, test, rng)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for class_index in range(counts.size):
        permutation = rng.permutation(np.flatnonzero(encoded == class_index))
        train_indices.extend(permutation[: train_counts[class_index]])
        test_indices.extend(
            permutation[
                train_counts[class_index] : train_counts[class_index]
                + test_counts[class_index]
            ]
        )
    return rng.permutation(train_indices), rng.permutation(test_indices)


def _approximate_mode(
    counts: NDArray[np.intp], draws: int, rng: np.random.RandomState
) -> NDArray[np.intp]:
    continuous = counts / counts.sum() * draws
    result = np.floor(continuous).astype(np.intp)
    remaining = draws - result.sum()
    if remaining:
        remainders = continuous - result
        for value in np.sort(np.unique(remainders))[::-1]:
            candidates = np.flatnonzero(remainders == value)
            selected = rng.choice(
                candidates, size=min(candidates.size, remaining), replace=False
            )
            result[selected] += 1
            remaining -= selected.size
            if remaining == 0:
                break
    return result


def train_test_split(
    *arrays: Any,
    test_size: Any = None,
    train_size: Any = None,
    random_state: Any = None,
    shuffle: bool = True,
    stratify: Any = None,
) -> list[Any]:
    """Split arrays into random train and test subsets."""
    if not arrays:
        raise ValueError("At least one array is required")
    if not isinstance(shuffle, (bool, np.bool_)):
        raise TypeError("shuffle must be bool")
    indexed = indexable(*arrays)
    samples = num_samples(indexed[0])
    train, test = _validate_shuffle_split(samples, test_size, train_size)
    if not shuffle:
        if stratify is not None:
            raise ValueError(
                "Stratified train/test split is not implemented for shuffle=False"
            )
        train_indices = np.arange(train, dtype=np.intp)
        test_indices = np.arange(train, train + test, dtype=np.intp)
    elif stratify is not None:
        check_consistent_length(indexed[0], stratify)
        train_indices, test_indices = _stratified_shuffle_indices(
            stratify, train, test, random_state
        )
    else:
        permutation = check_random_state(random_state).permutation(samples)
        test_indices = permutation[:test]
        train_indices = permutation[test : test + train]
    output: list[Any] = []
    for array in indexed:
        output.extend(
            (safe_indexing(array, train_indices), safe_indexing(array, test_indices))
        )
    return output


class BaseCrossValidator(ABC):
    """Base class for cross-validation splitters."""

    @abstractmethod
    def split(
        self, X: Any, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[Any, Any]]:
        """Yield train and test indices."""

    @abstractmethod
    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Return the number of folds."""


class _BaseKFold(BaseCrossValidator):
    def __init__(self, n_splits: int, *, shuffle: bool, random_state: Any) -> None:
        if isinstance(n_splits, (bool, np.bool_)) or not isinstance(
            n_splits, (int, np.integer)
        ):
            raise TypeError("n_splits must be an integer")
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if not isinstance(shuffle, (bool, np.bool_)):
            raise TypeError("shuffle must be bool")
        if not shuffle and random_state is not None:
            raise ValueError("Setting random_state has no effect when shuffle=False")
        if shuffle:
            check_random_state(random_state)
        self.n_splits = int(n_splits)
        self.shuffle = bool(shuffle)
        self.random_state = random_state

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        del X, y, groups
        return self.n_splits

    def _validate_samples(self, X: Any, y: Any, groups: Any) -> int:
        samples = check_consistent_length(X, y, groups)
        if self.n_splits > samples:
            raise ValueError(
                f"Cannot have number of splits n_splits={self.n_splits} greater "
                f"than the number of samples: n_samples={samples}."
            )
        return samples

    def split(
        self, X: Any, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[Any, Any]]:
        samples = self._validate_samples(X, y, groups)
        for test_indices in self._iter_test_indices(X, y, groups, samples):
            test_mask = np.zeros(samples, dtype=bool)
            test_mask[test_indices] = True
            yield np.flatnonzero(~test_mask), np.flatnonzero(test_mask)

    @abstractmethod
    def _iter_test_indices(
        self, X: Any, y: Any, groups: Any, samples: int
    ) -> Iterable[NDArray[np.intp]]:
        pass


class KFold(_BaseKFold):
    """K-fold cross-validator with optional deterministic shuffling."""

    def __init__(
        self, n_splits: int = 5, *, shuffle: bool = False, random_state: Any = None
    ) -> None:
        super().__init__(n_splits, shuffle=shuffle, random_state=random_state)

    def _iter_test_indices(
        self, X: Any, y: Any, groups: Any, samples: int
    ) -> Iterable[NDArray[np.intp]]:
        del X, y, groups
        indices = np.arange(samples, dtype=np.intp)
        if self.shuffle:
            check_random_state(self.random_state).shuffle(indices)
        fold_sizes = np.full(self.n_splits, samples // self.n_splits, dtype=np.intp)
        fold_sizes[: samples % self.n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            yield indices[current : current + fold_size]
            current += fold_size


class StratifiedKFold(_BaseKFold):
    """K-fold cross-validator preserving class proportions."""

    def __init__(
        self, n_splits: int = 5, *, shuffle: bool = False, random_state: Any = None
    ) -> None:
        super().__init__(n_splits, shuffle=shuffle, random_state=random_state)

    def _iter_test_indices(
        self, X: Any, y: Any, groups: Any, samples: int
    ) -> Iterable[NDArray[np.intp]]:
        del X, groups
        if y is None:
            raise ValueError("StratifiedKFold requires y")
        encoded, counts = _encode_stratification(y)
        if np.all(self.n_splits > counts):
            raise ValueError(
                f"n_splits={self.n_splits} cannot be greater than the number of "
                "members in each class."
            )
        if self.n_splits > np.min(counts):
            warnings.warn(
                f"The least populated class in y has only {np.min(counts)} members, "
                f"which is less than n_splits={self.n_splits}.",
                UserWarning,
                stacklevel=3,
            )
        allocation = np.asarray(
            [
                np.bincount(
                    np.sort(encoded)[fold :: self.n_splits],
                    minlength=counts.size,
                )
                for fold in range(self.n_splits)
            ]
        )
        test_folds = np.empty(samples, dtype=np.intp)
        rng = check_random_state(self.random_state)
        for class_index in range(counts.size):
            folds = np.arange(self.n_splits).repeat(allocation[:, class_index])
            if self.shuffle:
                rng.shuffle(folds)
            test_folds[encoded == class_index] = folds
        return (np.flatnonzero(test_folds == fold) for fold in range(self.n_splits))


__all__ = ["BaseCrossValidator", "KFold", "StratifiedKFold", "train_test_split"]
