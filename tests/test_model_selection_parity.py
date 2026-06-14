import numpy as np
import pytest
from rsklearn.base import BaseEstimator, RegressorMixin
from rsklearn.model_selection import KFold, StratifiedKFold, train_test_split
from rsklearn.model_selection import cross_val_score as rsklearn_cross_val_score

sklearn_model_selection = pytest.importorskip("sklearn.model_selection")


class MeanRegressor(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"test_size": 0.3, "random_state": 42},
        {"train_size": 0.6, "random_state": 7},
        {"train_size": 5, "test_size": 3, "random_state": 0},
        {"test_size": 3, "shuffle": False},
    ],
)
def test_train_test_split_matches_scikit_learn(kwargs):
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    ours = train_test_split(X, y, **kwargs)
    theirs = sklearn_model_selection.train_test_split(X, y, **kwargs)
    for left, right in zip(ours, theirs, strict=True):
        np.testing.assert_array_equal(left, right)


def test_stratified_train_test_split_matches_scikit_learn():
    X = np.arange(48).reshape(24, 2)
    y = np.repeat(["alpha", "beta", "gamma"], 8)
    ours = train_test_split(X, y, test_size=9, random_state=42, stratify=y)
    theirs = sklearn_model_selection.train_test_split(
        X, y, test_size=9, random_state=42, stratify=y
    )
    for left, right in zip(ours, theirs, strict=True):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("shuffle", [False, True])
def test_kfold_matches_scikit_learn(shuffle):
    kwargs = {"n_splits": 4, "shuffle": shuffle}
    if shuffle:
        kwargs["random_state"] = 42
    ours = KFold(**kwargs).split(np.arange(11))
    theirs = sklearn_model_selection.KFold(**kwargs).split(np.arange(11))
    for left, right in zip(ours, theirs, strict=True):
        np.testing.assert_array_equal(left[0], right[0])
        np.testing.assert_array_equal(left[1], right[1])


@pytest.mark.parametrize("shuffle", [False, True])
def test_stratified_kfold_matches_scikit_learn(shuffle):
    X = np.zeros((18, 1))
    y = np.asarray(["first", "second", "third"] * 6)
    kwargs = {"n_splits": 4, "shuffle": shuffle}
    if shuffle:
        kwargs["random_state"] = 42
    ours = StratifiedKFold(**kwargs).split(X, y)
    theirs = sklearn_model_selection.StratifiedKFold(**kwargs).split(X, y)
    for left, right in zip(ours, theirs, strict=True):
        np.testing.assert_array_equal(left[0], right[0])
        np.testing.assert_array_equal(left[1], right[1])


@pytest.mark.parametrize("scoring", [None, "neg_mean_squared_error", "r2"])
def test_cross_val_score_matches_scikit_learn(scoring):
    X = np.arange(24, dtype=float).reshape(12, 2)
    y = np.arange(12, dtype=float)
    ours = rsklearn_cross_val_score(MeanRegressor(), X, y, scoring=scoring, cv=3)
    theirs = sklearn_model_selection.cross_val_score(
        MeanRegressor(), X, y, scoring=scoring, cv=3
    )
    np.testing.assert_allclose(ours, theirs)
