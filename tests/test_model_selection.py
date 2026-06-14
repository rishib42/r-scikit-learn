import numpy as np
import pytest
from rsklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from rsklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from rsklearn.pipeline import make_pipeline
from rsklearn.preprocessing import StandardScaler


class MajorityClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y, sample_weight=None):
        del X
        labels, counts = np.unique(y, return_counts=True)
        if sample_weight is not None:
            counts = np.asarray(
                [
                    np.sum(np.asarray(sample_weight)[np.asarray(y) == label])
                    for label in labels
                ]
            )
        self.class_ = labels[np.argmax(counts)]
        self.classes_ = labels
        self.fit_samples_ = len(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.class_)


class MeanRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, fail_below=None):
        self.fail_below = fail_below

    def fit(self, X, y):
        if self.fail_below is not None and np.min(X) < self.fail_below:
            raise RuntimeError("requested failure")
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class UnsupervisedScorer(BaseEstimator):
    def fit(self, X):
        self.mean_ = float(np.mean(X))
        return self

    def score(self, X):
        return -abs(float(np.mean(X)) - self.mean_)


def test_train_test_split_is_deterministic_and_preserves_container_types():
    X = [[index, index + 1] for index in range(10)]
    y = tuple(range(10))
    first = train_test_split(X, y, test_size=0.3, random_state=42)
    second = train_test_split(X, y, test_size=0.3, random_state=42)
    assert first == second
    X_train, X_test, y_train, y_test = first
    assert isinstance(X_train, list)
    assert isinstance(y_train, tuple)
    assert len(X_train) == len(y_train) == 7
    assert len(X_test) == len(y_test) == 3


def test_train_test_split_supports_sequential_stratified_and_sparse_data():
    X = np.arange(24).reshape(12, 2)
    y = np.repeat([0, 1, 2], 4)
    sequential = train_test_split(X, test_size=3, shuffle=False)
    np.testing.assert_array_equal(sequential[0], X[:9])
    np.testing.assert_array_equal(sequential[1], X[9:])
    _, _, y_train, y_test = train_test_split(
        X, y, test_size=6, random_state=0, stratify=y
    )
    np.testing.assert_array_equal(np.bincount(y_train), [2, 2, 2])
    np.testing.assert_array_equal(np.bincount(y_test), [2, 2, 2])

    sparse = pytest.importorskip("scipy.sparse").csr_matrix(X)
    sparse_train, sparse_test = train_test_split(sparse, test_size=3, random_state=0)
    assert sparse_train.format == sparse_test.format == "csr"
    assert sparse_train.shape == (9, 2)
    assert sparse_test.shape == (3, 2)


def test_train_test_split_rejects_invalid_sizes_and_stratification():
    with pytest.raises(ValueError, match="At least one"):
        train_test_split()
    with pytest.raises(ValueError, match="inconsistent"):
        train_test_split([1, 2], [1])
    with pytest.raises(ValueError, match="exceeds"):
        train_test_split(range(10), train_size=8, test_size=3)
    with pytest.raises(ValueError, match="shuffle=False"):
        train_test_split(range(10), shuffle=False, stratify=[0, 1] * 5)
    with pytest.raises(ValueError, match="too few"):
        train_test_split(range(6), stratify=[0, 0, 0, 1, 1, 2])


def test_kfold_yields_complete_disjoint_folds_and_validates_configuration():
    folds = list(KFold(3).split(np.arange(10)))
    np.testing.assert_array_equal([len(test) for _, test in folds], [4, 3, 3])
    np.testing.assert_array_equal(
        np.concatenate([test for _, test in folds]), np.arange(10)
    )
    for train, test in folds:
        assert not np.intersect1d(train, test).size
    first = list(KFold(3, shuffle=True, random_state=42).split(np.arange(10)))
    second = list(KFold(3, shuffle=True, random_state=42).split(np.arange(10)))
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left[0], right[0])
        np.testing.assert_array_equal(left[1], right[1])
    with pytest.raises(ValueError, match="no effect"):
        KFold(random_state=0)
    with pytest.raises(ValueError, match="greater"):
        list(KFold(6).split(np.arange(5)))


def test_stratified_kfold_balances_classes_and_warns_for_small_classes():
    y = np.repeat([0, 1], 6)
    folds = list(StratifiedKFold(3).split(np.zeros((12, 1)), y))
    for _, test in folds:
        np.testing.assert_array_equal(np.bincount(y[test]), [2, 2])
    with pytest.warns(UserWarning, match="least populated"):
        folds = list(StratifiedKFold(3).split(np.zeros((5, 1)), [0, 0, 1, 1, 1]))
    assert len(folds) == 3
    with pytest.raises(ValueError, match="each class"):
        list(StratifiedKFold(4).split(np.zeros((6, 1)), [0, 0, 0, 1, 1, 1]))


def test_cross_val_score_supports_default_scores_strings_pipeline_and_callable():
    X = np.arange(24, dtype=float).reshape(12, 2)
    y_class = np.repeat([0, 1], 6)
    classifier = make_pipeline(StandardScaler(), MajorityClassifier())
    scores = cross_val_score(classifier, X, y_class, scoring="accuracy", cv=3)
    np.testing.assert_allclose(scores, [0.5, 0.5, 0.5])
    callable_scores = cross_val_score(
        classifier,
        X,
        y_class,
        scoring=lambda estimator, X_test, y_test: estimator.score(X_test, y_test),
        cv=3,
    )
    np.testing.assert_array_equal(callable_scores, scores)

    y_regression = np.arange(12, dtype=float)
    negative_mse = cross_val_score(
        MeanRegressor(), X, y_regression, scoring="neg_mean_squared_error", cv=3
    )
    assert np.all(negative_mse < 0)


def test_cross_val_score_slices_fit_params_and_handles_failures(capsys):
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.repeat([0, 1], 5)
    weights = np.arange(1, 11, dtype=float)
    scores = cross_val_score(
        MajorityClassifier(),
        X,
        y,
        cv=2,
        params={"sample_weight": weights},
        verbose=1,
    )
    assert scores.shape == (2,)
    assert "[CV] fold=1" in capsys.readouterr().out

    with pytest.warns(RuntimeWarning, match="score set"):
        scores = cross_val_score(
            MeanRegressor(fail_below=2),
            X,
            np.arange(10),
            cv=[(np.arange(1, 10), np.asarray([0])), (np.arange(5), np.arange(5, 10))],
            error_score=-1,
        )
    assert scores[0] != -1
    assert scores[1] == -1
    with pytest.raises(NotImplementedError, match="parallel"):
        cross_val_score(MeanRegressor(), X, np.arange(10), n_jobs=2)


def test_cross_val_score_supports_unsupervised_estimators():
    scores = cross_val_score(UnsupervisedScorer(), np.arange(12), cv=3)
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))
