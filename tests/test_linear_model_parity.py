import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from rsklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    _least_squares,
)
from rsklearn.linear_model._least_squares import _tall_solution_is_stable
from scipy import linalg

sklearn_linear = pytest.importorskip("sklearn.linear_model")


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("weighted", [False, True])
def test_linear_regression_matches_scikit_learn(fit_intercept, weighted):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 5))
    y = rng.normal(size=(100, 2))
    weights = rng.uniform(0.1, 2, size=100) if weighted else None
    ours = LinearRegression(fit_intercept=fit_intercept).fit(
        X, y, sample_weight=weights
    )
    theirs = sklearn_linear.LinearRegression(fit_intercept=fit_intercept).fit(
        X, y, sample_weight=weights
    )
    np.testing.assert_allclose(ours.coef_, theirs.coef_, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        ours.intercept_, theirs.intercept_, rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        ours.predict(X), theirs.predict(X), rtol=1e-8, atol=1e-10
    )


@pytest.mark.parametrize("perturbation", [1e-4, 1e-6, 1e-10, 0.0])
def test_tall_linear_regression_matches_svd_near_rank_deficiency(perturbation):
    rng = np.random.default_rng(99)
    independent = rng.normal(size=(10_000, 8))
    dependent = (
        independent[:, 0]
        + 2.0 * independent[:, 1]
        + perturbation * rng.normal(size=independent.shape[0])
    )
    X = np.ascontiguousarray(np.column_stack((independent, dependent)))
    y = rng.normal(size=X.shape[0])
    ours = LinearRegression(tol=1e-6).fit(X, y)
    x_mean = X.mean(axis=0)
    y_mean = y.mean()
    coefficients, _, rank, _ = linalg.lstsq(
        X - x_mean,
        y - y_mean,
        cond=1e-6,
        check_finite=False,
        lapack_driver="gelsd",
    )
    expected = X @ coefficients + y_mean - coefficients @ x_mean
    assert ours.rank_ == rank
    np.testing.assert_allclose(ours.predict(X), expected, rtol=1e-7, atol=5e-9)


@given(
    rows=st.integers(min_value=80, max_value=240),
    columns=st.integers(min_value=2, max_value=12),
    log_condition=st.floats(
        min_value=0.0, max_value=12.0, allow_nan=False, allow_infinity=False
    ),
    weighted=st.booleans(),
    fit_intercept=st.booleans(),
)
@settings(max_examples=40, deadline=None)
def test_linear_regression_matches_svd_across_condition_numbers(
    rows, columns, log_condition, weighted, fit_intercept
):
    rows = max(rows, 4 * columns)
    rng = np.random.default_rng(
        rows * 10_000 + columns * 100 + int(log_condition * 10) + int(weighted)
    )
    left, _ = np.linalg.qr(rng.normal(size=(rows, columns)))
    right, _ = np.linalg.qr(rng.normal(size=(columns, columns)))
    singular = np.geomspace(1.0, 10.0**-log_condition, columns)
    X = np.ascontiguousarray((left * singular) @ right.T)
    y = rng.normal(size=rows)
    weights = rng.uniform(0.1, 2.0, size=rows) if weighted else None
    ours = LinearRegression(tol=1e-10, fit_intercept=fit_intercept).fit(
        X, y, sample_weight=weights
    )
    reference_weights = np.ones(rows) if weights is None else weights
    if fit_intercept:
        x_mean = np.average(X, axis=0, weights=reference_weights)
        y_mean = np.average(y, weights=reference_weights)
    else:
        x_mean = np.zeros(columns)
        y_mean = 0.0
    root_weights = np.sqrt(reference_weights)
    coefficients, _, _, _ = linalg.lstsq(
        (X - x_mean) * root_weights[:, None],
        (y - y_mean) * root_weights,
        cond=1e-10,
        check_finite=False,
        lapack_driver="gelsd",
    )
    expected = X @ coefficients + y_mean - coefficients @ x_mean
    np.testing.assert_allclose(ours.predict(X), expected, rtol=1e-7, atol=1e-9)


def test_tall_solution_stability_gate_rejects_unsafe_spectra():
    assert _tall_solution_is_stable(np.asarray([10.0, 1.0]), 2, 1e-6)
    assert not _tall_solution_is_stable(np.asarray([10.0, 1e-10]), 2, 1e-6)
    assert _tall_solution_is_stable(np.asarray([10.0, 1.0, 0.0]), 2, 1e-6)
    assert not _tall_solution_is_stable(np.asarray([10.0, 1.0, 0.0]), 2, 1e-10)
    assert not _tall_solution_is_stable(np.asarray([0.0, 0.0]), 0, 1e-6)


@pytest.mark.parametrize(
    "log_condition, expects_fallback", [(2.0, False), (10.0, True)]
)
def test_tall_linear_regression_falls_back_only_when_numerically_unsafe(
    monkeypatch, log_condition, expects_fallback
):
    rng = np.random.default_rng(1234)
    left, _ = np.linalg.qr(rng.normal(size=(1_000, 5)))
    right, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    X = np.ascontiguousarray(
        (left * np.geomspace(1.0, 10.0**-log_condition, 5)) @ right.T
    )
    original = _least_squares.linalg.lstsq
    calls = 0

    def tracked_lstsq(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_least_squares.linalg, "lstsq", tracked_lstsq)
    LinearRegression(tol=1e-10).fit(X, rng.normal(size=X.shape[0]))
    assert bool(calls) is expects_fallback


@pytest.mark.parametrize("alpha", [0.0, 0.1, 10.0])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_ridge_matches_scikit_learn_svd(alpha, fit_intercept):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 6))
    y = rng.normal(size=80)
    ours = Ridge(alpha=alpha, fit_intercept=fit_intercept, solver="svd").fit(X, y)
    theirs = sklearn_linear.Ridge(
        alpha=alpha, fit_intercept=fit_intercept, solver="svd"
    ).fit(X, y)
    np.testing.assert_allclose(ours.coef_, theirs.coef_, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(ours.intercept_, theirs.intercept_, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(ours.predict(X), theirs.predict(X), rtol=1e-7, atol=1e-9)


def test_logistic_regression_matches_scikit_learn_predictions_and_probabilities():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(300, 4))
    logits = X @ np.asarray([1.5, -2.0, 0.7, 0.2]) + 0.3
    y = (logits + rng.normal(scale=0.4, size=300) > 0).astype(int)
    ours = LogisticRegression(C=2, max_iter=3000, tol=1e-7).fit(X, y)
    theirs = sklearn_linear.LogisticRegression(C=2, max_iter=3000, tol=1e-7).fit(X, y)
    assert np.mean(ours.predict(X) == theirs.predict(X)) > 0.98
    np.testing.assert_allclose(
        ours.predict_proba(X), theirs.predict_proba(X), rtol=0.08, atol=0.04
    )


def test_multiclass_logistic_regression_matches_scikit_learn():
    rng = np.random.default_rng(124)
    X = rng.normal(size=(2_000, 12))
    coefficients = rng.normal(size=(4, 12))
    y = np.argmax(X @ coefficients.T + rng.normal(scale=0.5, size=(2_000, 4)), axis=1)
    options = {"C": 0.7, "max_iter": 500, "tol": 1e-7}
    ours = LogisticRegression(**options).fit(X, y)
    theirs = sklearn_linear.LogisticRegression(**options).fit(X, y)
    assert np.mean(ours.predict(X) == theirs.predict(X)) > 0.995
    np.testing.assert_allclose(
        ours.predict_proba(X), theirs.predict_proba(X), rtol=0.02, atol=0.01
    )


@pytest.mark.parametrize(
    ("ours_type", "theirs_type", "kwargs"),
    [
        (Lasso, sklearn_linear.Lasso, {"alpha": 0.08}),
        (
            ElasticNet,
            sklearn_linear.ElasticNet,
            {"alpha": 0.08, "l1_ratio": 0.65},
        ),
    ],
)
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("weighted", [False, True])
def test_coordinate_descent_regressors_match_scikit_learn(
    ours_type, theirs_type, kwargs, fit_intercept, weighted
):
    rng = np.random.default_rng(90)
    X = rng.normal(size=(300, 10))
    coefficients = np.asarray([2.0, 0, -1.0, 0, 0.5, 0, 0, 0, 0, 0])
    y = 1.5 + X @ coefficients + rng.normal(scale=0.05, size=300)
    weights = rng.uniform(0.1, 2.0, size=300) if weighted else None
    common = dict(fit_intercept=fit_intercept, max_iter=10000, tol=1e-8, **kwargs)
    ours = ours_type(**common).fit(X, y, sample_weight=weights)
    theirs = theirs_type(**common).fit(X, y, sample_weight=weights)
    np.testing.assert_allclose(ours.coef_, theirs.coef_, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(ours.intercept_, theirs.intercept_, atol=2e-6)
    np.testing.assert_allclose(ours.predict(X), theirs.predict(X), atol=5e-6)


@pytest.mark.parametrize(
    ("penalty", "l1_ratio"),
    [("l1", None), ("elasticnet", 0.35)],
)
def test_sparse_logistic_regression_matches_scikit_learn(penalty, l1_ratio):
    rng = np.random.default_rng(81)
    X = rng.normal(size=(500, 12))
    logits = 2 * X[:, 0] - X[:, 3] + 0.4 * X[:, 7]
    y = (logits + rng.normal(scale=0.8, size=500) > 0).astype(int)
    options = dict(
        penalty=penalty,
        solver="saga",
        C=0.4,
        l1_ratio=l1_ratio,
        max_iter=10000,
        tol=1e-7,
        random_state=0,
    )
    ours = LogisticRegression(**options).fit(X, y)
    theirs = sklearn_linear.LogisticRegression(**options).fit(X, y)
    assert np.mean(ours.predict(X) == theirs.predict(X)) > 0.99
    np.testing.assert_allclose(
        ours.predict_proba(X), theirs.predict_proba(X), atol=0.02
    )
    assert np.count_nonzero(ours.coef_) == np.count_nonzero(theirs.coef_)
