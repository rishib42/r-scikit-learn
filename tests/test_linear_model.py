import numpy as np
import pytest
from rsklearn.linear_model import (
    ConvergenceWarning,
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from rsklearn.model_selection import cross_val_score
from rsklearn.pipeline import make_pipeline
from rsklearn.preprocessing import StandardScaler


def test_linear_regression_recovers_coefficients_and_multioutput():
    X = np.asarray([[1.0, 2.0], [2.0, 0.0], [3.0, 1.0], [4.0, 3.0]])
    y = 1.0 + X @ np.asarray([2.0, 1.5])
    model = LinearRegression().fit(X, y)
    np.testing.assert_allclose(model.coef_, [2.0, 1.5])
    assert model.intercept_ == pytest.approx(1.0)
    np.testing.assert_allclose(model.predict(X), y)
    assert model.score(X, y) == pytest.approx(1.0)

    targets = np.column_stack((y, 4.0 + X @ np.asarray([-1.0, 2.0])))
    model.fit(X, targets)
    assert model.coef_.shape == (2, 2)
    assert model.intercept_.shape == (2,)
    np.testing.assert_allclose(model.predict(X), targets)


def test_linear_regression_supports_weights_rank_deficiency_and_no_intercept():
    X = np.asarray([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [100.0, 200.0]])
    y = np.asarray([2.0, 4.0, 6.0, -100.0])
    model = LinearRegression().fit(X, y, sample_weight=[1, 1, 1, 0])
    np.testing.assert_allclose(model.predict(X[:3]), y[:3], atol=1e-10)
    no_intercept = LinearRegression(fit_intercept=False).fit([[1], [2]], [2, 4])
    assert no_intercept.intercept_ == 0
    np.testing.assert_allclose(no_intercept.coef_, [2])


def test_ridge_regularizes_coefficients_and_supports_multioutput():
    X = np.asarray([[1.0, 2.0], [2.0, 0.0], [3.0, 1.0], [4.0, 3.0]])
    y = 1.0 + X @ np.asarray([2.0, 1.5])
    linear = LinearRegression().fit(X, y)
    ridge = Ridge(alpha=10).fit(X, y)
    assert np.linalg.norm(ridge.coef_) < np.linalg.norm(linear.coef_)
    np.testing.assert_array_equal(ridge.n_iter_, [1])
    assert ridge.predict(X).shape == y.shape


def test_lasso_and_elastic_net_produce_sparse_multioutput_models():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(200, 8))
    y = 2.0 + X[:, 0] * 3.0 - X[:, 3] * 1.5
    lasso = Lasso(alpha=0.2, tol=1e-8, max_iter=5000).fit(X, y)
    assert np.count_nonzero(lasso.coef_) <= 2
    assert lasso.n_iter_ > 0
    assert lasso.dual_gap_ >= 0
    targets = np.column_stack((y, -y))
    elastic = ElasticNet(alpha=0.1, l1_ratio=0.7, tol=1e-8).fit(X, targets)
    assert elastic.coef_.shape == (2, 8)
    assert elastic.intercept_.shape == (2,)
    assert elastic.predict(X).shape == targets.shape


def test_lasso_supports_weights_positive_constraint_and_convergence_warning():
    X = np.asarray([[0.0], [1.0], [2.0], [10.0]])
    y = np.asarray([0.0, -1.0, -2.0, 100.0])
    weighted = Lasso(alpha=0.01).fit(X, y, sample_weight=[1, 1, 1, 0])
    assert weighted.coef_[0] < 0
    positive = Lasso(alpha=0.01, positive=True).fit(X[:3], y[:3])
    assert positive.coef_[0] == 0
    with pytest.warns(ConvergenceWarning):
        Lasso(alpha=0.001, max_iter=1, tol=1e-15).fit(np.column_stack((X, X**2)), y)


def test_logistic_regression_binary_probabilities_labels_and_weights():
    X = np.asarray([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    y = np.asarray(["negative"] * 3 + ["positive"] * 3)
    model = LogisticRegression(max_iter=1000, tol=1e-7).fit(X, y)
    np.testing.assert_array_equal(model.predict(X), y)
    probabilities = model.predict_proba(X)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(np.exp(model.predict_log_proba(X)), probabilities)
    assert model.coef_.shape == (1, 1)
    assert model.intercept_.shape == (1,)
    assert model.score(X, y) == 1.0
    weighted = LogisticRegression(max_iter=1000).fit(
        X, y, sample_weight=[1, 1, 1, 1, 1, 5]
    )
    assert weighted.predict_proba([[0]])[0, 1] > model.predict_proba([[0]])[0, 1]


def test_logistic_regression_multiclass_pipeline_and_cross_validation():
    X = np.asarray(
        [
            [-3, -3],
            [-2, -2],
            [-3, -2],
            [3, -3],
            [2, -2],
            [3, -2],
            [0, 3],
            [0, 2],
            [1, 3],
        ],
        dtype=float,
    )
    y = np.repeat(["left", "right", "top"], 3)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, tol=1e-6))
    model.fit(X, y)
    np.testing.assert_array_equal(model.predict(X), y)
    assert model.predict_proba(X).shape == (9, 3)
    scores = cross_val_score(model, X, y, cv=3, scoring="accuracy")
    assert scores.shape == (3,)


def test_linear_models_validate_unimplemented_and_invalid_configuration():
    with pytest.raises(NotImplementedError, match="positive"):
        LinearRegression(positive=True).fit([[1]], [1])
    with pytest.raises(NotImplementedError, match="solver"):
        Ridge(solver="saga").fit([[1]], [1])
    with pytest.raises(ValueError, match="alpha"):
        Ridge(alpha=-1).fit([[1]], [1])
    with pytest.raises(ValueError, match="require solver"):
        LogisticRegression(penalty="l1").fit([[1], [2]], [0, 1])
    with pytest.raises(ValueError, match="l1_ratio"):
        LogisticRegression(penalty="elasticnet", solver="saga").fit([[1], [2]], [0, 1])
    with pytest.raises(NotImplementedError, match="binary"):
        LogisticRegression(penalty="l1", solver="saga").fit([[1], [2], [3]], [0, 1, 2])
    with pytest.raises(NotImplementedError, match="selection"):
        Lasso(selection="random").fit([[1], [2]], [1, 2])
    with pytest.raises(ValueError, match="l1_ratio"):
        ElasticNet(l1_ratio=2).fit([[1], [2]], [1, 2])
    with pytest.raises(ValueError, match="two classes"):
        LogisticRegression().fit([[1], [2]], [0, 0])
    with pytest.raises(ValueError, match="not fitted"):
        LinearRegression().predict([[1]])
