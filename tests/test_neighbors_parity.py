import numpy as np
import pytest
from rsklearn.neighbors import KNeighborsClassifier

sklearn_neighbors = pytest.importorskip("sklearn.neighbors")


@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize(
    ("metric", "p"),
    [
        ("minkowski", 2),
        ("minkowski", 1),
        ("euclidean", 2),
        ("manhattan", 1),
    ],
)
def test_kneighbors_classifier_matches_scikit_learn(metric, p, weights):
    rng = np.random.default_rng(20260616)
    X = rng.normal(size=(80, 5))
    y = rng.integers(0, 3, size=80, dtype=np.int64)
    query = rng.normal(size=(12, 5))
    options = {
        "n_neighbors": 5,
        "weights": weights,
        "algorithm": "brute",
        "metric": metric,
        "p": p,
    }
    ours = KNeighborsClassifier(**options).fit(X, y)
    theirs = sklearn_neighbors.KNeighborsClassifier(**options).fit(X, y)
    np.testing.assert_array_equal(ours.predict(query), theirs.predict(query))
    np.testing.assert_allclose(ours.predict_proba(query), theirs.predict_proba(query))
    ours_distances, ours_indices = ours.kneighbors(query)
    their_distances, their_indices = theirs.kneighbors(query)
    np.testing.assert_allclose(ours_distances, their_distances)
    np.testing.assert_array_equal(ours_indices, their_indices)


def test_kneighbors_training_query_matches_scikit_learn_self_exclusion():
    X = np.asarray([[0.0], [2.0], [5.0], [9.0]])
    y = np.asarray([0, 0, 1, 1])
    ours = KNeighborsClassifier(n_neighbors=2, algorithm="brute").fit(X, y)
    theirs = sklearn_neighbors.KNeighborsClassifier(
        n_neighbors=2, algorithm="brute"
    ).fit(X, y)
    ours_distances, ours_indices = ours.kneighbors()
    their_distances, their_indices = theirs.kneighbors()
    np.testing.assert_allclose(ours_distances, their_distances)
    np.testing.assert_array_equal(ours_indices, their_indices)
