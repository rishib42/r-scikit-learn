import numpy as np
import pytest
from rsklearn.neighbors import KNeighborsClassifier
from rsklearn.utils.validation import NotFittedError


def test_kneighbors_classifier_predicts_string_labels():
    X = np.asarray([[0.0], [1.0], [10.0], [11.0]])
    y = np.asarray(["left", "left", "right", "right"])
    classifier = KNeighborsClassifier(n_neighbors=1).fit(X, y)
    np.testing.assert_array_equal(
        classifier.predict([[0.2], [10.8]]), np.asarray(["left", "right"])
    )


def test_kneighbors_returns_sorted_distances_and_indices():
    X = np.asarray([[0.0], [2.0], [5.0], [9.0]])
    classifier = KNeighborsClassifier(n_neighbors=2).fit(X, [0, 0, 1, 1])
    distances, indices = classifier.kneighbors([[1.0]], return_distance=True)
    np.testing.assert_allclose(distances, [[1.0, 1.0]])
    np.testing.assert_array_equal(indices, [[0, 1]])
    np.testing.assert_array_equal(
        classifier.kneighbors([[1.0]], return_distance=False), [[0, 1]]
    )


def test_kneighbors_without_query_excludes_training_sample_itself():
    X = np.asarray([[0.0], [3.0], [7.0]])
    classifier = KNeighborsClassifier(n_neighbors=1).fit(X, [0, 1, 1])
    distances, indices = classifier.kneighbors()
    np.testing.assert_allclose(distances[:, 0], [3.0, 3.0, 4.0])
    np.testing.assert_array_equal(indices[:, 0], [1, 0, 1])


def test_distance_weighting_uses_exact_matches_only_for_zero_distance():
    X = np.asarray([[0.0], [0.0], [10.0]])
    y = np.asarray([0, 1, 1])
    classifier = KNeighborsClassifier(n_neighbors=3, weights="distance").fit(X, y)
    probabilities = classifier.predict_proba([[0.0]])
    np.testing.assert_allclose(probabilities, [[0.5, 0.5]])


def test_manhattan_metric_predicts_expected_class():
    X = np.asarray([[0.0, 0.0], [2.0, 2.0], [10.0, 10.0]])
    classifier = KNeighborsClassifier(n_neighbors=1, metric="manhattan", p=1).fit(
        X, [0, 0, 1]
    )
    np.testing.assert_array_equal(classifier.predict([[9.0, 10.0]]), [1])


def test_invalid_parameters_raise_clear_errors():
    with pytest.raises(ValueError, match="n_neighbors"):
        KNeighborsClassifier(n_neighbors=0).fit([[0], [1]], [0, 1])
    with pytest.raises(NotImplementedError, match="algorithm"):
        KNeighborsClassifier(algorithm="kd_tree").fit([[0], [1]], [0, 1])
    with pytest.raises(NotImplementedError, match="Minkowski p=1 or p=2"):
        KNeighborsClassifier(p=3).fit([[0], [1]], [0, 1])


def test_predict_rejects_unfitted_estimator():
    with pytest.raises(NotFittedError):
        KNeighborsClassifier().predict([[0.0]])


def test_too_many_neighbors_raises():
    classifier = KNeighborsClassifier(n_neighbors=3).fit([[0.0], [1.0]], [0, 1])
    with pytest.raises(ValueError, match="n_neighbors"):
        classifier.predict([[0.5]])
