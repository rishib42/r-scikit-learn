import numpy as np
import pytest
from rsklearn.base import BaseEstimator
from rsklearn.preprocessing._categorical import discover_categories, encode_categories

sklearn_preprocessing = pytest.importorskip("sklearn.preprocessing")
ScikitOrdinalEncoder = sklearn_preprocessing.OrdinalEncoder


class CategoricalEstimator(BaseEstimator):
    def fit(self, X):
        self._category_state, self.encoded_ = discover_categories(X, estimator=self)
        self.categories_ = list(self._category_state.categories)
        return self

    def transform(self, X):
        return encode_categories(X, self._category_state, estimator=self)


class ArrayWithColumns:
    def __init__(self, values, columns):
        self.values = np.asarray(values, dtype=object)
        self.columns = columns

    def __array__(self, dtype=None):
        return np.asarray(self.values, dtype=dtype)


def test_per_feature_numeric_boolean_and_unicode_discovery():
    cases = [
        (
            np.asarray([[3, 1], [1, 2], [3, 1]], dtype=np.int64),
            [[1, 3], [1, 2]],
            [[1, 0], [0, 1], [1, 0]],
        ),
        (
            np.asarray([[True, False], [False, True]]),
            [[False, True], [False, True]],
            [[1, 0], [0, 1]],
        ),
        (
            np.asarray([["東京", "b"], ["café", "a"], ["東京", "a"]]),
            [["café", "東京"], ["a", "b"]],
            [[1, 1], [0, 0], [1, 0]],
        ),
    ]
    for X, categories, expected in cases:
        estimator = CategoricalEstimator().fit(X)
        for actual, wanted in zip(estimator.categories_, categories, strict=True):
            np.testing.assert_array_equal(actual, wanted)
        np.testing.assert_array_equal(estimator.encoded_, expected)


def test_mixed_types_are_allowed_across_columns_but_rejected_within_one():
    X = np.asarray([[1, "a"], [2, "b"]], dtype=object)
    estimator = CategoricalEstimator().fit(X)
    np.testing.assert_array_equal(estimator.categories_[0], [1, 2])
    np.testing.assert_array_equal(estimator.categories_[1], ["a", "b"])
    with pytest.raises(TypeError, match="uniformly strings or numbers"):
        CategoricalEstimator().fit(np.asarray([[1], ["a"]], dtype=object))


def test_nan_and_none_are_known_missing_categories_and_unknowns_are_marked():
    X = np.asarray([[1.0, "a"], [np.nan, None], [2.0, np.nan]], dtype=object)
    estimator = CategoricalEstimator().fit(X)
    assert np.isnan(estimator.categories_[0][-1])
    assert np.isnan(estimator.categories_[1][-1])
    np.testing.assert_array_equal(estimator.encoded_, [[0, 0], [2, 1], [1, 2]])
    encoded, known = estimator.transform(
        np.asarray([[3.0, "unknown"], [np.nan, None]], dtype=object)
    )
    np.testing.assert_array_equal(encoded, [[-1, -1], [2, 1]])
    np.testing.assert_array_equal(known, [[False, False], [True, True]])


def test_boolean_with_nan_uses_numeric_missing_category():
    X = np.asarray([[True], [False], [np.nan]], dtype=object)
    estimator = CategoricalEstimator().fit(X)
    np.testing.assert_array_equal(estimator.encoded_.ravel(), [1, 0, 2])
    np.testing.assert_allclose(
        estimator.categories_[0], [0.0, 1.0, np.nan], equal_nan=True
    )


@pytest.mark.parametrize(
    "X",
    [
        np.asarray([[3, 1], [1, 2], [3, 1]], dtype=np.int64),
        np.asarray([[True, False], [False, True]], dtype=np.bool_),
        np.asarray([["東京", "b"], ["café", "a"], ["東京", "a"]]),
        np.asarray([[1.0, np.nan], [2.0, 3.0], [1.0, np.nan]]),
    ],
)
def test_discovery_and_sorting_match_scikit_learn(X):
    ours = CategoricalEstimator().fit(X)
    theirs = ScikitOrdinalEncoder().fit(X)
    for actual, expected in zip(ours.categories_, theirs.categories_, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_large_integer_values_are_preserved_exactly():
    X = np.asarray([[2**63], [2**63 + 1]], dtype=np.uint64)
    estimator = CategoricalEstimator().fit(X)
    assert estimator.categories_[0].dtype == np.uint64
    np.testing.assert_array_equal(estimator.categories_[0], X[:, 0])


def test_feature_names_and_feature_count_are_tracked():
    estimator = CategoricalEstimator().fit(
        ArrayWithColumns([["a", 1], ["b", 2]], ["name", "number"])
    )
    np.testing.assert_array_equal(estimator.feature_names_in_, ["name", "number"])
    with pytest.raises(ValueError, match="feature names must match"):
        estimator.transform(ArrayWithColumns([["a", 1]], ["number", "name"]))
    with pytest.raises(ValueError, match="expecting 2 features"):
        estimator.transform([["a"]])


def test_invalid_shape_complex_and_sparse_like_input_are_rejected():
    with pytest.raises(ValueError, match="2-dimensional"):
        CategoricalEstimator().fit(["a", "b"])
    with pytest.raises(ValueError, match="Complex data"):
        CategoricalEstimator().fit([[1 + 2j]])
    with pytest.raises(ValueError, match="infinity"):
        CategoricalEstimator().fit([[1.0], [np.inf]])
    with pytest.raises(ValueError, match="infinity"):
        CategoricalEstimator().fit(np.asarray([[1.0], [-np.inf]], dtype=object))


def test_non_contiguous_homogeneous_input_uses_the_same_categories_and_codes():
    X = np.asarray([[3, 9, 1], [1, 8, 2], [3, 7, 1]], dtype=np.int64)[:, ::2]
    assert not X.flags.c_contiguous
    estimator = CategoricalEstimator().fit(X)
    np.testing.assert_array_equal(estimator.categories_[0], [1, 3])
    np.testing.assert_array_equal(estimator.categories_[1], [1, 2])
    np.testing.assert_array_equal(estimator.encoded_, [[1, 0], [0, 1], [1, 0]])


def test_real_sparse_input_is_rejected():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    with pytest.raises(TypeError, match="does not support sparse input"):
        CategoricalEstimator().fit(scipy_sparse.csr_matrix([[0, 1], [1, 0]]))
