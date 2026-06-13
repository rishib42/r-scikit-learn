import numpy as np
import pytest
from rsklearn.preprocessing import OrdinalEncoder
from rsklearn.utils.validation import check_is_fitted

sklearn_preprocessing = pytest.importorskip("sklearn.preprocessing")
ScikitOrdinalEncoder = sklearn_preprocessing.OrdinalEncoder


@pytest.mark.parametrize(
    "X",
    [
        np.asarray([[3, 1], [1, 2], [3, 1]], dtype=np.int64),
        np.asarray([[True, False], [False, True]], dtype=np.bool_),
        np.asarray([["東京", "b"], ["café", "a"], ["東京", "a"]]),
        np.asarray([[1.0, np.nan], [2.0, 3.0], [1.0, np.nan]]),
        np.asarray([[1, "a"], [2, "b"], [1, None]], dtype=object),
    ],
)
def test_default_fit_transform_and_inverse_match_scikit_learn(X):
    ours = OrdinalEncoder()
    theirs = ScikitOrdinalEncoder()
    ours_encoded = ours.fit_transform(X)
    theirs_encoded = theirs.fit_transform(X)
    np.testing.assert_allclose(ours_encoded, theirs_encoded, equal_nan=True)
    for actual, expected in zip(ours.categories_, theirs.categories_, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_equal(
        ours.inverse_transform(ours_encoded), theirs.inverse_transform(theirs_encoded)
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_output_dtype(dtype):
    options = {"dtype": dtype}
    if np.issubdtype(dtype, np.integer):
        options["encoded_missing_value"] = -2
    encoder = OrdinalEncoder(**options)
    encoded = encoder.fit_transform([["a"], ["b"], [np.nan]])
    assert encoded.dtype == np.dtype(dtype)


@pytest.mark.parametrize("unknown_value", [-1, 99, np.nan])
def test_unknown_categories_use_configured_value_and_inverse_to_none(unknown_value):
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=unknown_value
    ).fit([["a"], ["b"]])
    encoded = encoder.transform([["other"], ["a"]])
    if np.isnan(unknown_value):
        assert np.isnan(encoded[0, 0])
    else:
        assert encoded[0, 0] == unknown_value
    np.testing.assert_array_equal(encoder.inverse_transform(encoded), [[None], ["a"]])


def test_unknown_categories_error_by_default():
    encoder = OrdinalEncoder().fit([["a"], ["b"]])
    with pytest.raises(ValueError, match="unknown category"):
        encoder.transform([["other"]])


def test_missing_values_use_configured_value_and_round_trip():
    encoder = OrdinalEncoder(encoded_missing_value=-7).fit([["a"], [np.nan]])
    np.testing.assert_array_equal(encoder.transform([[np.nan], ["a"]]), [[-7], [0]])
    transformed = encoder.transform([["a"], [np.nan]])
    inverse = encoder.inverse_transform(transformed)
    assert inverse[0, 0] == "a"
    assert np.isnan(inverse[1, 0])


def test_unknown_and_missing_values_may_share_a_non_category_code():
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
    ).fit([["a"], [np.nan]])
    np.testing.assert_array_equal(
        encoder.transform([["other"], [np.nan]]), [[-1], [-1]]
    )


@pytest.mark.parametrize(
    "options",
    [
        {"min_frequency": 2},
        {"min_frequency": 0.3},
        {"max_categories": 3},
        {"min_frequency": 2, "max_categories": 2},
    ],
)
def test_infrequent_category_encoding_matches_scikit_learn(options):
    X = np.asarray(
        [["a"], ["b"], ["b"], ["c"], ["c"], ["c"], ["d"], [np.nan]],
        dtype=object,
    )
    ours = OrdinalEncoder(**options).fit(X)
    theirs = ScikitOrdinalEncoder(**options).fit(X)
    np.testing.assert_allclose(ours.transform(X), theirs.transform(X), equal_nan=True)
    np.testing.assert_array_equal(
        ours.infrequent_categories_[0], theirs.infrequent_categories_[0]
    )


def test_explicit_categories_preserve_string_order_and_validate_numeric_order():
    encoder = OrdinalEncoder(categories=[["b", "a"]]).fit([["a"], ["b"]])
    np.testing.assert_array_equal(encoder.transform([["a"], ["b"]]), [[1], [0]])
    with pytest.raises(ValueError, match="sorted"):
        OrdinalEncoder(categories=[[2, 1]]).fit([[1], [2]])


def test_explicit_categories_must_cover_fit_data_and_be_unique():
    with pytest.raises(ValueError, match="unknown category"):
        OrdinalEncoder(categories=[["a"]]).fit([["a"], ["b"]])
    with pytest.raises(ValueError, match="duplicate"):
        OrdinalEncoder(categories=[["a", "a"]]).fit([["a"]])


@pytest.mark.parametrize(
    "options,match",
    [
        ({"dtype": np.int64}, "encoded_missing_value"),
        (
            {"handle_unknown": "use_encoded_value", "unknown_value": None},
            "unknown_value",
        ),
        (
            {"handle_unknown": "error", "unknown_value": -1},
            "unknown_value",
        ),
        (
            {"handle_unknown": "use_encoded_value", "unknown_value": 0},
            "distinct",
        ),
        ({"encoded_missing_value": 0}, "distinct"),
        ({"min_frequency": 0}, "min_frequency"),
        ({"min_frequency": 1.0}, "min_frequency"),
        ({"max_categories": 0}, "max_categories"),
    ],
)
def test_invalid_parameter_combinations(options, match):
    with pytest.raises((TypeError, ValueError), match=match):
        OrdinalEncoder(**options).fit([["a"], [np.nan]])


def test_feature_metadata_non_contiguous_and_sparse_rejection():
    X = np.asarray([["a", "x", "b"], ["b", "y", "a"]])[:, ::2]
    assert not X.flags.c_contiguous
    encoder = OrdinalEncoder().fit(X)
    assert encoder.n_features_in_ == 2
    np.testing.assert_array_equal(encoder.fit_transform(X), [[0, 1], [1, 0]])
    scipy_sparse = pytest.importorskip("scipy.sparse")
    with pytest.raises(TypeError, match="does not support sparse input"):
        OrdinalEncoder().fit(scipy_sparse.csr_matrix([[0, 1], [1, 0]]))


def test_feature_names_out():
    encoder = OrdinalEncoder().fit([["a", "b"], ["c", "d"]])
    np.testing.assert_array_equal(encoder.get_feature_names_out(), ["x0", "x1"])
    np.testing.assert_array_equal(
        encoder.get_feature_names_out(["left", "right"]), ["left", "right"]
    )
    with pytest.raises(ValueError, match="2 feature names"):
        encoder.get_feature_names_out(["left"])


def test_fitted_validation_and_repr():
    encoder = OrdinalEncoder()
    with pytest.raises(ValueError, match="not fitted"):
        encoder.transform([["a"]])
    with pytest.raises(ValueError, match="not fitted"):
        check_is_fitted(encoder)
    assert "handle_unknown='error'" in repr(encoder)
    encoder.fit([["a"], ["b"]])
    assert not hasattr(encoder, "infrequent_categories_")
