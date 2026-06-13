from contextlib import nullcontext
from importlib.metadata import version

import numpy as np
import pytest
from rsklearn.preprocessing import OneHotEncoder

sklearn_preprocessing = pytest.importorskip("sklearn.preprocessing")
ScikitOneHotEncoder = sklearn_preprocessing.OneHotEncoder
SCIKIT_LEARN_VERSION = tuple(
    int(part) for part in version("scikit-learn").split(".")[:2]
)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"sparse_output": False},
        {"drop": "first"},
        {"drop": "if_binary"},
        {"drop": ["b", 2]},
        {"handle_unknown": "ignore"},
        {"handle_unknown": "infrequent_if_exist", "min_frequency": 2},
        {"min_frequency": 2},
        {"max_categories": 2},
        {"dtype": np.float32},
    ],
)
def test_fit_transform_matches_scikit_learn(options):
    X = np.asarray([["a", 1], ["b", 2], ["a", 3], ["c", 1], [np.nan, 2]], dtype=object)
    ours = OneHotEncoder(**options)
    theirs = ScikitOneHotEncoder(**options)
    ours_encoded = ours.fit_transform(X)
    theirs_encoded = theirs.fit_transform(X)
    ours_dense = (
        ours_encoded.toarray() if hasattr(ours_encoded, "toarray") else ours_encoded
    )
    theirs_dense = (
        theirs_encoded.toarray()
        if hasattr(theirs_encoded, "toarray")
        else theirs_encoded
    )
    np.testing.assert_array_equal(ours_dense, theirs_dense)
    np.testing.assert_array_equal(
        ours.get_feature_names_out(["left", "right"]),
        theirs.get_feature_names_out(["left", "right"]),
    )
    if ours.drop_idx_ is None:
        assert theirs.drop_idx_ is None
    else:
        np.testing.assert_array_equal(ours.drop_idx_, theirs.drop_idx_)


@pytest.mark.parametrize(
    "handle_unknown,frequency",
    [
        ("ignore", {}),
        ("infrequent_if_exist", {"min_frequency": 2}),
        ("warn", {"min_frequency": 2}),
    ],
)
def test_unknown_transform_and_inverse_match_scikit_learn(handle_unknown, frequency):
    X = np.asarray([["a"], ["a"], ["b"], ["c"]], dtype=object)
    test = np.asarray([["a"], ["other"]], dtype=object)
    ours = OneHotEncoder(handle_unknown=handle_unknown, **frequency).fit(X)
    theirs = ScikitOneHotEncoder(handle_unknown=handle_unknown, **frequency).fit(X)
    context = pytest.warns(UserWarning) if handle_unknown == "warn" else nullcontext()
    with context:
        ours_encoded = ours.transform(test)
    context = pytest.warns(UserWarning) if handle_unknown == "warn" else nullcontext()
    with context:
        theirs_encoded = theirs.transform(test)
    if handle_unknown == "warn" and SCIKIT_LEARN_VERSION < (1, 8):
        # scikit-learn 1.8 changed "warn" to encode unknowns as infrequent.
        np.testing.assert_array_equal(ours_encoded.toarray(), [[1, 0], [0, 1]])
        np.testing.assert_array_equal(
            ours.inverse_transform(ours_encoded), [["a"], ["infrequent_sklearn"]]
        )
        np.testing.assert_array_equal(theirs_encoded.toarray(), [[1, 0], [0, 0]])
        return
    np.testing.assert_array_equal(ours_encoded.toarray(), theirs_encoded.toarray())
    np.testing.assert_array_equal(
        ours.inverse_transform(ours_encoded), theirs.inverse_transform(theirs_encoded)
    )


def test_unknown_error_drop_and_explicit_categories():
    with pytest.raises(ValueError, match="unknown categories"):
        OneHotEncoder().fit([["a"], ["b"]]).transform([["other"]])
    with pytest.raises(ValueError, match="drop category"):
        OneHotEncoder(drop=["other"]).fit([["a"], ["b"]])
    encoder = OneHotEncoder(categories=[["b", "a"]], sparse_output=False)
    np.testing.assert_array_equal(
        encoder.fit_transform([["a"], ["b"]]), [[0, 1], [1, 0]]
    )
    with pytest.warns(UserWarning, match="all zeros"):
        OneHotEncoder(drop="first", handle_unknown="ignore").fit(
            [["a"], ["b"]]
        ).transform([["other"]])


def test_sparse_output_dense_output_and_inverse_validation():
    sparse_encoder = OneHotEncoder().fit([["a"], ["b"]])
    assert sparse_encoder.transform([["a"]]).format == "csr"
    dense_encoder = OneHotEncoder(sparse_output=False).fit([["a"], ["b"]])
    assert isinstance(dense_encoder.transform([["a"]]), np.ndarray)
    with pytest.raises(ValueError, match="feature count"):
        sparse_encoder.inverse_transform([[1, 0, 0]])
    with pytest.raises(ValueError, match="at most one"):
        sparse_encoder.inverse_transform([[1, 1]])


@pytest.mark.parametrize(
    "X",
    [
        np.asarray([[1], [2]], dtype=np.int64),
        np.asarray([["a"], ["b"]]),
        np.asarray([[True], [False]]),
    ],
)
def test_inverse_preserves_homogeneous_input_dtype(X):
    encoder = OneHotEncoder().fit(X)
    assert encoder.inverse_transform(encoder.transform(X)).dtype == X.dtype


def test_feature_name_callable_and_invalid_parameters():
    encoder = OneHotEncoder(
        feature_name_combiner=lambda feature, category: f"{feature}={category}"
    ).fit([["a"], ["b"]])
    np.testing.assert_array_equal(
        encoder.get_feature_names_out(["kind"]), ["kind=a", "kind=b"]
    )
    invalid = [
        {"drop": "bad"},
        {"sparse_output": 1},
        {"dtype": object},
        {"handle_unknown": "bad"},
        {"min_frequency": 0},
        {"max_categories": 0},
        {"feature_name_combiner": "bad"},
    ]
    for options in invalid:
        with pytest.raises((TypeError, ValueError)):
            OneHotEncoder(**options).fit([["a"]])


def test_non_contiguous_input_and_sparse_input_rejection():
    X = np.asarray([["a", "x", "b"], ["b", "y", "a"]])[:, ::2]
    encoder = OneHotEncoder().fit(X)
    assert encoder.transform(X).shape == (2, 4)
    sparse = pytest.importorskip("scipy.sparse")
    with pytest.raises(TypeError, match="dense data is required"):
        OneHotEncoder().fit(sparse.csr_matrix([[0, 1], [1, 0]]))
