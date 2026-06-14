import numpy as np
import pytest
from rsklearn.compose import ColumnTransformer, make_column_transformer
from rsklearn.preprocessing import OneHotEncoder, StandardScaler

sklearn_compose = pytest.importorskip("sklearn.compose")
sklearn_preprocessing = pytest.importorskip("sklearn.preprocessing")


def test_dense_sparse_values_metadata_and_feature_names_match_scikit_learn():
    X = np.column_stack((np.arange(10), np.arange(10), np.arange(10) + 100))
    ours = ColumnTransformer(
        [
            ("numeric", StandardScaler(), [0]),
            ("categorical", OneHotEncoder(), [1]),
        ],
        remainder="passthrough",
        sparse_threshold=0.5,
    )
    theirs = sklearn_compose.ColumnTransformer(
        [
            ("numeric", sklearn_preprocessing.StandardScaler(), [0]),
            ("categorical", sklearn_preprocessing.OneHotEncoder(), [1]),
        ],
        remainder="passthrough",
        sparse_threshold=0.5,
    )
    ours_output = ours.fit_transform(X)
    theirs_output = theirs.fit_transform(X)
    np.testing.assert_allclose(ours_output.toarray(), theirs_output.toarray())
    np.testing.assert_allclose(
        ours.transform(X).toarray(), theirs.transform(X).toarray()
    )
    np.testing.assert_array_equal(
        ours.get_feature_names_out(), theirs.get_feature_names_out()
    )
    assert ours.sparse_output_ == theirs.sparse_output_
    assert ours.output_indices_ == theirs.output_indices_


def test_parameters_and_automatic_names_match_scikit_learn():
    ours = make_column_transformer((StandardScaler(), [0]), (StandardScaler(), [1]))
    theirs = sklearn_compose.make_column_transformer(
        (sklearn_preprocessing.StandardScaler(), [0]),
        (sklearn_preprocessing.StandardScaler(), [1]),
    )
    assert [name for name, _, _ in ours.transformers] == [
        name for name, _, _ in theirs.transformers
    ]
    ours_params = set(ours.get_params(deep=False))
    theirs_params = set(theirs.get_params(deep=False))
    ours_params.discard("force_int_remainder_cols")
    theirs_params.discard("force_int_remainder_cols")
    assert ours_params == theirs_params
    assert "standardscaler-1__with_mean" in ours.get_params()
