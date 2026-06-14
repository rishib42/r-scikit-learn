import numpy as np
import pytest
from rsklearn.base import BaseEstimator, TransformerMixin, clone
from rsklearn.compose import ColumnTransformer, make_column_transformer
from rsklearn.impute import SimpleImputer
from rsklearn.pipeline import make_pipeline
from rsklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

sparse = pytest.importorskip("scipy.sparse")


class NamedTable:
    def __init__(self, values, columns):
        self.values = np.asarray(values, dtype=object)
        self.columns = np.asarray(columns, dtype=object)

    @property
    def shape(self):
        return self.values.shape

    def __array__(self, dtype=None):
        return np.asarray(self.values, dtype=dtype)

    def __getitem__(self, columns):
        if isinstance(columns, str):
            index = int(np.flatnonzero(self.columns == columns)[0])
            return self.values[:, index]
        indices = [int(np.flatnonzero(self.columns == name)[0]) for name in columns]
        return NamedTable(self.values[:, indices], self.columns[indices])


class OneDimensionalTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        del y
        self.seen_ndim_ = np.asarray(X).ndim
        return self

    def transform(self, X):
        return np.asarray(X)[:, None]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


class DuplicateNameTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        del y
        return self

    def transform(self, X):
        return np.asarray(X)

    def get_feature_names_out(self, input_features=None):
        del input_features
        return np.asarray(["same"])


class FitTransformOnly(BaseEstimator):
    def fit_transform(self, X, y=None):
        del y
        return np.asarray(X) + 1

    def transform(self, X):
        return np.asarray(X) + 1


def test_dense_composition_remainder_metadata_and_cloning():
    scaler = StandardScaler()
    transformer = ColumnTransformer(
        [("scale", scaler, [0, 1]), ("drop", "drop", [2])],
        remainder="passthrough",
    )
    X = np.asarray([[1.0, 10.0, 100.0, 5.0], [3.0, 30.0, 200.0, 7.0]])
    output = transformer.fit_transform(X)
    np.testing.assert_allclose(output, [[-1.0, -1.0, 5.0], [1.0, 1.0, 7.0]])
    assert not hasattr(scaler, "mean_")
    assert transformer.named_transformers_.scale is not scaler
    assert transformer.output_indices_ == {
        "scale": slice(0, 2),
        "drop": slice(2, 2),
        "remainder": slice(2, 3),
    }
    np.testing.assert_array_equal(
        transformer.get_feature_names_out(),
        ["scale__x0", "scale__x1", "remainder__x3"],
    )
    np.testing.assert_allclose(transformer.transform(X), output)


def test_empty_transformer_list_produces_zero_column_output():
    transformer = ColumnTransformer([])
    output = transformer.fit_transform([[1.0, 2.0], [3.0, 4.0]])
    assert output.shape == (2, 0)
    assert transformer.transform([[5.0, 6.0]]).shape == (1, 0)


def test_empty_selection_is_not_fitted_and_fit_transform_only_is_supported():
    transformer = ColumnTransformer(
        [("empty", StandardScaler(), []), ("active", FitTransformOnly(), [0])]
    )
    output = transformer.fit_transform([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(output, [[2.0], [4.0]])
    assert not hasattr(transformer.named_transformers_.empty, "mean_")
    np.testing.assert_array_equal(transformer.transform([[5.0, 6.0]]), [[6.0]])


def test_sparse_and_dense_stacking_obeys_density_threshold():
    X = np.column_stack((np.arange(10), np.arange(10)))
    sparse_output = ColumnTransformer(
        [
            ("numeric", StandardScaler(), [0]),
            ("categorical", OneHotEncoder(), [1]),
        ],
        sparse_threshold=0.5,
    ).fit_transform(X)
    assert sparse.isspmatrix_csr(sparse_output)

    dense_output = ColumnTransformer(
        [
            ("numeric", StandardScaler(), [0]),
            ("categorical", OneHotEncoder(), [1]),
        ],
        sparse_threshold=0,
    ).fit_transform(X)
    assert isinstance(dense_output, np.ndarray)
    np.testing.assert_allclose(dense_output, sparse_output.toarray())


def test_transform_preserves_fitted_sparse_output_policy():
    transformer = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), [0])],
        sparse_threshold=0.5,
    ).fit(np.arange(10)[:, None])
    output = transformer.transform(np.zeros((10, 1), dtype=np.int64))
    assert sparse.isspmatrix_csr(output)


def test_sparse_input_can_be_selected_and_stacked_without_densifying():
    X = sparse.csr_matrix([[0.0, 1.0], [2.0, 0.0]])
    transformer = ColumnTransformer(
        [("left", "passthrough", [0])],
        remainder="passthrough",
        sparse_threshold=0.8,
    )
    output = transformer.fit_transform(X)
    assert sparse.isspmatrix_csr(output)
    np.testing.assert_array_equal(output.toarray(), X.toarray())
    assert sparse.isspmatrix_csr(transformer.transform(X))


def test_real_mixed_pipeline_with_named_dataframe():
    pandas = pytest.importorskip("pandas")
    X = pandas.DataFrame(
        {
            "age": [20.0, np.nan, 40.0],
            "city": ["a", "b", "a"],
            "unused": [1, 2, 3],
        }
    )
    transformer = ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(SimpleImputer(), StandardScaler()),
                ["age"],
            ),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                ["city"],
            ),
        ]
    )
    output = transformer.fit_transform(X)
    assert isinstance(output, np.ndarray)
    assert output.shape == (3, 3)
    np.testing.assert_allclose(
        output,
        [
            [-1.224744871391589, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.224744871391589, 1.0, 0.0],
        ],
    )
    np.testing.assert_array_equal(
        transformer.get_feature_names_out(),
        ["numeric__age", "categorical__city_a", "categorical__city_b"],
    )
    reordered = X[["city", "age", "unused"]].assign(extra=10)
    assert transformer.transform(reordered).shape == output.shape
    with pytest.raises(ValueError, match="missing fitted"):
        transformer.transform(X[["city", "unused"]])


def test_named_columns_follow_names_across_reordered_and_extra_input():
    fitted = NamedTable([[1, "a", 9], [3, "b", 8]], ["number", "kind", "unused"])
    transformer = ColumnTransformer(
        [
            ("number", StandardScaler(), ["number"]),
            ("kind", OneHotEncoder(sparse_output=False), ["kind"]),
        ],
        remainder="passthrough",
    ).fit(fitted)
    reordered = NamedTable(
        [["a", 1, 100, 9], ["b", 3, 200, 8]],
        ["kind", "number", "extra", "unused"],
    )
    np.testing.assert_array_equal(
        transformer.transform(reordered),
        transformer.transform(fitted),
    )
    with pytest.raises(ValueError, match="missing fitted"):
        transformer.transform(NamedTable([["a"], ["b"]], ["kind"]))


def test_selectors_weights_scalar_columns_and_callable_resolution():
    X = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    transformer = ColumnTransformer(
        [
            ("scalar", OneDimensionalTransformer(), 0),
            ("mask", "passthrough", [False, True, False]),
            ("callable", "passthrough", lambda value: [value.shape[1] - 1]),
        ],
        transformer_weights={"mask": 10},
    )
    output = transformer.fit_transform(X)
    np.testing.assert_array_equal(output, [[1, 20, 3], [4, 50, 6]])
    assert transformer.named_transformers_.scalar.seen_ndim_ == 1
    np.testing.assert_array_equal(transformer.transform(X), output)


def test_nested_parameters_replacement_clone_and_automatic_names():
    transformer = make_column_transformer(
        (StandardScaler(), [0]),
        (StandardScaler(), [1]),
        (MinMaxScaler(), [2]),
    )
    assert [name for name, _, _ in transformer.transformers] == [
        "standardscaler-1",
        "standardscaler-2",
        "minmaxscaler",
    ]
    transformer.set_params(
        **{
            "standardscaler-1__with_mean": False,
            "standardscaler-2": StandardScaler(with_std=False),
        }
    )
    assert transformer.get_params()["standardscaler-1__with_mean"] is False
    copied = clone(transformer)
    assert copied is not transformer
    assert copied.transformers[0][1] is not transformer.transformers[0][1]


def test_remainder_estimator_nested_parameters_and_weights():
    transformer = ColumnTransformer(
        [("first", "passthrough", [0])],
        remainder=StandardScaler(),
        transformer_weights={"first": 2},
    )
    assert transformer.get_params()["remainder__with_mean"] is True
    transformer.set_params(remainder__with_mean=False)
    output = transformer.fit_transform([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(output[:, 0], [2.0, 6.0])
    assert transformer.named_transformers_.remainder.with_mean is False


def test_feature_name_configuration_and_duplicate_detection():
    X = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    plain = ColumnTransformer(
        [("left", StandardScaler(), [0]), ("right", StandardScaler(), [1])],
        verbose_feature_names_out=False,
    ).fit(X)
    np.testing.assert_array_equal(plain.get_feature_names_out(), ["x0", "x1"])
    formatted = ColumnTransformer(
        [("left", StandardScaler(), [0])],
        verbose_feature_names_out="{transformer_name}:{feature_name}",
    ).fit(X)
    np.testing.assert_array_equal(formatted.get_feature_names_out(), ["left:x0"])
    duplicate = ColumnTransformer(
        [
            ("left", DuplicateNameTransformer(), [0]),
            ("right", DuplicateNameTransformer(), [1]),
        ],
        verbose_feature_names_out=False,
    ).fit(X)
    with pytest.raises(ValueError, match="not unique"):
        duplicate.get_feature_names_out()


@pytest.mark.parametrize(
    "transformer,exception,match",
    [
        (
            ColumnTransformer([("same", "drop", [0]), ("same", "drop", [1])]),
            ValueError,
            "unique",
        ),
        (
            ColumnTransformer([("bad__name", "drop", [0])]),
            ValueError,
            "must not contain",
        ),
        (
            ColumnTransformer([("bad", object(), [0])]),
            TypeError,
            "fit or fit_transform",
        ),
        (
            ColumnTransformer([("bad", "unknown", [0])]),
            ValueError,
            "must be an estimator",
        ),
        (ColumnTransformer([("ok", "drop", [2])]), ValueError, "out-of-bounds"),
        (ColumnTransformer([("ok", "drop", [True])]), ValueError, "match input width"),
        (ColumnTransformer([("ok", "drop", [[0]])]), ValueError, "one-dimensional"),
        (
            ColumnTransformer([("ok", "drop", [0])], sparse_threshold=2),
            ValueError,
            "in",
        ),
        (
            ColumnTransformer([("ok", "drop", [0])], n_jobs=2),
            NotImplementedError,
            "parallel",
        ),
    ],
)
def test_invalid_configuration_is_rejected(transformer, exception, match):
    with pytest.raises(exception, match=match):
        transformer.fit([[1.0, 2.0]])
