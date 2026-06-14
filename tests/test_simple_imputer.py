import numpy as np
import pytest
import rsklearn.impute._simple_imputer as simple_imputer_module
from rsklearn.impute import SimpleImputer


@pytest.mark.parametrize(
    "strategy,expected_statistics",
    [
        ("mean", [2.0, 6.0, 2.0]),
        ("median", [2.0, 6.0, 2.0]),
        ("most_frequent", [1.0, 4.0, 1.0]),
        ("constant", [-5.0, -5.0, -5.0]),
    ],
)
def test_numeric_strategies_and_custom_fill(strategy, expected_statistics):
    X = np.asarray([[1.0, 4.0, 1.0], [np.nan, 8.0, 2.0], [3.0, np.nan, 3.0]])
    options = {"fill_value": -5.0} if strategy == "constant" else {}
    imputer = SimpleImputer(strategy=strategy, **options).fit(X)
    np.testing.assert_allclose(imputer.statistics_, expected_statistics)
    output = imputer.transform(X)
    assert not np.isnan(output).any()
    np.testing.assert_array_equal(X, [[1, 4, 1], [np.nan, 8, 2], [3, np.nan, 3]])


def test_numeric_custom_sentinel_float32_and_copy_false():
    X = np.asarray([[1.0, -1.0], [3.0, 5.0]], dtype=np.float32)
    output = SimpleImputer(missing_values=-1.0, copy=False).fit_transform(X)
    assert output is X
    assert output.dtype == np.float32
    np.testing.assert_array_equal(output, [[1.0, 5.0], [3.0, 5.0]])


def test_native_numeric_mean_fit_uses_fused_rust_metadata(monkeypatch):
    def fail_if_python_mask_is_built(*args, **kwargs):
        raise AssertionError("numeric mean fit should not build a Python missing mask")

    monkeypatch.setattr(
        simple_imputer_module, "_missing_mask", fail_if_python_mask_is_built
    )
    imputer = SimpleImputer().fit([[1.0, np.nan], [3.0, np.nan]])
    np.testing.assert_array_equal(imputer.statistics_[:1], [2.0])
    assert np.isnan(imputer.statistics_[1])
    np.testing.assert_array_equal(imputer._indicator_features, [1])
    np.testing.assert_array_equal(imputer._retained_features, [0])


def test_string_object_most_frequent_and_constant_defaults():
    X = np.asarray([["b", None], ["a", "x"], ["a", None]], dtype=object)
    frequent = SimpleImputer(missing_values=None, strategy="most_frequent").fit(X)
    np.testing.assert_array_equal(frequent.statistics_, ["a", "x"])
    np.testing.assert_array_equal(
        frequent.transform(X), [["b", "x"], ["a", "x"], ["a", "x"]]
    )
    strings = np.asarray([["a"], ["?"]])
    output = SimpleImputer(missing_values="?", strategy="constant").fit_transform(
        strings
    )
    assert output.dtype == object
    np.testing.assert_array_equal(output, [["a"], ["missing_value"]])


def test_callable_strategy_receives_non_missing_column_values():
    seen = []

    def statistic(values):
        seen.append(values.copy())
        return np.max(values) + 1

    X = np.asarray([[1.0, np.nan], [3.0, 4.0]])
    imputer = SimpleImputer(strategy=statistic).fit(X)
    np.testing.assert_array_equal(imputer.statistics_, [4.0, 5.0])
    np.testing.assert_array_equal(seen[0], [1.0, 3.0])
    np.testing.assert_array_equal(seen[1], [4.0])


def test_empty_features_are_dropped_or_retained():
    X = np.asarray([[1.0, np.nan], [3.0, np.nan]])
    dropped = SimpleImputer().fit(X)
    with pytest.warns(UserWarning, match="Skipping features"):
        np.testing.assert_array_equal(dropped.transform(X), [[1.0], [3.0]])
    np.testing.assert_array_equal(dropped.get_feature_names_out(), ["x0"])
    retained = SimpleImputer(keep_empty_features=True).fit(X)
    np.testing.assert_array_equal(retained.statistics_, [2.0, 0.0])
    np.testing.assert_array_equal(retained.transform(X), [[1.0, 0.0], [3.0, 0.0]])


def test_indicator_feature_names_and_inverse_transform():
    X = np.asarray([[1.0, np.nan], [np.nan, 2.0]])
    imputer = SimpleImputer(add_indicator=True).fit(X)
    transformed = imputer.transform(X)
    np.testing.assert_array_equal(
        transformed, [[1.0, 2.0, 0.0, 1.0], [1.0, 2.0, 1.0, 0.0]]
    )
    np.testing.assert_array_equal(imputer.indicator_.features_, [0, 1])
    np.testing.assert_array_equal(
        imputer.indicator_.transform(X), [[False, True], [True, False]]
    )
    np.testing.assert_array_equal(
        imputer.get_feature_names_out(),
        ["x0", "x1", "missingindicator_x0", "missingindicator_x1"],
    )
    np.testing.assert_array_equal(imputer.inverse_transform(transformed), X)


def test_feature_tracking_and_transform_validation():
    imputer = SimpleImputer().fit([[1.0, np.nan], [2.0, 3.0]])
    assert imputer.n_features_in_ == 2
    with pytest.raises(ValueError, match="expecting 2 features"):
        imputer.transform([[1.0]])
    with pytest.raises(ValueError, match="not fitted"):
        SimpleImputer().transform([[1.0]])
    with pytest.raises(ValueError, match="add_indicator=True"):
        imputer.inverse_transform([[1.0, 2.0]])


@pytest.mark.parametrize(
    "options,exception",
    [
        ({"strategy": "bad"}, ValueError),
        ({"copy": 1}, TypeError),
        ({"add_indicator": 1}, TypeError),
        ({"keep_empty_features": 1}, TypeError),
    ],
)
def test_invalid_parameters_are_rejected_on_fit(options, exception):
    with pytest.raises(exception):
        SimpleImputer(**options).fit([[1.0]])


def test_numeric_strategy_rejects_strings_and_sparse_is_explicitly_rejected():
    with pytest.raises(ValueError, match="requires numeric"):
        SimpleImputer(strategy="mean").fit([["a"], ["b"]])
    sparse = pytest.importorskip("scipy.sparse")
    with pytest.raises(TypeError, match="dense data is required"):
        SimpleImputer().fit(sparse.csr_matrix([[1.0, np.nan]]))


def test_infinity_and_unexpected_nan_are_rejected():
    with pytest.raises(ValueError, match="infinity"):
        SimpleImputer().fit([[np.inf]])
    with pytest.raises(ValueError, match="contains NaN"):
        SimpleImputer(missing_values=-1).fit([[np.nan]])
    with pytest.raises(ValueError, match="infinity"):
        SimpleImputer(strategy="constant").fit(
            np.asarray([["a"], [np.inf]], dtype=object)
        )
