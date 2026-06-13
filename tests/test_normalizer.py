import numpy as np
import pytest
from rsklearn.preprocessing import Normalizer


@pytest.mark.parametrize(
    "norm,expected",
    [
        ("l1", [[0.0, 0.0], [3.0 / 7.0, 4.0 / 7.0], [-3.0 / 7.0, 4.0 / 7.0]]),
        ("l2", [[0.0, 0.0], [0.6, 0.8], [-0.6, 0.8]]),
        ("max", [[0.0, 0.0], [0.75, 1.0], [-0.75, 1.0]]),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_normalizes_rows_and_preserves_float_dtype(norm, expected, dtype):
    X = np.asarray([[0.0, 0.0], [3.0, 4.0], [-3.0, 4.0]], dtype=dtype)
    original = X.copy()
    output = Normalizer(norm=norm).fit_transform(X)
    np.testing.assert_allclose(output, expected, rtol=1e-6)
    np.testing.assert_array_equal(X, original)
    assert output.dtype == dtype


def test_copy_false_mutates_compatible_input_and_returns_it():
    X = np.asarray([[3.0, 4.0]], dtype=np.float64)
    output = Normalizer(copy=False).fit_transform(X)
    assert output is X
    np.testing.assert_allclose(X, [[0.6, 0.8]])


def test_copy_false_is_best_effort_for_input_requiring_conversion():
    X = np.asarray([[3, 4]], dtype=np.int64)
    output = Normalizer(copy=False).fit_transform(X)
    np.testing.assert_array_equal(X, [[3, 4]])
    np.testing.assert_allclose(output, [[0.6, 0.8]])
    assert output.dtype == np.float64


def test_fit_records_features_and_transform_checks_them():
    normalizer = Normalizer().fit([[1.0, 2.0]])
    assert normalizer.n_features_in_ == 2
    with pytest.raises(ValueError, match="expecting 2 features"):
        normalizer.transform([[1.0]])


def test_non_contiguous_and_tiny_rows_are_supported():
    X = np.asarray([[3.0, 0.0, 4.0], [0.0, 0.0, 0.0]])[:, ::2]
    np.testing.assert_allclose(Normalizer().fit_transform(X), [[0.6, 0.8], [0.0, 0.0]])
    tiny = np.asarray([[1e-308, 1e-308]])
    np.testing.assert_array_equal(Normalizer().fit_transform(tiny), tiny)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_values_are_rejected(value):
    with pytest.raises(ValueError, match="contains"):
        Normalizer().fit_transform([[value, 1.0]])


@pytest.mark.parametrize(
    "kwargs,exception",
    [
        ({"norm": "invalid"}, ValueError),
        ({"norm": 1}, ValueError),
        ({"copy": 1}, TypeError),
    ],
)
def test_parameter_validation_is_delayed_until_use(kwargs, exception):
    normalizer = Normalizer(**kwargs)
    assert normalizer.get_params(deep=False) == {
        "copy": kwargs.get("copy", True),
        "norm": kwargs.get("norm", "l2"),
    }
    with pytest.raises(exception):
        normalizer.fit([[1.0]])


def test_transform_requires_fit():
    with pytest.raises(ValueError, match="not fitted"):
        Normalizer().transform([[1.0]])
