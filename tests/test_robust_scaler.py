import numpy as np
import pytest
from rsklearn.preprocessing import RobustScaler


def test_attributes_transform_and_inverse_roundtrip():
    X = np.asarray([[1.0, 10.0], [2.0, 20.0], [100.0, 30.0]])
    original = X.copy()
    scaler = RobustScaler().fit(X)
    np.testing.assert_allclose(scaler.center_, [2.0, 20.0])
    np.testing.assert_allclose(scaler.scale_, [49.5, 10.0])
    transformed = scaler.transform(X)
    np.testing.assert_allclose(scaler.inverse_transform(transformed), X)
    np.testing.assert_array_equal(X, original)
    assert scaler.n_features_in_ == 2


@pytest.mark.parametrize(
    "options",
    [
        {"with_centering": False},
        {"with_scaling": False},
        {"with_centering": False, "with_scaling": False},
        {"quantile_range": (10.0, 90.0)},
        {"unit_variance": True},
    ],
)
def test_options_and_nan_handling(options):
    X = np.asarray([[1.0, np.nan], [2.0, 4.0], [5.0, 8.0]], dtype=np.float32)
    scaler = RobustScaler(**options)
    transformed = scaler.fit_transform(X)
    assert transformed.dtype == np.float32
    assert np.isnan(transformed[0, 1])
    np.testing.assert_allclose(
        scaler.inverse_transform(transformed), X, rtol=1e-6, equal_nan=True
    )
    assert (scaler.center_ is None) == (not options.get("with_centering", True))
    assert (scaler.scale_ is None) == (not options.get("with_scaling", True))


def test_constant_and_all_nan_features():
    scaler = RobustScaler().fit([[4.0, np.nan], [4.0, np.nan]])
    np.testing.assert_allclose(scaler.center_[0], 4.0)
    np.testing.assert_allclose(scaler.scale_[0], 1.0)
    assert np.isnan(scaler.center_[1])
    assert np.isnan(scaler.scale_[1])


def test_copy_false_mutates_compatible_input_best_effort():
    X = np.asarray([[1.0], [3.0]], dtype=np.float64)
    output = RobustScaler(copy=False).fit_transform(X)
    assert output is X
    np.testing.assert_allclose(X, [[-1.0], [1.0]])
    integers = np.asarray([[1], [3]], dtype=np.int64)
    integer_output = RobustScaler(copy=False).fit_transform(integers)
    np.testing.assert_array_equal(integers, [[1], [3]])
    assert integer_output.dtype == np.float64


def test_non_contiguous_input_and_extreme_unit_variance_ranges():
    X = np.asarray([[1.0, 0.0, 2.0], [4.0, 0.0, 100.0]])[:, ::2]
    output = RobustScaler().fit_transform(X)
    assert output.flags.c_contiguous
    for quantile_range in [(0.0, 100.0), (50.0, 50.0), (5.0, 95.0)]:
        scaler = RobustScaler(quantile_range=quantile_range, unit_variance=True).fit(X)
        assert scaler.scale_.shape == (2,)


@pytest.mark.parametrize(
    "kwargs,exception",
    [
        ({"with_centering": 1}, TypeError),
        ({"with_scaling": 1}, TypeError),
        ({"copy": 1}, TypeError),
        ({"unit_variance": 1}, TypeError),
        ({"quantile_range": "invalid"}, TypeError),
        ({"quantile_range": (-1, 50)}, ValueError),
        ({"quantile_range": (90, 10)}, ValueError),
        ({"quantile_range": ("a", 10)}, ValueError),
    ],
)
def test_parameter_validation_is_delayed_until_fit(kwargs, exception):
    scaler = RobustScaler(**kwargs)
    with pytest.raises(exception):
        scaler.fit([[1.0], [2.0]])


def test_fitted_feature_and_infinity_checks():
    with pytest.raises(ValueError, match="not fitted"):
        RobustScaler().transform([[1.0]])
    scaler = RobustScaler().fit([[1.0, 2.0]])
    with pytest.raises(ValueError, match="expecting 2 features"):
        scaler.transform([[1.0]])
    with pytest.raises(ValueError, match="infinity"):
        RobustScaler().fit([[np.inf]])
