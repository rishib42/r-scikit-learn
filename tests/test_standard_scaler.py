import numpy as np
import pytest
from rsklearn.preprocessing import StandardScaler


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"with_mean": False},
        {"with_std": False},
        {"with_mean": False, "with_std": False},
    ],
)
def test_standard_scaler_roundtrip_and_no_mutation(options):
    X = np.array([[1.0, -5.0, 2.0], [2.0, 0.0, 2.0], [3.0, 5.0, 2.0]])
    original = X.copy()
    scaler = StandardScaler(**options)
    transformed = scaler.fit_transform(X)
    np.testing.assert_allclose(scaler.inverse_transform(transformed), X)
    np.testing.assert_array_equal(X, original)
    assert transformed.dtype == np.float64
    assert transformed is not X


def test_standard_scaler_attributes_and_options():
    scaler = StandardScaler().fit([[1, 10], [3, 10]])
    np.testing.assert_allclose(scaler.mean_, [2, 10])
    np.testing.assert_allclose(scaler.var_, [1, 0])
    np.testing.assert_allclose(scaler.scale_, [1, 1])
    assert scaler.n_features_in_ == 2
    assert scaler.n_samples_seen_ == 2
    assert StandardScaler(with_std=False).fit([[1], [2]]).scale_ is None
    assert StandardScaler(with_mean=False, with_std=False).fit([[1], [2]]).mean_ is None


def test_standard_scaler_single_row():
    scaler = StandardScaler()
    np.testing.assert_array_equal(scaler.fit_transform([[4.0, -2.0]]), [[0.0, 0.0]])
    np.testing.assert_array_equal(scaler.scale_, [1.0, 1.0])


def test_standard_scaler_partial_fit_matches_complete_fit():
    X = np.array([[1.0, 3.0], [2.0, 4.0], [5.0, 8.0]])
    incremental = StandardScaler().partial_fit(X[:2]).partial_fit(X[2:])
    complete = StandardScaler().fit(X)
    np.testing.assert_allclose(incremental.mean_, complete.mean_)
    np.testing.assert_allclose(incremental.var_, complete.var_)
    np.testing.assert_allclose(incremental.scale_, complete.scale_)
    np.testing.assert_array_equal(incremental.n_samples_seen_, complete.n_samples_seen_)


def test_standard_scaler_fitted_and_feature_checks():
    with pytest.raises(ValueError, match="not fitted"):
        StandardScaler().transform([[1.0]])
    scaler = StandardScaler().fit([[1.0, 2.0]])
    with pytest.raises(ValueError, match="expected 2 features"):
        scaler.transform([[1.0]])
    with pytest.raises(ValueError, match="expected 2 features"):
        scaler.partial_fit([[1.0]])


def test_standard_scaler_params_and_repr():
    scaler = StandardScaler().set_params(with_mean=False)
    assert scaler.get_params() == {"with_mean": False, "with_std": True}
    assert repr(scaler) == "StandardScaler(with_mean=False, with_std=True)"
    with pytest.raises(ValueError, match="Invalid parameter"):
        scaler.set_params(no_such_parameter=True)
    with pytest.raises(TypeError, match="must be bool"):
        StandardScaler(with_mean=1)
