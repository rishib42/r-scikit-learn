import numpy as np
import pytest
from rsklearn.preprocessing import MinMaxScaler


def test_minmax_attributes_roundtrip_and_constant_feature():
    X = np.array([[-2.0, 4.0], [0.0, 4.0], [2.0, 4.0]])
    scaler = MinMaxScaler(feature_range=(-1.0, 2.0))
    transformed = scaler.fit_transform(X)
    np.testing.assert_allclose(transformed[:, 0], [-1.0, 0.5, 2.0])
    np.testing.assert_allclose(transformed[:, 1], [-1.0, -1.0, -1.0])
    np.testing.assert_allclose(scaler.inverse_transform(transformed), X)
    np.testing.assert_allclose(scaler.data_min_, [-2.0, 4.0])
    np.testing.assert_allclose(scaler.data_max_, [2.0, 4.0])
    np.testing.assert_allclose(scaler.data_range_, [4.0, 0.0])
    assert scaler.n_features_in_ == 2
    assert scaler.n_samples_seen_ == 3


def test_minmax_outside_range_and_clip():
    X = [[0.0], [10.0]]
    np.testing.assert_allclose(
        MinMaxScaler().fit(X).transform([[-5.0], [15.0]]), [[-0.5], [1.5]]
    )
    np.testing.assert_allclose(
        MinMaxScaler(clip=True).fit(X).transform([[-5.0], [15.0]]), [[0.0], [1.0]]
    )


def test_minmax_partial_fit_matches_complete_fit():
    X = np.array([[1.0, 3.0], [2.0, 4.0], [5.0, 8.0]])
    incremental = MinMaxScaler().partial_fit(X[:2]).partial_fit(X[2:])
    complete = MinMaxScaler().fit(X)
    np.testing.assert_allclose(incremental.data_min_, complete.data_min_)
    np.testing.assert_allclose(incremental.data_max_, complete.data_max_)
    np.testing.assert_allclose(incremental.scale_, complete.scale_)
    assert incremental.n_samples_seen_ == complete.n_samples_seen_


@pytest.mark.parametrize(
    "kwargs,exception",
    [
        ({"feature_range": (1, 1)}, ValueError),
        ({"feature_range": (2, 1)}, ValueError),
        ({"feature_range": (0,)}, TypeError),
        ({"feature_range": (0, np.inf)}, ValueError),
        ({"clip": 1}, TypeError),
    ],
)
def test_minmax_constructor_validation(kwargs, exception):
    with pytest.raises(exception):
        MinMaxScaler(**kwargs)


def test_minmax_params_repr_and_state_checks():
    scaler = MinMaxScaler().set_params(feature_range=(-2.0, 2.0), clip=True)
    assert scaler.get_params() == {"feature_range": (-2.0, 2.0), "clip": True}
    assert "feature_range=(-2.0, 2.0)" in repr(scaler)
    with pytest.raises(ValueError, match="not fitted"):
        MinMaxScaler().inverse_transform([[0.0]])
    scaler.fit([[1.0, 2.0]])
    with pytest.raises(ValueError, match="expected 2 features"):
        scaler.transform([[1.0]])
    with pytest.raises(ValueError, match="expected 2 features"):
        scaler.partial_fit([[1.0]])
