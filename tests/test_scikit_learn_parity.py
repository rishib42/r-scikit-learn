import numpy as np
import pytest
from rsklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# The scikit-learn distribution intentionally exposes the `sklearn` import package.
scikit_learn_preprocessing = pytest.importorskip("sklearn.preprocessing")
ScikitLabelEncoder = scikit_learn_preprocessing.LabelEncoder
ScikitMinMaxScaler = scikit_learn_preprocessing.MinMaxScaler
ScikitStandardScaler = scikit_learn_preprocessing.StandardScaler


@pytest.mark.parametrize(
    "X",
    [
        np.array([[1.0], [2.0], [3.0]]),
        np.array([[1.0, -5.0], [1.0, 0.0], [1.0, 5.0]]),
        np.array([[1e-9, 1e9], [2e-9, 2e9], [3e-9, 3e9]]),
        np.array([[2.0, -7.0]]),
    ],
)
@pytest.mark.parametrize(
    "with_mean,with_std", [(True, True), (False, True), (True, False)]
)
def test_standard_scaler_parity(X, with_mean, with_std):
    ours = StandardScaler(with_mean=with_mean, with_std=with_std)
    theirs = ScikitStandardScaler(with_mean=with_mean, with_std=with_std)
    np.testing.assert_allclose(
        ours.fit_transform(X), theirs.fit_transform(X), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(ours.mean_, theirs.mean_, rtol=1e-12, atol=1e-12)
    if with_std:
        np.testing.assert_allclose(ours.var_, theirs.var_, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("feature_range", [(0.0, 1.0), (-2.0, 3.0)])
@pytest.mark.parametrize("clip", [False, True])
def test_minmax_scaler_parity(feature_range, clip):
    train = np.array([[-3.0, 4.0], [1.0, 4.0], [5.0, 4.0]])
    test = np.array([[-5.0, 4.0], [7.0, 4.0]])
    ours = MinMaxScaler(feature_range=feature_range, clip=clip).fit(train)
    theirs = ScikitMinMaxScaler(feature_range=feature_range, clip=clip).fit(train)
    np.testing.assert_allclose(
        ours.transform(test), theirs.transform(test), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(ours.inverse_transform(ours.transform(train)), train)


@pytest.mark.parametrize(
    "labels", [[3, -1, 3, 2], [1.5, -2.0, 1.5], ["東京", "café", "東京"]]
)
def test_label_encoder_parity(labels):
    ours = LabelEncoder()
    theirs = ScikitLabelEncoder()
    np.testing.assert_array_equal(
        ours.fit_transform(labels), theirs.fit_transform(labels)
    )
    np.testing.assert_array_equal(ours.classes_, theirs.classes_)


def test_randomized_scaler_parity():
    rng = np.random.default_rng(20260613)
    for rows, columns in [(1, 1), (20, 5), (250, 12)]:
        X = rng.normal(size=(rows, columns))
        np.testing.assert_allclose(
            StandardScaler().fit_transform(X),
            ScikitStandardScaler().fit_transform(X),
            rtol=1e-11,
            atol=1e-11,
        )
        np.testing.assert_allclose(
            MinMaxScaler(feature_range=(-3, 7)).fit_transform(X),
            ScikitMinMaxScaler(feature_range=(-3, 7)).fit_transform(X),
            rtol=1e-11,
            atol=1e-11,
        )
