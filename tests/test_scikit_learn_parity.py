import numpy as np
import pytest
from rsklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    Normalizer,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

# The scikit-learn distribution intentionally exposes the `sklearn` import package.
scikit_learn_preprocessing = pytest.importorskip("sklearn.preprocessing")
ScikitLabelEncoder = scikit_learn_preprocessing.LabelEncoder
ScikitMinMaxScaler = scikit_learn_preprocessing.MinMaxScaler
ScikitNormalizer = scikit_learn_preprocessing.Normalizer
ScikitOrdinalEncoder = scikit_learn_preprocessing.OrdinalEncoder
ScikitRobustScaler = scikit_learn_preprocessing.RobustScaler
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
    "labels",
    [
        [3, -1, 3, 2],
        [1.5, -2.0, 1.5],
        ["東京", "café", "東京"],
        [True, False, True],
        [1, "a"],
        [1.0, np.nan, np.inf],
        np.array([2**53, 2**53 + 1], dtype=np.int64),
        np.array([2**63, 2**63 + 1], dtype=np.uint64),
        [],
    ],
)
def test_label_encoder_parity(labels):
    ours = LabelEncoder()
    theirs = ScikitLabelEncoder()
    np.testing.assert_array_equal(
        ours.fit_transform(labels), theirs.fit_transform(labels)
    )
    np.testing.assert_array_equal(ours.classes_, theirs.classes_)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"dtype": np.float32},
        {"handle_unknown": "use_encoded_value", "unknown_value": -1},
        {"encoded_missing_value": -2},
        {"min_frequency": 2},
        {"max_categories": 3},
        {"categories": [["z", "a", "b"], [1, 2, 3, np.nan]]},
    ],
)
def test_ordinal_encoder_parity(options):
    X = np.asarray(
        [["a", 1], ["b", 3], ["a", 2], ["z", 1], ["b", np.nan]],
        dtype=object,
    )
    ours = OrdinalEncoder(**options)
    theirs = ScikitOrdinalEncoder(**options)
    np.testing.assert_allclose(
        ours.fit_transform(X), theirs.fit_transform(X), equal_nan=True
    )


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


@pytest.mark.parametrize("norm", ["l1", "l2", "max"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_normalizer_parity(norm, dtype):
    rng = np.random.default_rng(20260613)
    X = rng.normal(size=(250, 12)).astype(dtype)
    X[0] = 0
    ours = Normalizer(norm=norm).fit_transform(X)
    theirs = ScikitNormalizer(norm=norm).fit_transform(X)
    assert ours.dtype == theirs.dtype == dtype
    np.testing.assert_allclose(ours, theirs, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("norm", ["l1", "l2", "max"])
def test_normalizer_extreme_value_parity(norm):
    X = np.asarray([[0.0, 0.0], [1e-308, 1e-308], [1e308, 1e308], [-3.0, 4.0]])
    np.testing.assert_allclose(
        Normalizer(norm=norm).fit_transform(X),
        ScikitNormalizer(norm=norm).fit_transform(X),
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"with_centering": False},
        {"with_scaling": False},
        {"with_centering": False, "with_scaling": False},
        {"quantile_range": (10.0, 90.0)},
        {"unit_variance": True},
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_robust_scaler_parity(options, dtype):
    rng = np.random.default_rng(20260613)
    X = rng.standard_t(df=2, size=(251, 12)).astype(dtype)
    X[0, 0] = np.nan
    ours = RobustScaler(**options).fit(X)
    theirs = ScikitRobustScaler(**options).fit(X)
    if ours.center_ is not None:
        np.testing.assert_allclose(ours.center_, theirs.center_, rtol=1e-6)
    if ours.scale_ is not None:
        np.testing.assert_allclose(ours.scale_, theirs.scale_, rtol=1e-12)
    ours_transformed = ours.transform(X)
    theirs_transformed = theirs.transform(X)
    assert ours_transformed.dtype == theirs_transformed.dtype == dtype
    np.testing.assert_allclose(
        ours_transformed, theirs_transformed, rtol=1e-5, atol=1e-6, equal_nan=True
    )


@pytest.mark.parametrize("quantile_range", [(0.0, 100.0), (50.0, 50.0), (5.0, 95.0)])
def test_robust_scaler_unit_variance_extreme_quantile_parity(quantile_range):
    X = np.asarray([[1.0], [2.0], [4.0], [100.0]])
    with np.errstate(all="ignore"):
        ours = RobustScaler(quantile_range=quantile_range, unit_variance=True).fit(X)
        theirs = ScikitRobustScaler(
            quantile_range=quantile_range, unit_variance=True
        ).fit(X)
    np.testing.assert_allclose(ours.scale_, theirs.scale_)


@pytest.mark.parametrize(
    "ours_class,theirs_class",
    [
        (StandardScaler, ScikitStandardScaler),
        (MinMaxScaler, ScikitMinMaxScaler),
    ],
)
def test_nan_dtype_and_partial_fit_parity(ours_class, theirs_class):
    X = np.asarray(
        [[1.0, np.nan, np.nan], [2.0, 4.0, np.nan], [5.0, 8.0, np.nan]],
        dtype=np.float32,
    )
    ours = ours_class().partial_fit(X[:1]).partial_fit(X[1:])
    theirs = theirs_class().partial_fit(X[:1]).partial_fit(X[1:])
    ours_transformed = ours.transform(X)
    theirs_transformed = theirs.transform(X)
    assert ours_transformed.dtype == theirs_transformed.dtype == np.float32
    np.testing.assert_allclose(
        ours_transformed, theirs_transformed, rtol=1e-6, atol=1e-6, equal_nan=True
    )
    np.testing.assert_array_equal(ours.n_samples_seen_, theirs.n_samples_seen_)
