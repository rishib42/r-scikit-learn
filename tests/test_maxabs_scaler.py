import numpy as np
import pytest
from rsklearn.preprocessing import MaxAbsScaler

sparse = pytest.importorskip("scipy.sparse")


@pytest.mark.parametrize("format_name", ["csr", "csc"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_maxabs_scaler_sparse_preserves_format_dtype_and_roundtrips(format_name, dtype):
    X = getattr(sparse, f"{format_name}_matrix")(
        np.asarray(
            [[0.0, 2.0, 0.0], [-4.0, 0.0, np.nan], [0.0, -8.0, 0.0]], dtype=dtype
        )
    )
    original = X.copy()
    scaler = MaxAbsScaler().fit(X)
    transformed = scaler.transform(X)
    assert transformed.format == format_name
    assert transformed.dtype == dtype
    np.testing.assert_allclose(scaler.max_abs_, [4.0, 8.0, 0.0])
    np.testing.assert_allclose(scaler.scale_, [4.0, 8.0, 1.0])
    np.testing.assert_array_equal(X.toarray(), original.toarray())
    np.testing.assert_allclose(
        scaler.inverse_transform(transformed).toarray(),
        X.toarray(),
        equal_nan=True,
        rtol=1e-6,
        atol=1e-7,
    )


def test_maxabs_scaler_dense_attributes_and_no_mutation():
    X = np.asarray([[0.0, 2.0, 0.0], [-4.0, 0.0, np.nan], [0.0, -8.0, 0.0]])
    original = X.copy()
    scaler = MaxAbsScaler().fit(X)
    transformed = scaler.transform(X)
    np.testing.assert_allclose(scaler.max_abs_, [4.0, 8.0, 0.0])
    np.testing.assert_allclose(scaler.scale_, [4.0, 8.0, 1.0])
    np.testing.assert_allclose(scaler.inverse_transform(transformed), X, equal_nan=True)
    np.testing.assert_array_equal(X, original)


def test_maxabs_scaler_partial_fit_matches_complete_fit():
    X = sparse.csr_matrix([[0.0, 2.0], [-4.0, 0.0], [0.0, -8.0]])
    incremental = MaxAbsScaler().partial_fit(X[:1]).partial_fit(X[1:])
    complete = MaxAbsScaler().fit(X)
    np.testing.assert_allclose(incremental.max_abs_, complete.max_abs_)
    np.testing.assert_allclose(incremental.scale_, complete.scale_)
    assert incremental.n_samples_seen_ == complete.n_samples_seen_


def test_maxabs_scaler_fitted_feature_and_params_checks():
    with pytest.raises(ValueError, match="not fitted"):
        MaxAbsScaler().transform([[1.0]])
    scaler = MaxAbsScaler().fit([[1.0, 2.0]])
    with pytest.raises(ValueError, match="expecting 2 features"):
        scaler.transform([[1.0]])
    assert scaler.get_params() == {}
    assert repr(scaler) == "MaxAbsScaler()"
