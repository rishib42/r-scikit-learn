import numpy as np
import pytest
from rsklearn.utils import (
    SparseComponents,
    check_array,
    scale_sparse_columns,
    sparse_components,
    sparse_from_components,
    validate_compressed_structure,
)
from rsklearn.utils.validation import check_X_y, validate_data

sparse = pytest.importorskip("scipy.sparse")


@pytest.mark.parametrize(
    "format_name", ["csr", "csc", "coo", "lil", "dok", "bsr", "dia"]
)
def test_check_array_accepts_and_converts_sparse_formats(format_name):
    matrix = getattr(sparse, f"{format_name}_matrix")([[0.0, 1.0], [2.0, 0.0]])
    same = check_array(matrix, accept_sparse=True, dtype=None)
    assert same is matrix
    converted = check_array(matrix, accept_sparse="csr", dtype=np.float32)
    assert converted.format == "csr"
    assert converted.dtype == np.float32
    np.testing.assert_array_equal(converted.toarray(), matrix.toarray())


@pytest.mark.parametrize("constructor", [sparse.csr_array, sparse.csc_array])
def test_check_array_supports_sparse_arrays(constructor):
    matrix = constructor([[0.0, 1.0], [2.0, 0.0]])
    checked = check_array(matrix, accept_sparse=("csr", "csc"), dtype=None)
    assert checked is matrix


def test_check_array_sparse_copy_finite_nonnegative_and_index_width_policies():
    matrix = sparse.csr_matrix([[0.0, np.nan], [2.0, 0.0]])
    with pytest.raises(ValueError, match="NaN or infinity"):
        check_array(matrix, accept_sparse="csr")
    assert (
        check_array(matrix, accept_sparse="csr", ensure_all_finite="allow-nan")
        is matrix
    )
    with pytest.raises(ValueError, match="infinity"):
        check_array(
            sparse.csr_matrix([[np.inf]]),
            accept_sparse="csr",
            ensure_all_finite="allow-nan",
        )
    with pytest.raises(ValueError, match="negative"):
        check_array(
            sparse.csr_matrix([[-1.0]]), accept_sparse=True, ensure_non_negative=True
        )
    assert (
        check_array(
            matrix, accept_sparse="csr", copy=True, ensure_all_finite="allow-nan"
        )
        is not matrix
    )

    large = sparse.csr_matrix([[0.0, 1.0]])
    large.indices = large.indices.astype(np.int64)
    large.indptr = large.indptr.astype(np.int64)
    with pytest.raises(ValueError, match="32-bit"):
        check_array(large, accept_sparse="csr", accept_large_sparse=False)


def test_sparse_components_are_canonical_contiguous_and_rust_validated():
    matrix = sparse.csr_matrix(
        (
            np.asarray([2.0, 1.0, 3.0]),
            np.asarray([1, 0, 1], dtype=np.int32),
            np.asarray([0, 3, 3], dtype=np.int32),
        ),
        shape=(2, 2),
    )
    components = sparse_components(matrix)
    assert components.format == "csr"
    assert components.data.flags.c_contiguous
    assert components.indices.dtype == components.indptr.dtype == np.int64
    rebuilt = sparse_from_components(components)
    np.testing.assert_array_equal(rebuilt.toarray(), matrix.toarray())
    assert rebuilt.has_canonical_format


def test_invalid_compressed_structures_are_rejected_by_rust():
    invalid_pointer = sparse.csr_matrix([[1.0]])
    invalid_pointer.indptr[-1] = 2
    with pytest.raises(ValueError, match="invalid compressed sparse"):
        validate_compressed_structure(invalid_pointer)
    invalid_index = SparseComponents(
        "csr",
        (1, 1),
        np.asarray([1.0]),
        np.asarray([2], dtype=np.int64),
        np.asarray([0, 1], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="outside dimension"):
        sparse_from_components(invalid_index)
    invalid_dtype = SparseComponents(
        "csr",
        (1, 1),
        np.asarray([1.0]),
        np.asarray([0.0]),
        np.asarray([0.0, 1.0]),
    )
    with pytest.raises(TypeError, match="signed integers"):
        sparse_from_components(invalid_dtype)


@pytest.mark.parametrize("format_name", ["csr", "csc"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rust_column_scaling_preserves_sparse_format_dtype_and_input(
    format_name, dtype
):
    matrix = getattr(sparse, f"{format_name}_matrix")(
        np.asarray([[0.0, 6.0], [8.0, 0.0]], dtype=dtype)
    )
    original = matrix.copy()
    scaled = scale_sparse_columns(matrix, [2.0, 3.0])
    assert scaled.format == format_name
    assert scaled.dtype == dtype
    np.testing.assert_allclose(scaled.toarray(), [[0.0, 2.0], [4.0, 0.0]])
    np.testing.assert_array_equal(matrix.toarray(), original.toarray())
    restored = scale_sparse_columns(scaled, [2.0, 3.0], inverse=True)
    np.testing.assert_allclose(restored.toarray(), original.toarray())


def test_sparse_input_remains_rejected_by_dense_only_estimators():
    from rsklearn.preprocessing import OrdinalEncoder, StandardScaler

    matrix = sparse.csr_matrix([[0.0, 1.0], [1.0, 0.0]])
    for estimator in (StandardScaler(), OrdinalEncoder()):
        with pytest.raises(TypeError, match="dense data is required"):
            estimator.fit(matrix)


def test_check_x_y_and_validate_data_preserve_sparse_feature_input():
    class SparseEstimator:
        pass

    matrix = sparse.csr_matrix([[0.0, 1.0], [2.0, 0.0]])
    X, y = check_X_y(matrix, [0, 1], accept_sparse="csr")
    assert X is matrix
    np.testing.assert_array_equal(y, [0, 1])
    estimator = SparseEstimator()
    assert validate_data(estimator, matrix, accept_sparse="csr", reset=True) is matrix
    assert estimator.n_features_in_ == 2


def test_randomized_rust_sparse_scaling_matches_dense_math():
    rng = np.random.default_rng(20260613)
    for rows, columns, density in [(1, 1, 1.0), (20, 7, 0.1), (100, 30, 0.03)]:
        dense = rng.normal(size=(rows, columns))
        dense[rng.random(size=dense.shape) > density] = 0
        scale = rng.uniform(0.5, 3.0, size=columns)
        for format_name in ("csr", "csc"):
            for dtype in (np.float32, np.float64):
                matrix = getattr(sparse, f"{format_name}_matrix")(dense.astype(dtype))
                actual = scale_sparse_columns(matrix, scale)
                np.testing.assert_allclose(
                    actual.toarray(),
                    dense.astype(dtype) / scale.astype(dtype),
                    rtol=1e-6,
                    atol=1e-7,
                )
