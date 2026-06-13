"""Validated SciPy CSR/CSC infrastructure for sparse-aware estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rsklearn import _core
from rsklearn.utils.validation import check_array


@dataclass(frozen=True)
class SparseComponents:
    """Canonical compressed sparse arrays validated for safe Rust kernels."""

    format: str
    shape: tuple[int, int]
    data: NDArray[Any]
    indices: NDArray[np.int64]
    indptr: NDArray[np.int64]


def sparse_components(
    matrix: Any,
    *,
    accept_sparse: str | list[str] | tuple[str, ...] = ("csr", "csc"),
    dtype: Any = "numeric",
    copy: bool = False,
    ensure_all_finite: bool | str = True,
    ensure_non_negative: bool = False,
) -> SparseComponents:
    """Return canonical CSR/CSC components after Python and Rust validation."""
    checked = check_array(
        matrix,
        accept_sparse=accept_sparse,
        dtype=dtype,
        copy=copy,
        ensure_all_finite=ensure_all_finite,
        ensure_non_negative=ensure_non_negative,
    )
    if checked.format not in ("csr", "csc"):
        checked = checked.tocsr()
    validate_compressed_structure(checked)
    if not checked.has_canonical_format or not checked.has_sorted_indices:
        checked = checked.copy()
        checked.sum_duplicates()
        checked.sort_indices()
    validate_compressed_structure(checked)
    return SparseComponents(
        checked.format,
        checked.shape,
        np.ascontiguousarray(checked.data),
        np.ascontiguousarray(checked.indices, dtype=np.int64),
        np.ascontiguousarray(checked.indptr, dtype=np.int64),
    )


def validate_compressed_structure(matrix: Any) -> None:
    """Validate CSR/CSC index arrays in safe Rust."""
    if getattr(matrix, "format", None) not in ("csr", "csc"):
        raise TypeError("compressed sparse structure must be CSR or CSC")
    major_dimension, minor_dimension = (
        matrix.shape if matrix.format == "csr" else matrix.shape[::-1]
    )
    _validate_component_arrays(
        matrix.indices,
        matrix.indptr,
        major_dimension,
        minor_dimension,
        matrix.data.size,
    )


def _validate_component_arrays(
    indices: Any,
    indptr: Any,
    major_dimension: int,
    minor_dimension: int,
    nnz: int,
) -> None:
    indices = np.asarray(indices)
    indptr = np.asarray(indptr)
    if (
        indices.ndim != 1
        or indptr.ndim != 1
        or not np.issubdtype(indices.dtype, np.signedinteger)
        or not np.issubdtype(indptr.dtype, np.signedinteger)
    ):
        raise TypeError(
            "sparse indices and indptr must be one-dimensional signed integers"
        )
    indices = np.ascontiguousarray(indices)
    indptr = np.ascontiguousarray(indptr)
    function = (
        _core.sparse_validate_i32
        if indices.dtype == np.dtype(np.int32) and indptr.dtype == np.dtype(np.int32)
        else _core.sparse_validate_i64
    )
    function(
        indices.astype(np.int32 if function is _core.sparse_validate_i32 else np.int64),
        indptr.astype(np.int32 if function is _core.sparse_validate_i32 else np.int64),
        major_dimension,
        minor_dimension,
        nnz,
    )


def scale_sparse_columns(
    matrix: Any, scale: Any, *, inverse: bool = False, copy: bool = True
) -> Any:
    """Scale stored CSR/CSC values by feature without densifying."""
    checked = check_array(
        matrix,
        accept_sparse=("csr", "csc"),
        dtype="numeric",
        copy=copy,
        ensure_all_finite="allow-nan",
    )
    original_format = checked.format
    if original_format == "csc":
        checked = checked.tocsr(copy=False)
    factors = np.asarray(scale)
    if factors.ndim != 1 or factors.size != checked.shape[1]:
        raise ValueError("scale must contain one value per sparse feature")
    if checked.data.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        checked = checked.astype(np.float64)
    index_width = "i32" if checked.indices.dtype == np.dtype(np.int32) else "i64"
    value_width = "f32" if checked.data.dtype == np.dtype(np.float32) else "f64"
    function = getattr(_core, f"sparse_scale_csr_{index_width}_{value_width}")
    function(
        checked.data,
        checked.indices,
        np.ascontiguousarray(factors, dtype=checked.data.dtype),
        inverse,
    )
    return checked.tocsc(copy=False) if original_format == "csc" else checked


def sparse_from_components(
    components: SparseComponents, *, canonicalize: bool = True
) -> Any:
    """Construct a validated SciPy CSR/CSC matrix from compressed components."""
    if components.format not in ("csr", "csc"):
        raise ValueError("sparse component format must be 'csr' or 'csc'")
    if len(components.shape) != 2 or any(
        isinstance(dimension, (bool, np.bool_))
        or not isinstance(dimension, (int, np.integer))
        or dimension < 0
        for dimension in components.shape
    ):
        raise ValueError(
            "sparse component shape must contain two non-negative dimensions"
        )
    data = np.asarray(components.data)
    if data.ndim != 1:
        raise ValueError("sparse component data must be one-dimensional")
    major_dimension, minor_dimension = (
        components.shape if components.format == "csr" else components.shape[::-1]
    )
    _validate_component_arrays(
        components.indices,
        components.indptr,
        major_dimension,
        minor_dimension,
        data.size,
    )
    from scipy import sparse

    constructor = sparse.csr_matrix if components.format == "csr" else sparse.csc_matrix
    matrix = constructor(
        (
            np.ascontiguousarray(data),
            np.ascontiguousarray(components.indices),
            np.ascontiguousarray(components.indptr),
        ),
        shape=components.shape,
        copy=False,
    )
    validate_compressed_structure(matrix)
    if canonicalize:
        matrix.sum_duplicates()
        matrix.sort_indices()
    return matrix


__all__ = [
    "SparseComponents",
    "scale_sparse_columns",
    "sparse_from_components",
    "sparse_components",
    "validate_compressed_structure",
]
