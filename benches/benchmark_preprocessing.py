"""Honest preprocessing benchmark; prints measurements, never canned claims."""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import numpy as np
from rsklearn._validation import validate_numeric_2d
from rsklearn.base import BaseEstimator
from rsklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    Normalizer,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)
from rsklearn.preprocessing._categorical import discover_categories, encode_categories
from rsklearn.utils import check_array, scale_sparse_columns, sparse_components
from scipy import sparse

# The scikit-learn distribution intentionally exposes the `sklearn` import package.
from sklearn.preprocessing import LabelEncoder as ScikitLabelEncoder
from sklearn.preprocessing import MinMaxScaler as ScikitMinMaxScaler
from sklearn.preprocessing import Normalizer as ScikitNormalizer
from sklearn.preprocessing import OrdinalEncoder as ScikitOrdinalEncoder
from sklearn.preprocessing import RobustScaler as ScikitRobustScaler
from sklearn.preprocessing import StandardScaler as ScikitStandardScaler
from sklearn.utils.sparsefuncs import inplace_column_scale
from sklearn.utils.validation import check_array as ScikitCheckArray


def measure(function: Callable[[], object], repetitions: int) -> tuple[float, float]:
    function()
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    return statistics.mean(samples), statistics.stdev(
        samples
    ) if repetitions > 1 else 0.0


def report_single(name: str, function: Callable[[], object], repetitions: int) -> None:
    mean, deviation = measure(function, repetitions)
    print(f"{name:46} {mean:9.6f}s mean  {deviation:9.6f}s stdev")


def report_comparison(
    name: str,
    r_scikit_learn_function: Callable[[], object],
    scikit_learn_function: Callable[[], object],
    repetitions: int,
) -> None:
    r_scikit_learn_mean, r_scikit_learn_deviation = measure(
        r_scikit_learn_function, repetitions
    )
    scikit_learn_mean, scikit_learn_deviation = measure(
        scikit_learn_function, repetitions
    )
    improvement = (scikit_learn_mean - r_scikit_learn_mean) / scikit_learn_mean * 100
    print(
        f"{name:38} "
        f"r-scikit-learn {r_scikit_learn_mean:9.6f}s ± "
        f"{r_scikit_learn_deviation:9.6f}s  "
        f"scikit-learn {scikit_learn_mean:9.6f}s ± "
        f"{scikit_learn_deviation:9.6f}s  "
        f"impr. {improvement:+7.2f}%"
    )


def benchmark_matrix(rows: int, columns: int, repetitions: int) -> None:
    rng = np.random.default_rng(42)
    X = np.ascontiguousarray(rng.normal(size=(rows, columns)))
    print(f"\nMatrix: {rows:,} x {columns}")
    report_single(
        "r-scikit-learn input validation/conversion",
        lambda: validate_numeric_2d(X, estimator="Benchmark"),
        repetitions,
    )
    for name, ours, theirs in [
        ("StandardScaler", StandardScaler, ScikitStandardScaler),
        ("MinMaxScaler", MinMaxScaler, ScikitMinMaxScaler),
        ("Normalizer", Normalizer, ScikitNormalizer),
        ("RobustScaler", RobustScaler, ScikitRobustScaler),
    ]:
        ours_fitted = ours().fit(X)
        theirs_fitted = theirs().fit(X)
        report_comparison(
            f"{name} end-to-end",
            lambda c=ours: c().fit_transform(X),
            lambda c=theirs: c().fit_transform(X),
            repetitions,
        )
        report_comparison(
            f"{name} transform",
            lambda f=ours_fitted: f.transform(X),
            lambda f=theirs_fitted: f.transform(X),
            repetitions,
        )
        report_comparison(
            f"{name} fit",
            lambda c=ours: c().fit(X),
            lambda c=theirs: c().fit(X),
            repetitions,
        )
    X_float32 = X.astype(np.float32)
    ours_float32 = Normalizer().fit(X_float32)
    theirs_float32 = ScikitNormalizer().fit(X_float32)
    report_comparison(
        "Normalizer float32 transform",
        lambda: ours_float32.transform(X_float32),
        lambda: theirs_float32.transform(X_float32),
        repetitions,
    )
    ours_robust_float32 = RobustScaler().fit(X_float32)
    theirs_robust_float32 = ScikitRobustScaler().fit(X_float32)
    report_comparison(
        "RobustScaler float32 transform",
        lambda: ours_robust_float32.transform(X_float32),
        lambda: theirs_robust_float32.transform(X_float32),
        repetitions,
    )


def benchmark_labels(repetitions: int) -> None:
    rng = np.random.default_rng(42)
    numeric = rng.integers(0, 10_000, size=1_000_000)
    strings = np.asarray([f"label-{value}" for value in numeric])
    print("\nLabels: 1,000,000")
    for label_type, labels in [("numeric", numeric), ("string", strings)]:
        ours = LabelEncoder().fit(labels)
        theirs = ScikitLabelEncoder().fit(labels)
        report_comparison(
            f"LabelEncoder {label_type} fit_transform",
            lambda y=labels: LabelEncoder().fit_transform(y),
            lambda y=labels: ScikitLabelEncoder().fit_transform(y),
            repetitions,
        )
        report_comparison(
            f"LabelEncoder {label_type} transform",
            lambda y=labels, encoder=ours: encoder.transform(y),
            lambda y=labels, encoder=theirs: encoder.transform(y),
            repetitions,
        )


class _CategoricalBenchmarkEstimator(BaseEstimator):
    pass


def benchmark_categories(repetitions: int) -> None:
    rng = np.random.default_rng(42)
    numeric = rng.integers(0, 1_000, size=(100_000, 4), dtype=np.int64)
    strings = np.asarray(
        [[f"category-{value}" for value in row] for row in numeric],
        dtype=str,
    )
    print("\nCategorical matrix: 100,000 x 4")
    for category_type, X in [("numeric", numeric), ("unicode", strings)]:
        ours_estimator = _CategoricalBenchmarkEstimator()
        state, _ = discover_categories(X, estimator=ours_estimator)
        theirs = ScikitOrdinalEncoder().fit(X)
        ours_public = OrdinalEncoder().fit(X)
        report_comparison(
            f"Category discovery + encoding {category_type}",
            lambda values=X: discover_categories(
                values, estimator=_CategoricalBenchmarkEstimator()
            ),
            lambda values=X: ScikitOrdinalEncoder().fit_transform(values),
            repetitions,
        )
        report_comparison(
            f"Category lookup {category_type}",
            lambda values=X, learned=state, estimator=ours_estimator: encode_categories(
                values, learned, estimator=estimator
            ),
            lambda values=X, encoder=theirs: encoder.transform(values),
            repetitions,
        )
        report_comparison(
            f"OrdinalEncoder fit_transform {category_type}",
            lambda values=X: OrdinalEncoder().fit_transform(values),
            lambda values=X: ScikitOrdinalEncoder().fit_transform(values),
            repetitions,
        )
        report_comparison(
            f"OrdinalEncoder transform {category_type}",
            lambda values=X, encoder=ours_public: encoder.transform(values),
            lambda values=X, encoder=theirs: encoder.transform(values),
            repetitions,
        )


def benchmark_sparse(repetitions: int) -> None:
    rng = np.random.default_rng(42)
    matrix = sparse.random(
        1_000_000,
        100,
        density=0.001,
        format="csr",
        random_state=rng,
        dtype=np.float64,
    )
    scale = rng.uniform(0.5, 2.0, size=matrix.shape[1])

    def scikit_scale() -> object:
        output = matrix.copy()
        inplace_column_scale(output, 1.0 / scale)
        return output

    print(
        f"\nSparse CSR matrix: {matrix.shape[0]:,} x {matrix.shape[1]} "
        f"({matrix.nnz:,} nnz)"
    )
    report_comparison(
        "Sparse check_array validation",
        lambda: check_array(matrix, accept_sparse="csr"),
        lambda: ScikitCheckArray(matrix, accept_sparse="csr"),
        repetitions,
    )
    report_single(
        "Rust-validated sparse components",
        lambda: sparse_components(matrix),
        repetitions,
    )
    report_comparison(
        "Sparse column scaling",
        lambda: scale_sparse_columns(matrix, scale),
        scikit_scale,
        repetitions,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--include-largest", action="store_true")
    args = parser.parse_args()
    for shape in [(1_000, 10), (100_000, 20)]:
        benchmark_matrix(*shape, args.repetitions)
    if args.include_largest:
        benchmark_matrix(1_000_000, 10, args.repetitions)
    benchmark_labels(args.repetitions)
    benchmark_categories(args.repetitions)
    benchmark_sparse(args.repetitions)
    print(
        "\nTimes include Python-to-Rust validation/conversion for "
        "r-scikit-learn public API calls. Positive improvement means "
        "r-scikit-learn is faster."
    )


if __name__ == "__main__":
    main()
