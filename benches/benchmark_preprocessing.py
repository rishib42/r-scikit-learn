"""Honest preprocessing benchmark; prints measurements, never canned claims."""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import numpy as np
from rsklearn._validation import validate_numeric_2d
from rsklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    Normalizer,
    RobustScaler,
    StandardScaler,
)

# The scikit-learn distribution intentionally exposes the `sklearn` import package.
from sklearn.preprocessing import LabelEncoder as ScikitLabelEncoder
from sklearn.preprocessing import MinMaxScaler as ScikitMinMaxScaler
from sklearn.preprocessing import Normalizer as ScikitNormalizer
from sklearn.preprocessing import RobustScaler as ScikitRobustScaler
from sklearn.preprocessing import StandardScaler as ScikitStandardScaler


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
    print(
        "\nTimes include Python-to-Rust validation/conversion for "
        "r-scikit-learn public API calls. Positive improvement means "
        "r-scikit-learn is faster."
    )


if __name__ == "__main__":
    main()
