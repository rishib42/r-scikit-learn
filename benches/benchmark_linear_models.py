"""Compare r-scikit-learn and scikit-learn dense linear-model performance."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Callable

import numpy as np
import rsklearn.linear_model as rlinear
import sklearn.linear_model as slinear
from rsklearn import _core


def measure(
    function: Callable[[], object], repetitions: int, warmups: int
) -> tuple[float, float]:
    for _ in range(warmups):
        function()
    values = []
    for _ in range(repetitions):
        started = time.perf_counter()
        function()
        values.append(time.perf_counter() - started)
    return statistics.mean(values), statistics.stdev(values) if repetitions > 1 else 0


def report(
    name: str,
    ours: Callable[[], object],
    theirs: Callable[[], object],
    repetitions: int,
    warmups: int,
) -> None:
    ours_mean, ours_stdev = measure(ours, repetitions, warmups)
    theirs_mean, theirs_stdev = measure(theirs, repetitions, warmups)
    improvement = (theirs_mean - ours_mean) / theirs_mean * 100
    print(
        f"{name:<32} r-scikit-learn {ours_mean:9.6f}s ± {ours_stdev:9.6f}s  "
        f"scikit-learn {theirs_mean:9.6f}s ± {theirs_stdev:9.6f}s  "
        f"impr. {improvement:+7.2f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument(
        "--allow-debug",
        action="store_true",
        help="run even when r-scikit-learn's Rust extension is a debug build",
    )
    args = parser.parse_args()
    profile = _core.build_profile()
    if profile != "release" and not args.allow_debug:
        raise SystemExit(
            "Refusing to benchmark a debug Rust extension. Install a release build "
            "with `maturin develop --release`, then rerun. Pass --allow-debug only "
            "when intentionally measuring debug code."
        )
    print(f"Python: {sys.executable}")
    print(f"Rust extension: {_core.__file__} ({profile})")
    rng = np.random.default_rng(20260614)
    X = rng.normal(size=(args.samples, args.features))
    coefficients = rng.normal(size=args.features)
    y_regression = X @ coefficients + rng.normal(scale=0.1, size=args.samples)
    y_classification = (X @ coefficients + rng.normal(size=args.samples) > 0).astype(
        np.int64
    )
    print(f"Matrix: {args.samples:,} x {args.features:,}")
    report(
        "LinearRegression fit",
        lambda: rlinear.LinearRegression().fit(X, y_regression),
        lambda: slinear.LinearRegression().fit(X, y_regression),
        args.repetitions,
        args.warmups,
    )
    report(
        "Ridge fit",
        lambda: rlinear.Ridge(alpha=1.0, solver="svd").fit(X, y_regression),
        lambda: slinear.Ridge(alpha=1.0, solver="svd").fit(X, y_regression),
        args.repetitions,
        args.warmups,
    )
    report(
        "Lasso fit",
        lambda: rlinear.Lasso(alpha=0.05).fit(X, y_regression),
        lambda: slinear.Lasso(alpha=0.05).fit(X, y_regression),
        args.repetitions,
        args.warmups,
    )
    report(
        "ElasticNet fit",
        lambda: rlinear.ElasticNet(alpha=0.05, l1_ratio=0.5).fit(X, y_regression),
        lambda: slinear.ElasticNet(alpha=0.05, l1_ratio=0.5).fit(X, y_regression),
        args.repetitions,
        args.warmups,
    )
    report(
        "LogisticRegression fit",
        lambda: rlinear.LogisticRegression(max_iter=500).fit(X, y_classification),
        lambda: slinear.LogisticRegression(max_iter=500).fit(X, y_classification),
        args.repetitions,
        args.warmups,
    )
    report(
        "LogisticRegression L1 fit",
        lambda: rlinear.LogisticRegression(
            penalty="l1", solver="saga", max_iter=500
        ).fit(X, y_classification),
        lambda: slinear.LogisticRegression(
            penalty="l1", solver="saga", max_iter=500
        ).fit(X, y_classification),
        args.repetitions,
        args.warmups,
    )
    report(
        "LogisticRegression elastic-net fit",
        lambda: rlinear.LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=500
        ).fit(X, y_classification),
        lambda: slinear.LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=500
        ).fit(X, y_classification),
        args.repetitions,
        args.warmups,
    )
    ours = rlinear.LinearRegression().fit(X, y_regression)
    theirs = slinear.LinearRegression().fit(X, y_regression)
    report(
        "LinearRegression predict",
        lambda: ours.predict(X),
        lambda: theirs.predict(X),
        args.repetitions,
        args.warmups,
    )


if __name__ == "__main__":
    main()
