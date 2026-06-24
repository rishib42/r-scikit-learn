"""Compare r-scikit-learn and scikit-learn nearest-neighbor performance."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Callable

import numpy as np
import rsklearn.neighbors as rneighbors
import scipy
import sklearn
import sklearn.neighbors as sneighbors
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
        f"{name:<32} rsklearn {ours_mean:9.6f}s ± {ours_stdev:9.6f}s  "
        f"sklearn {theirs_mean:9.6f}s ± {theirs_stdev:9.6f}s  "
        f"impr. {improvement:+7.2f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=int, default=20_000)
    parser.add_argument("--query-samples", type=int, default=1_000)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
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
    print(
        f"Dependencies: numpy {np.__version__}, scipy {scipy.__version__}, "
        f"scikit-learn {sklearn.__version__}"
    )
    rng = np.random.default_rng(20260616)
    X_train = rng.normal(size=(args.train_samples, args.features))
    X_query = rng.normal(size=(args.query_samples, args.features))
    y = rng.integers(0, args.classes, size=args.train_samples, dtype=np.int64)
    y_regression = rng.normal(size=args.train_samples)
    options = {
        "n_neighbors": args.neighbors,
        "weights": "uniform",
        "algorithm": "brute",
        "metric": "euclidean",
    }
    print(
        f"Train matrix: {args.train_samples:,} x {args.features:,}; "
        f"query matrix: {args.query_samples:,} x {args.features:,}"
    )
    report(
        "KNeighborsClassifier fit",
        lambda: rneighbors.KNeighborsClassifier(**options).fit(X_train, y),
        lambda: sneighbors.KNeighborsClassifier(**options).fit(X_train, y),
        args.repetitions,
        args.warmups,
    )
    ours = rneighbors.KNeighborsClassifier(**options).fit(X_train, y)
    theirs = sneighbors.KNeighborsClassifier(**options).fit(X_train, y)
    report(
        "KNeighborsClassifier kneighbors",
        lambda: ours.kneighbors(X_query),
        lambda: theirs.kneighbors(X_query),
        args.repetitions,
        args.warmups,
    )
    report(
        "KNeighborsClassifier predict",
        lambda: ours.predict(X_query),
        lambda: theirs.predict(X_query),
        args.repetitions,
        args.warmups,
    )
    report(
        "KNeighborsClassifier proba",
        lambda: ours.predict_proba(X_query),
        lambda: theirs.predict_proba(X_query),
        args.repetitions,
        args.warmups,
    )
    report(
        "KNeighborsRegressor fit",
        lambda: rneighbors.KNeighborsRegressor(**options).fit(X_train, y_regression),
        lambda: sneighbors.KNeighborsRegressor(**options).fit(X_train, y_regression),
        args.repetitions,
        args.warmups,
    )
    ours_regressor = rneighbors.KNeighborsRegressor(**options).fit(
        X_train, y_regression
    )
    theirs_regressor = sneighbors.KNeighborsRegressor(**options).fit(
        X_train, y_regression
    )
    report(
        "KNeighborsRegressor kneighbors",
        lambda: ours_regressor.kneighbors(X_query),
        lambda: theirs_regressor.kneighbors(X_query),
        args.repetitions,
        args.warmups,
    )
    report(
        "KNeighborsRegressor predict",
        lambda: ours_regressor.predict(X_query),
        lambda: theirs_regressor.predict(X_query),
        args.repetitions,
        args.warmups,
    )


if __name__ == "__main__":
    main()
