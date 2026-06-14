"""Compare r-scikit-learn and scikit-learn metric performance."""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import numpy as np
import rsklearn.metrics as rmetrics
import sklearn.metrics as smetrics


def measure(function: Callable[[], object], repetitions: int) -> tuple[float, float]:
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
) -> None:
    ours_mean, ours_stdev = measure(ours, repetitions)
    theirs_mean, theirs_stdev = measure(theirs, repetitions)
    improvement = (theirs_mean - ours_mean) / theirs_mean * 100
    print(
        f"{name:<32} r-scikit-learn {ours_mean:9.6f}s ± {ours_stdev:9.6f}s  "
        f"scikit-learn {theirs_mean:9.6f}s ± {theirs_stdev:9.6f}s  "
        f"impr. {improvement:+7.2f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()
    rng = np.random.default_rng(20260614)
    expected_class = rng.integers(0, 20, size=args.samples)
    predicted_class = rng.integers(0, 20, size=args.samples)
    expected_regression = rng.normal(size=(args.samples, 4))
    predicted_regression = expected_regression + rng.normal(
        scale=0.5, size=expected_regression.shape
    )
    print(f"Samples: {args.samples:,}")
    report(
        "accuracy_score",
        lambda: rmetrics.accuracy_score(expected_class, predicted_class),
        lambda: smetrics.accuracy_score(expected_class, predicted_class),
        args.repetitions,
    )
    report(
        "confusion_matrix",
        lambda: rmetrics.confusion_matrix(expected_class, predicted_class),
        lambda: smetrics.confusion_matrix(expected_class, predicted_class),
        args.repetitions,
    )
    report(
        "precision_score macro",
        lambda: rmetrics.precision_score(
            expected_class, predicted_class, average="macro", zero_division=0
        ),
        lambda: smetrics.precision_score(
            expected_class, predicted_class, average="macro", zero_division=0
        ),
        args.repetitions,
    )
    report(
        "recall_score macro",
        lambda: rmetrics.recall_score(
            expected_class, predicted_class, average="macro", zero_division=0
        ),
        lambda: smetrics.recall_score(
            expected_class, predicted_class, average="macro", zero_division=0
        ),
        args.repetitions,
    )
    report(
        "f1_score macro",
        lambda: rmetrics.f1_score(
            expected_class, predicted_class, average="macro", zero_division=0
        ),
        lambda: smetrics.f1_score(
            expected_class, predicted_class, average="macro", zero_division=0
        ),
        args.repetitions,
    )
    report(
        "mean_squared_error",
        lambda: rmetrics.mean_squared_error(expected_regression, predicted_regression),
        lambda: smetrics.mean_squared_error(expected_regression, predicted_regression),
        args.repetitions,
    )
    report(
        "mean_absolute_error",
        lambda: rmetrics.mean_absolute_error(expected_regression, predicted_regression),
        lambda: smetrics.mean_absolute_error(expected_regression, predicted_regression),
        args.repetitions,
    )
    report(
        "r2_score",
        lambda: rmetrics.r2_score(expected_regression, predicted_regression),
        lambda: smetrics.r2_score(expected_regression, predicted_regression),
        args.repetitions,
    )


if __name__ == "__main__":
    main()
