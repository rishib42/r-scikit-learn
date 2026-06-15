"""Minimal installed-distribution smoke test used by release workflows."""

from __future__ import annotations

import numpy as np
import rsklearn
from rsklearn.linear_model import LinearRegression, LogisticRegression
from rsklearn.preprocessing import OneHotEncoder, StandardScaler


def main() -> None:
    X = np.asarray([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]])
    regression = LinearRegression().fit(X, [3.0, 3.0, 7.0, 7.0])
    np.testing.assert_allclose(regression.predict(X), [3.0, 3.0, 7.0, 7.0])

    classification = LogisticRegression(max_iter=500).fit(X, [0, 0, 1, 1])
    np.testing.assert_array_equal(classification.predict(X), [0, 0, 1, 1])

    scaled = StandardScaler().fit_transform(X)
    np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-12)

    encoded = OneHotEncoder().fit_transform([["a"], ["b"], ["a"]])
    assert encoded.shape == (3, 2)
    assert rsklearn.__version__


if __name__ == "__main__":
    main()
