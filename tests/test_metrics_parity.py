import numpy as np
import pytest
import rsklearn.metrics as ours

theirs = pytest.importorskip("sklearn.metrics")


@pytest.mark.parametrize(
    "expected,predicted,weights",
    [
        ([0, 1, 2, 2], [0, 2, 2, 1], None),
        (["a", "b", "b"], ["a", "a", "b"], [1, 2, 3]),
    ],
)
def test_classification_metric_parity(expected, predicted, weights):
    assert ours.accuracy_score(
        expected, predicted, sample_weight=weights
    ) == pytest.approx(
        theirs.accuracy_score(expected, predicted, sample_weight=weights)
    )
    np.testing.assert_allclose(
        ours.confusion_matrix(expected, predicted, sample_weight=weights),
        theirs.confusion_matrix(expected, predicted, sample_weight=weights),
    )
    for metric in ("precision_score", "recall_score", "f1_score"):
        for average in (None, "micro", "macro", "weighted"):
            np.testing.assert_allclose(
                getattr(ours, metric)(
                    expected,
                    predicted,
                    average=average,
                    sample_weight=weights,
                    zero_division=0,
                ),
                getattr(theirs, metric)(
                    expected,
                    predicted,
                    average=average,
                    sample_weight=weights,
                    zero_division=0,
                ),
            )


@pytest.mark.parametrize("sample_weight", [None, [1, 2, 3]])
@pytest.mark.parametrize("multioutput", ["raw_values", "uniform_average", [0.25, 0.75]])
def test_regression_metric_parity(sample_weight, multioutput):
    expected = [[1.0, 2.0], [3.0, 5.0], [6.0, 8.0]]
    predicted = [[1.0, 3.0], [2.0, 5.0], [8.0, 7.0]]
    for metric in ("mean_squared_error", "mean_absolute_error", "r2_score"):
        np.testing.assert_allclose(
            getattr(ours, metric)(
                expected,
                predicted,
                sample_weight=sample_weight,
                multioutput=multioutput,
            ),
            getattr(theirs, metric)(
                expected,
                predicted,
                sample_weight=sample_weight,
                multioutput=multioutput,
            ),
        )


def test_r2_variance_weighted_and_force_finite_parity():
    expected = [[1, 5], [2, 5], [3, 5]]
    predicted = [[1, 4], [1, 4], [4, 4]]
    for force_finite in (True, False):
        np.testing.assert_allclose(
            ours.r2_score(
                expected,
                predicted,
                multioutput="variance_weighted",
                force_finite=force_finite,
            ),
            theirs.r2_score(
                expected,
                predicted,
                multioutput="variance_weighted",
                force_finite=force_finite,
            ),
            equal_nan=True,
        )
