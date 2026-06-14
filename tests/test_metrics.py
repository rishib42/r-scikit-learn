import numpy as np
import pytest
from rsklearn.metrics import (
    UndefinedMetricWarning,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def test_accuracy_supports_weights_counts_strings_and_multilabel():
    assert accuracy_score([0, 1, 1], [0, 0, 1]) == pytest.approx(2 / 3)
    assert accuracy_score([0, 1, 1], [0, 0, 1], normalize=False) == 2
    assert accuracy_score(
        ["a", "b"], ["a", "a"], sample_weight=[1, 3]
    ) == pytest.approx(0.25)
    assert accuracy_score([[1, 0], [1, 1]], [[1, 0], [0, 1]]) == 0.5
    assert np.isnan(accuracy_score([], []))


def test_confusion_matrix_labels_weights_and_normalization():
    expected = [0, 1, 1, 2]
    predicted = [0, 0, 1, 1]
    np.testing.assert_array_equal(
        confusion_matrix(expected, predicted),
        [[1, 0, 0], [1, 1, 0], [0, 1, 0]],
    )
    weighted = confusion_matrix(
        expected, predicted, labels=[2, 1, 0], sample_weight=[1, 2, 3, 4]
    )
    np.testing.assert_array_equal(weighted, [[0, 4, 0], [0, 3, 2], [0, 0, 1]])
    np.testing.assert_allclose(
        confusion_matrix(expected, predicted, normalize="true"),
        [[1, 0, 0], [0.5, 0.5, 0], [0, 1, 0]],
    )
    assert confusion_matrix([], []).shape == (0, 0)
    with pytest.raises(ValueError, match="must be in y_true"):
        confusion_matrix([0, 1], [0, 1], labels=[2])


@pytest.mark.parametrize("metric", [precision_score, recall_score, f1_score])
@pytest.mark.parametrize("average", [None, "micro", "macro", "weighted"])
def test_precision_recall_f1_multiclass_averages(metric, average):
    result = metric(
        [0, 1, 2, 2],
        [0, 2, 2, 1],
        average=average,
        zero_division=0,
    )
    expected = np.asarray([1.0, 0.0, 0.5]) if average is None else 0.5
    np.testing.assert_allclose(result, expected)


def test_binary_metrics_and_zero_division_behavior():
    assert precision_score([0, 1, 1], [0, 0, 1]) == 1
    assert recall_score([0, 1, 1], [0, 0, 1]) == 0.5
    assert f1_score([0, 1, 1], [0, 0, 1]) == pytest.approx(2 / 3)
    with pytest.warns(UndefinedMetricWarning):
        assert precision_score([0, 0], [0, 0]) == 0
    assert precision_score([0, 0], [0, 0], zero_division=1) == 1
    assert np.isnan(precision_score([0, 0], [0, 0], zero_division=np.nan))


def test_regression_metrics_multioutput_weights_and_constant_targets():
    expected = np.asarray([[1.0, 2.0], [3.0, 5.0], [6.0, 8.0]])
    predicted = np.asarray([[1.0, 3.0], [2.0, 5.0], [8.0, 7.0]])
    np.testing.assert_allclose(
        mean_squared_error(expected, predicted, multioutput="raw_values"),
        [5 / 3, 2 / 3],
    )
    np.testing.assert_allclose(
        mean_absolute_error(expected, predicted, multioutput="raw_values"),
        [1, 2 / 3],
    )
    assert mean_squared_error(
        expected, predicted, sample_weight=[1, 2, 3]
    ) == pytest.approx(1.5)
    np.testing.assert_allclose(
        r2_score(expected, predicted, multioutput="raw_values"),
        [0.6052631578947368, 0.8888888888888888],
    )
    assert r2_score([1, 1], [1, 1]) == 1
    assert r2_score([1, 1], [0, 0]) == 0
    assert np.isneginf(r2_score([1, 1], [0, 0], force_finite=False))


def test_metrics_reject_invalid_inputs():
    with pytest.raises(ValueError, match="different shapes"):
        accuracy_score([1], [1, 2])
    with pytest.raises(ValueError, match="continuous"):
        confusion_matrix([0.1, 0.2], [0.1, 0.2])
    with pytest.raises(ValueError, match="sample_weight"):
        accuracy_score([0, 1], [0, 1], sample_weight=[1])
    with pytest.raises(ZeroDivisionError):
        accuracy_score([0, 1], [0, 1], sample_weight=[0, 0])
    with pytest.raises(ValueError, match="NaN or infinity"):
        mean_squared_error([1, np.nan], [1, 2])
    with pytest.raises(ZeroDivisionError):
        mean_absolute_error([1, 2], [1, 2], sample_weight=[0, 0])
    with pytest.warns(UndefinedMetricWarning):
        assert np.isnan(r2_score([1], [1]))
