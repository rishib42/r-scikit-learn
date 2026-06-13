import numpy as np
import pytest
from rsklearn.preprocessing import MinMaxScaler, StandardScaler


@pytest.mark.parametrize("estimator", [StandardScaler(), MinMaxScaler()])
@pytest.mark.parametrize("value", [[1, 2], [[[1.0]]]])
def test_numeric_input_must_be_2d(estimator, value):
    with pytest.raises(ValueError, match="2-dimensional"):
        estimator.fit(value)


@pytest.mark.parametrize("estimator", [StandardScaler(), MinMaxScaler()])
@pytest.mark.parametrize("value", [np.empty((0, 2)), np.empty((2, 0))])
def test_numeric_input_must_not_be_empty(estimator, value):
    with pytest.raises(ValueError, match="at least one sample and one feature"):
        estimator.fit(value)


@pytest.mark.parametrize("estimator", [StandardScaler(), MinMaxScaler()])
def test_numeric_input_accepts_nan(estimator):
    transformed = estimator.fit_transform([[1.0, np.nan], [2.0, 3.0]])
    assert np.isnan(transformed[0, 1])


@pytest.mark.parametrize("estimator", [StandardScaler(), MinMaxScaler()])
def test_numeric_input_rejects_infinity(estimator):
    with pytest.raises(ValueError, match="infinity"):
        estimator.fit([[1.0, np.inf]])


def test_array_like_is_accepted_and_converted_to_float64():
    output = StandardScaler().fit_transform([[1, 2], [3, 4]])
    assert output.dtype == np.float64


@pytest.mark.parametrize("estimator", [StandardScaler(), MinMaxScaler()])
def test_float32_transform_output_is_preserved(estimator):
    X = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    assert estimator.fit_transform(X).dtype == np.float32
