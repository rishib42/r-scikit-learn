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
@pytest.mark.parametrize("value", [[[1.0, np.nan]], [[1.0, np.inf]]])
def test_numeric_input_must_be_finite(estimator, value):
    with pytest.raises(ValueError, match="NaN or infinity"):
        estimator.fit(value)


def test_array_like_is_accepted_and_converted_to_float64():
    output = StandardScaler().fit_transform([[1, 2], [3, 4]])
    assert output.dtype == np.float64
