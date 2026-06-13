import numpy as np
import pytest
from rsklearn.base import BaseEstimator
from rsklearn.preprocessing import StandardScaler
from rsklearn.utils.validation import (
    check_array,
    check_is_fitted,
    check_X_y,
    validate_data,
)


class ArrayWithColumns:
    def __init__(self, values, columns):
        self.values = np.asarray(values)
        self.columns = columns

    def __array__(self, dtype=None):
        return np.asarray(self.values, dtype=dtype)


class ValidatingEstimator(BaseEstimator):
    def fit(self, X, y=None):
        validate_data(self, X, y, dtype=np.float64)
        self.fitted_ = True
        return self


class HookEstimator:
    def fit(self, X, y=None):
        del X, y
        return self

    def __sklearn_is_fitted__(self):
        return True


def test_check_array_conversion_copy_shape_and_finite_policies():
    original = np.asarray([[1, 2], [3, 4]], dtype=np.int32)
    checked = check_array(original, dtype=np.float64, copy=True)
    assert checked.dtype == np.float64
    assert checked is not original
    with pytest.raises(ValueError, match="2-dimensional"):
        check_array([1, 2])
    with pytest.raises(ValueError, match="NaN or infinity"):
        check_array([[np.nan]])
    assert np.isnan(check_array([[np.nan]], ensure_all_finite="allow-nan")[0, 0])
    with pytest.raises(ValueError, match="infinity"):
        check_array([[np.inf]], ensure_all_finite="allow-nan")
    with pytest.raises(ValueError, match="negative"):
        check_array([[-1]], ensure_non_negative=True)


def test_check_X_y_validates_length_and_target_shape():
    X, y = check_X_y([[1], [2]], ["a", "b"])
    assert X.shape == (2, 1)
    assert y.shape == (2,)
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        check_X_y([[1], [2]], ["a"])


def test_check_is_fitted_supports_default_and_explicit_attributes():
    estimator = ValidatingEstimator()
    with pytest.raises(ValueError, match="not fitted"):
        check_is_fitted(estimator)
    estimator.fit([[1.0]])
    check_is_fitted(estimator)
    check_is_fitted(estimator, ("fitted_", "n_features_in_"))
    with pytest.raises(TypeError, match="estimator instance"):
        check_is_fitted(object())
    check_is_fitted(HookEstimator())


def test_validate_data_records_and_checks_feature_metadata():
    estimator = ValidatingEstimator().fit(
        ArrayWithColumns([[1, 2], [3, 4]], ["first", "second"])
    )
    assert estimator.n_features_in_ == 2
    np.testing.assert_array_equal(estimator.feature_names_in_, ["first", "second"])
    validate_data(
        estimator,
        ArrayWithColumns([[5, 6]], ["first", "second"]),
        reset=False,
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="expecting 2 features"):
        validate_data(estimator, [[1]], reset=False)
    with pytest.raises(ValueError, match="feature names must match"):
        validate_data(
            estimator,
            ArrayWithColumns([[5, 6]], ["second", "first"]),
            reset=False,
        )
    with pytest.raises(ValueError, match="does not have feature names"):
        validate_data(estimator, [[5, 6]], reset=False)


def test_scalers_use_public_fitted_and_feature_validation():
    scaler = StandardScaler()
    with pytest.raises(ValueError, match="not fitted"):
        scaler.transform([[1.0]])
    scaler.fit([[1.0, 2.0]])
    with pytest.raises(ValueError, match="expecting 2 features"):
        scaler.transform([[1.0]])
