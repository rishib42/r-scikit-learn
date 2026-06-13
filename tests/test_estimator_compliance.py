import warnings

import pytest
from rsklearn.preprocessing import (
    MinMaxScaler,
    Normalizer,
    RobustScaler,
    StandardScaler,
)

sklearn_checks = pytest.importorskip("sklearn.utils.estimator_checks")


@pytest.mark.parametrize(
    "estimator", [StandardScaler(), MinMaxScaler(), Normalizer(), RobustScaler()]
)
def test_scaler_passes_scikit_learn_estimator_checks(estimator):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sklearn_checks.check_estimator(estimator)
