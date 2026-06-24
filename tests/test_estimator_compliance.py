import warnings

import pytest
from rsklearn.impute import SimpleImputer
from rsklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from rsklearn.neighbors import KNeighborsClassifier
from rsklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

sklearn_checks = pytest.importorskip("sklearn.utils.estimator_checks")


@pytest.mark.parametrize(
    "estimator",
    [
        StandardScaler(),
        MaxAbsScaler(),
        MinMaxScaler(),
        Normalizer(),
        OneHotEncoder(),
        RobustScaler(),
        OrdinalEncoder(),
        SimpleImputer(),
        LinearRegression(),
        Ridge(),
        Lasso(),
        ElasticNet(),
        LogisticRegression(max_iter=500),
        KNeighborsClassifier(n_neighbors=1),
    ],
)
def test_scaler_passes_scikit_learn_estimator_checks(estimator):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sklearn_checks.check_estimator(estimator)
