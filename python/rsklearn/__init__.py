"""Scikit-learn-style preprocessing powered by Rust."""

from .base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
    clone,
)
from .compose import ColumnTransformer, make_column_transformer
from .impute import SimpleImputer
from .pipeline import Pipeline, make_pipeline
from .preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

__all__ = [
    "BaseEstimator",
    "ClassifierMixin",
    "ColumnTransformer",
    "LabelEncoder",
    "MinMaxScaler",
    "Normalizer",
    "OneHotEncoder",
    "OrdinalEncoder",
    "Pipeline",
    "RegressorMixin",
    "RobustScaler",
    "SimpleImputer",
    "StandardScaler",
    "TransformerMixin",
    "clone",
    "make_column_transformer",
    "make_pipeline",
]
__version__ = "0.1.0"
