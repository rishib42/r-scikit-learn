"""Scikit-learn-style preprocessing powered by Rust."""

from .base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
    clone,
)
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
    "make_pipeline",
]
__version__ = "0.1.0"
