"""Scikit-learn-style preprocessing powered by Rust."""

from .base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
    clone,
)
from .preprocessing import LabelEncoder, MinMaxScaler, Normalizer, StandardScaler

__all__ = [
    "BaseEstimator",
    "ClassifierMixin",
    "LabelEncoder",
    "MinMaxScaler",
    "Normalizer",
    "RegressorMixin",
    "StandardScaler",
    "TransformerMixin",
    "clone",
]
__version__ = "0.1.0"
