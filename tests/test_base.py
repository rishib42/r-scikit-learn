import numpy as np
import pytest
from rsklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
    clone,
)
from rsklearn.preprocessing import StandardScaler


class Wrapper(BaseEstimator):
    def __init__(self, estimator=None, *, enabled=True):
        self.estimator = estimator
        self.enabled = enabled


class OffsetTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, *, offset=0):
        self.offset = offset

    def fit(self, X, y=None):
        del X, y
        self.fitted_ = True
        return self

    def transform(self, X):
        return np.asarray(X) + self.offset


class FixedClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, X):
        del X
        return self.prediction


class FixedRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, X):
        del X
        return self.prediction


def test_get_and_set_params_support_nested_estimators():
    wrapper = Wrapper(StandardScaler(with_mean=False))
    assert wrapper.get_params(deep=False) == {
        "enabled": True,
        "estimator": wrapper.estimator,
    }
    assert wrapper.get_params()["estimator__with_mean"] is False
    assert wrapper.set_params(estimator__with_mean=True, enabled=False) is wrapper
    assert wrapper.estimator.with_mean is True
    assert wrapper.enabled is False
    with pytest.raises(ValueError, match="Invalid parameter"):
        wrapper.set_params(missing=True)


def test_clone_preserves_params_without_fitted_state_or_shared_nested_estimator():
    original = Wrapper(StandardScaler().fit([[1.0], [2.0]]))
    copied = clone(original)
    assert copied is not original
    assert copied.estimator is not original.estimator
    assert copied.get_params(deep=False)["enabled"] is True
    assert not hasattr(copied.estimator, "mean_")
    with pytest.raises(TypeError, match="Cannot clone"):
        clone(object())
    assert clone(object(), safe=False) is not None
    assert clone([1, "value"]) == [1, "value"]


def test_clone_protocol_is_respected():
    class CustomClone:
        def __sklearn_clone__(self):
            return "custom clone"

    assert clone(CustomClone()) == "custom clone"


def test_transformer_mixin_and_scores():
    transformed = OffsetTransformer(offset=2).fit_transform([[1], [2]])
    np.testing.assert_array_equal(transformed, [[3], [4]])
    assert FixedClassifier([1, 0]).score([[0], [1]], [1, 1]) == 0.5
    assert FixedClassifier([[1, 0], [0, 1]]).score([[0], [1]], [[1, 0], [1, 1]]) == 0.5
    assert FixedRegressor([1.0, 2.0]).score([[0], [1]], [1.0, 2.0]) == 1.0


def test_estimators_must_not_use_varargs():
    class Invalid(BaseEstimator):
        def __init__(self, *args):
            self.args = args

    with pytest.raises(RuntimeError, match=r"may not use \*args"):
        Invalid().get_params()


def test_optional_sklearn_tags_describe_transformer_capabilities():
    pytest.importorskip("sklearn")
    tags = StandardScaler().__sklearn_tags__()
    assert tags.input_tags.allow_nan
    assert tags.transformer_tags.preserves_dtype == ["float64", "float32"]
