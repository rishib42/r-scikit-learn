import numpy as np
import pytest
from rsklearn.pipeline import Pipeline, make_pipeline
from rsklearn.preprocessing import MinMaxScaler, StandardScaler

sklearn_pipeline = pytest.importorskip("sklearn.pipeline")
sklearn_preprocessing = pytest.importorskip("sklearn.preprocessing")
ScikitPipeline = sklearn_pipeline.Pipeline
ScikitMinMaxScaler = sklearn_preprocessing.MinMaxScaler
ScikitStandardScaler = sklearn_preprocessing.StandardScaler


def test_transformer_pipeline_matches_scikit_learn():
    X = np.asarray([[1.0, 10.0], [2.0, 30.0], [4.0, 20.0]])
    ours = Pipeline(
        [
            ("standard", StandardScaler()),
            ("minmax", MinMaxScaler(feature_range=(-1, 1))),
        ]
    )
    theirs = ScikitPipeline(
        [
            ("standard", ScikitStandardScaler()),
            ("minmax", ScikitMinMaxScaler((-1, 1))),
        ]
    )
    np.testing.assert_allclose(ours.fit_transform(X), theirs.fit_transform(X))
    np.testing.assert_allclose(ours.transform(X), theirs.transform(X))
    np.testing.assert_allclose(
        ours.inverse_transform(ours.transform(X)),
        theirs.inverse_transform(theirs.transform(X)),
    )
    np.testing.assert_array_equal(
        ours.get_feature_names_out(), theirs.get_feature_names_out()
    )


def test_pipeline_parameters_and_automatic_names_match_scikit_learn():
    ours = make_pipeline(StandardScaler(), StandardScaler(), MinMaxScaler())
    theirs = sklearn_pipeline.make_pipeline(
        ScikitStandardScaler(), ScikitStandardScaler(), ScikitMinMaxScaler()
    )
    assert [name for name, _ in ours.steps] == [name for name, _ in theirs.steps]
    assert set(ours.get_params(deep=False)) == set(theirs.get_params(deep=False))
    assert "standardscaler-1__with_mean" in ours.get_params()
    assert "minmaxscaler__feature_range" in ours.get_params()
    ours.set_params(**{"standardscaler-1__with_mean": False})
    theirs.set_params(**{"standardscaler-1__with_mean": False})
    assert (
        ours.named_steps["standardscaler-1"].with_mean
        == theirs.named_steps["standardscaler-1"].with_mean
    )
    assert ours.named_steps.minmaxscaler is ours.named_steps["minmaxscaler"]


def test_pipeline_hides_methods_not_supported_by_final_estimator():
    transformer = Pipeline([("scale", StandardScaler())])
    predictor = Pipeline([("predictor", _Predictor())])
    assert hasattr(transformer, "transform")
    assert not hasattr(transformer, "predict")
    assert hasattr(predictor, "predict")
    assert not hasattr(predictor, "transform")


def test_pipeline_delegates_estimator_type_tags_to_final_estimator():
    pytest.importorskip("sklearn")
    pipeline = Pipeline([("predictor", _TaggedClassifier())])
    assert pipeline.__sklearn_tags__().estimator_type == "classifier"


class _Predictor:
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class _TaggedClassifier(_Predictor):
    def __sklearn_tags__(self):
        from sklearn.utils import ClassifierTags, InputTags, Tags, TargetTags

        return Tags(
            estimator_type="classifier",
            target_tags=TargetTags(required=True),
            classifier_tags=ClassifierTags(),
            input_tags=InputTags(),
        )
