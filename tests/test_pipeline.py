import numpy as np
import pytest
from rsklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from rsklearn.impute import SimpleImputer
from rsklearn.pipeline import Pipeline, make_pipeline
from rsklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


class AddTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, amount=0):
        self.amount = amount

    def fit(self, X, y=None, *, marker=None):
        del y
        self.n_features_in_ = np.asarray(X).shape[1]
        self.marker_ = marker
        return self

    def transform(self, X):
        return np.asarray(X) + self.amount

    def inverse_transform(self, X):
        return np.asarray(X) - self.amount

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray([f"x{index}" for index in range(self.n_features_in_)])
        return np.asarray(input_features)


class ThresholdClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, threshold=0):
        self.threshold = threshold

    def fit(self, X, y=None, *, marker=None):
        del y
        self.marker_ = marker
        self.classes_ = np.asarray([0, 1])
        return self

    def predict(self, X):
        return (np.asarray(X)[:, 0] > self.threshold).astype(int)

    def predict_proba(self, X):
        positive = self.predict(X)
        return np.column_stack((1 - positive, positive))


class FitTransformOnly(BaseEstimator):
    def fit(self, X, y=None):
        del X, y
        return self

    def fit_transform(self, X, y=None):
        del y
        return np.asarray(X) + 1


def test_transformer_pipeline_fit_transform_inverse_and_indexing():
    pipeline = Pipeline([("add", AddTransformer(2)), ("scale", MinMaxScaler())])
    X = np.asarray([[1.0], [3.0]])
    transformed = pipeline.fit_transform(X)
    np.testing.assert_array_equal(transformed, [[0.0], [1.0]])
    np.testing.assert_allclose(pipeline.inverse_transform(transformed), X)
    assert pipeline[0] is pipeline.named_steps["add"]
    assert pipeline["scale"] is pipeline.named_steps["scale"]
    assert pipeline.named_steps.scale is pipeline.named_steps["scale"]
    assert len(pipeline[:1]) == 1
    assert pipeline.n_features_in_ == 1


def test_predictor_pipeline_routes_fit_params_and_delegates_methods():
    pipeline = Pipeline(
        [("add", AddTransformer(2)), ("classifier", ThresholdClassifier(2))]
    )
    pipeline.fit([[0.0], [1.0]], [0, 1], add__marker="first", classifier__marker="last")
    assert pipeline.named_steps["add"].marker_ == "first"
    assert pipeline.named_steps["classifier"].marker_ == "last"
    np.testing.assert_array_equal(pipeline.predict([[0.0], [1.0]]), [0, 1])
    np.testing.assert_array_equal(
        pipeline.predict_proba([[0.0], [1.0]]), [[1, 0], [0, 1]]
    )
    assert pipeline.score([[0.0], [1.0]], [0, 1]) == 1.0
    np.testing.assert_array_equal(pipeline.classes_, [0, 1])


def test_nested_parameters_replacement_clone_and_make_pipeline_names():
    pipeline = make_pipeline(StandardScaler(), StandardScaler(), MinMaxScaler())
    assert [name for name, _ in pipeline.steps] == [
        "standardscaler-1",
        "standardscaler-2",
        "minmaxscaler",
    ]
    assert pipeline.get_params()["standardscaler-1__with_mean"] is True
    replacement = StandardScaler(with_std=False)
    pipeline.set_params(
        **{"standardscaler-1__with_mean": False, "standardscaler-2": replacement}
    )
    assert pipeline.named_steps["standardscaler-1"].with_mean is False
    assert pipeline.named_steps["standardscaler-2"] is replacement
    copied = clone(pipeline)
    assert copied is not pipeline
    assert copied.named_steps["standardscaler-2"] is not replacement


def test_passthrough_feature_names_and_real_estimator_composition():
    pipeline = Pipeline(
        [
            ("skip", "passthrough"),
            ("impute", SimpleImputer()),
            ("scale", StandardScaler()),
        ]
    )
    X = np.asarray([[1.0, np.nan], [3.0, 4.0]])
    transformed = pipeline.fit_transform(X)
    assert transformed.shape == X.shape
    np.testing.assert_array_equal(pipeline.get_feature_names_out(), ["x0", "x1"])
    assert pipeline.n_features_in_ == 2
    encoded = make_pipeline(OneHotEncoder(sparse_output=False)).fit_transform(
        [["a"], ["b"]]
    )
    np.testing.assert_array_equal(encoded, [[1, 0], [0, 1]])


@pytest.mark.parametrize(
    "steps,exception",
    [
        ([], ValueError),
        ([("same", AddTransformer()), ("same", AddTransformer())], ValueError),
        ([("bad__name", AddTransformer())], ValueError),
        ([("steps", AddTransformer())], ValueError),
        ([("bad", object()), ("last", AddTransformer())], TypeError),
        ([("bad", ThresholdClassifier()), ("last", AddTransformer())], TypeError),
    ],
)
def test_invalid_step_definitions_are_rejected(steps, exception):
    with pytest.raises(exception):
        Pipeline(steps).fit([[1.0]])


def test_unsupported_configuration_and_method_errors_are_explicit():
    with pytest.raises(NotImplementedError, match="caching"):
        Pipeline([("add", AddTransformer())], memory="cache").fit([[1.0]])
    with pytest.raises(NotImplementedError, match="metadata routing"):
        Pipeline([("add", AddTransformer())], transform_input=["sample_weight"]).fit(
            [[1.0]]
        )
    with pytest.raises(ValueError, match="step__parameter"):
        Pipeline([("add", AddTransformer())]).fit([[1.0]], marker="bad")
    pipeline = Pipeline([("classifier", ThresholdClassifier())]).fit([[1.0]], [1])
    assert not hasattr(pipeline, "transform")
    with pytest.raises(ValueError, match="not fitted"):
        Pipeline([("add", AddTransformer())]).transform([[1.0]])


def test_verbose_reports_fitted_steps(capsys):
    Pipeline(
        [("add", AddTransformer()), ("scale", StandardScaler())], verbose=True
    ).fit([[1.0], [2.0]])
    output = capsys.readouterr().out
    assert "[Pipeline] add completed" in output
    assert "[Pipeline] scale completed" in output


def test_fit_transform_only_final_estimator_is_supported():
    pipeline = Pipeline([("add", AddTransformer(2)), ("final", FitTransformOnly())])
    assert hasattr(pipeline, "fit_transform")
    assert not hasattr(pipeline, "transform")
    np.testing.assert_array_equal(pipeline.fit_transform([[1.0]]), [[4.0]])
