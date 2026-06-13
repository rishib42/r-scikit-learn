import numpy as np
import pytest
from rsklearn.preprocessing import LabelEncoder


def test_numeric_labels_sorted_encoded_and_roundtrip():
    labels = np.array([10, -2, 10, 3])
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    np.testing.assert_array_equal(encoder.classes_, [-2, 3, 10])
    np.testing.assert_array_equal(encoded, [2, 0, 2, 1])
    np.testing.assert_array_equal(encoder.inverse_transform(encoded), labels)


def test_unicode_labels_are_not_corrupted():
    labels = ["東京", "café", "東京", "å"]
    encoder = LabelEncoder().fit(labels)
    np.testing.assert_array_equal(
        encoder.inverse_transform(encoder.transform(labels)), labels
    )


def test_numpy_unicode_array_uses_sorted_classes_and_handles_non_contiguous_input():
    labels = np.asarray(["東京", "café", "å", "café", "東京"])[::2]
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    np.testing.assert_array_equal(encoder.classes_, ["å", "東京"])
    np.testing.assert_array_equal(encoded, [1, 0, 1])
    np.testing.assert_array_equal(encoder.inverse_transform(encoded), labels)


def test_unknown_and_invalid_codes_are_rejected():
    encoder = LabelEncoder().fit(["a", "b"])
    with pytest.raises(ValueError, match="unknown label"):
        encoder.transform(["c"])
    with pytest.raises(ValueError, match="outside the valid range"):
        encoder.inverse_transform([-1])
    with pytest.raises(ValueError, match="outside the valid range"):
        encoder.inverse_transform([2])
    with pytest.raises(TypeError, match="integer encoded"):
        encoder.inverse_transform([0.5])


@pytest.mark.parametrize("labels", [[], np.array([])])
def test_empty_labels_are_supported(labels):
    encoder = LabelEncoder()
    np.testing.assert_array_equal(encoder.fit_transform(labels), [])
    np.testing.assert_array_equal(encoder.classes_, [])


def test_shape_type_and_fitted_validation():
    with pytest.raises(ValueError, match="1-dimensional"):
        LabelEncoder().fit([["a"], ["b"]])
    with pytest.raises(ValueError, match="not fitted"):
        LabelEncoder().transform(["a"])


def test_boolean_mixed_and_non_finite_labels():
    for labels in ([True, False, True], [1, "a"], [1.0, np.nan, np.inf]):
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(labels)
        np.testing.assert_array_equal(encoder.inverse_transform(encoded), labels)


@pytest.mark.parametrize(
    "labels",
    [
        np.array([2**53, 2**53 + 1], dtype=np.int64),
        np.array([2**63, 2**63 + 1], dtype=np.uint64),
        np.array([1, 2, 1], dtype=np.int32),
    ],
)
def test_integer_labels_preserve_values_and_dtype(labels):
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    assert encoder.classes_.dtype == labels.dtype
    np.testing.assert_array_equal(encoder.inverse_transform(encoded), labels)


def test_python_integer_outside_uint64_is_rejected():
    with pytest.raises(ValueError, match="must fit"):
        LabelEncoder().fit([2**100])


def test_params_and_repr():
    encoder = LabelEncoder()
    assert encoder.get_params() == {}
    assert repr(encoder) == "LabelEncoder()"
    assert encoder.set_params() is encoder
    with pytest.raises(ValueError, match="Invalid parameter"):
        encoder.set_params(foo=True)
