#![forbid(unsafe_code)]

mod categorical;
mod error;
mod label_encoder;
mod linear_model;
mod metrics;
mod minmax_scaler;
mod normalizer;
mod robust_scaler;
mod simple_imputer;
mod sparse;
mod standard_scaler;

use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyReadwriteArray1, PyUntypedArrayMethods,
};
use pyo3::prelude::*;

type FloatArray1<'py> = Bound<'py, PyArray1<f64>>;
type IntArray1<'py> = Bound<'py, PyArray1<i64>>;
type UIntArray1<'py> = Bound<'py, PyArray1<u64>>;
type IntLabelOutput<'py> = (IntArray1<'py>, IntArray1<'py>);
type UIntLabelOutput<'py> = (UIntArray1<'py>, IntArray1<'py>);
type StandardFitOutput<'py> = (
    FloatArray1<'py>,
    FloatArray1<'py>,
    FloatArray1<'py>,
    Bound<'py, PyArray1<i64>>,
);
type ThreeFloatArrays<'py> = (FloatArray1<'py>, FloatArray1<'py>, FloatArray1<'py>);
type FloatCategoryOutput<'py> = (FloatArray1<'py>, IntArray1<'py>);
type CategoryMatrixOutput<'py, T> = (Vec<Vec<T>>, Bound<'py, PyArray2<i64>>);
type RegressionReductionOutput<'py> = (
    FloatArray1<'py>,
    FloatArray1<'py>,
    FloatArray1<'py>,
    FloatArray1<'py>,
    f64,
);
type LinearFitOutput<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>);
type LogisticFitOutput<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    bool,
);
type CoordinateFitOutput<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
    Bound<'py, PyArray1<f64>>,
    bool,
);

#[pyfunction]
fn build_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

fn array2_output<'py>(
    py: Python<'py>,
    values: Vec<f64>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let array = Array2::from_shape_vec((rows, cols), values)
        .map_err(|_| error::CoreError::ShapeMismatch)?;
    Ok(array.into_pyarray(py))
}

fn array2_output_f32<'py>(
    py: Python<'py>,
    values: Vec<f32>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let array = Array2::from_shape_vec((rows, cols), values)
        .map_err(|_| error::CoreError::ShapeMismatch)?;
    Ok(array.into_pyarray(py))
}

fn array2_output_i64<'py>(
    py: Python<'py>,
    values: Vec<i64>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let array = Array2::from_shape_vec((rows, cols), values)
        .map_err(|_| error::CoreError::ShapeMismatch)?;
    Ok(array.into_pyarray(py))
}

#[pyfunction]
fn metric_accuracy(
    py: Python<'_>,
    expected: PyReadonlyArray1<'_, i64>,
    predicted: PyReadonlyArray1<'_, i64>,
    weights: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, f64)> {
    let expected = expected.as_slice()?;
    let predicted = predicted.as_slice()?;
    let weights = weights.as_slice()?;
    py.detach(|| metrics::accuracy(expected, predicted, weights))
        .map_err(Into::into)
}

#[pyfunction]
fn metric_confusion_matrix<'py>(
    py: Python<'py>,
    expected: PyReadonlyArray1<'py, i64>,
    predicted: PyReadonlyArray1<'py, i64>,
    weights: PyReadonlyArray1<'py, f64>,
    classes: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let expected = expected.as_slice()?;
    let predicted = predicted.as_slice()?;
    let weights = weights.as_slice()?;
    let output = py.detach(|| metrics::confusion_matrix(expected, predicted, weights, classes))?;
    array2_output(py, output, classes, classes)
}

#[pyfunction]
fn metric_confusion_i64<'py>(
    py: Python<'py>,
    expected: PyReadonlyArray1<'py, i64>,
    predicted: PyReadonlyArray1<'py, i64>,
    weights: PyReadonlyArray1<'py, f64>,
) -> PyResult<(IntArray1<'py>, Bound<'py, PyArray2<f64>>)> {
    let expected = expected.as_slice()?;
    let predicted = predicted.as_slice()?;
    let weights = weights.as_slice()?;
    let (classes, output) = py.detach(|| metrics::confusion_i64(expected, predicted, weights))?;
    let size = classes.len();
    Ok((
        classes.into_pyarray(py),
        array2_output(py, output, size, size)?,
    ))
}

#[pyfunction]
fn metric_regression_reductions<'py>(
    py: Python<'py>,
    expected: PyReadonlyArray2<'py, f64>,
    predicted: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
) -> PyResult<RegressionReductionOutput<'py>> {
    let shape = expected.shape();
    if predicted.shape() != shape {
        return Err(error::CoreError::ShapeMismatch.into());
    }
    let expected = expected.as_slice()?;
    let predicted = predicted.as_slice()?;
    let weights = weights.as_slice()?;
    let output = py.detach(|| {
        metrics::regression_reductions(expected, predicted, weights, shape[0], shape[1])
    })?;
    Ok((
        output.absolute_error.into_pyarray(py),
        output.squared_error.into_pyarray(py),
        output.true_sum.into_pyarray(py),
        output.true_squared_sum.into_pyarray(py),
        output.weight_sum,
    ))
}

#[pyfunction]
fn metric_regression_error<'py>(
    py: Python<'py>,
    expected: PyReadonlyArray2<'py, f64>,
    predicted: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    squared: bool,
) -> PyResult<(FloatArray1<'py>, f64)> {
    let shape = expected.shape();
    if predicted.shape() != shape {
        return Err(error::CoreError::ShapeMismatch.into());
    }
    let expected = expected.as_slice()?;
    let predicted = predicted.as_slice()?;
    let weights = weights.as_slice()?;
    let (output, weight_sum) = py.detach(|| {
        metrics::regression_error(expected, predicted, weights, shape[0], shape[1], squared)
    })?;
    Ok((output.into_pyarray(py), weight_sum))
}

#[pyfunction]
fn metric_regression_error_unweighted<'py>(
    py: Python<'py>,
    expected: PyReadonlyArray2<'py, f64>,
    predicted: PyReadonlyArray2<'py, f64>,
    squared: bool,
) -> PyResult<FloatArray1<'py>> {
    let shape = expected.shape();
    if predicted.shape() != shape {
        return Err(error::CoreError::ShapeMismatch.into());
    }
    let expected = expected.as_slice()?;
    let predicted = predicted.as_slice()?;
    let output = py.detach(|| {
        metrics::regression_error_unweighted(expected, predicted, shape[0], shape[1], squared)
    })?;
    Ok(output.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (input, targets, weights, alpha=0.0, fit_intercept=true))]
fn linear_fit<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f64>,
    targets: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    fit_intercept: bool,
) -> PyResult<LinearFitOutput<'py>> {
    let shape = input.shape();
    let target_shape = targets.shape();
    if shape[0] != target_shape[0] {
        return Err(error::CoreError::ShapeMismatch.into());
    }
    let input = input.as_slice()?;
    let targets = targets.as_slice()?;
    let weights = weights.as_slice()?;
    let fit = py.detach(|| {
        linear_model::fit_linear(
            input,
            shape[0],
            shape[1],
            targets,
            target_shape[1],
            weights,
            alpha,
            fit_intercept,
        )
    })?;
    Ok((
        array2_output(py, fit.coefficients, target_shape[1], shape[1])?,
        fit.intercepts.into_pyarray(py),
    ))
}

#[pyfunction]
fn linear_predict<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f64>,
    coefficients: PyReadonlyArray2<'py, f64>,
    intercepts: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = input.shape();
    let coefficient_shape = coefficients.shape();
    let input = input.as_slice()?;
    let coefficients = coefficients.as_slice()?;
    let intercepts = intercepts.as_slice()?;
    let output = py.detach(|| {
        linear_model::predict_linear(input, shape[0], shape[1], coefficients, intercepts)
    })?;
    array2_output(py, output, shape[0], coefficient_shape[0])
}

#[pyfunction]
fn linear_all_finite(py: Python<'_>, input: PyReadonlyArray2<'_, f64>) -> PyResult<bool> {
    let input = input.as_slice()?;
    Ok(py.detach(|| linear_model::all_finite(input)))
}

#[pyfunction]
#[pyo3(signature = (input, targets, weights, alpha, l1_ratio, fit_intercept, tolerance, max_iterations, positive))]
#[allow(clippy::too_many_arguments)]
fn linear_coordinate_fit<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f64>,
    targets: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    l1_ratio: f64,
    fit_intercept: bool,
    tolerance: f64,
    max_iterations: usize,
    positive: bool,
) -> PyResult<CoordinateFitOutput<'py>> {
    let shape = input.shape();
    let target_shape = targets.shape();
    if shape[0] != target_shape[0] {
        return Err(error::CoreError::ShapeMismatch.into());
    }
    let input = input.as_slice()?;
    let targets = targets.as_slice()?;
    let weights = weights.as_slice()?;
    let fit = py.detach(|| {
        linear_model::fit_coordinate_descent(
            input,
            shape[0],
            shape[1],
            targets,
            target_shape[1],
            weights,
            alpha,
            l1_ratio,
            fit_intercept,
            tolerance,
            max_iterations,
            positive,
        )
    })?;
    Ok((
        array2_output(py, fit.coefficients, target_shape[1], shape[1])?,
        fit.intercepts.into_pyarray(py),
        fit.iterations,
        fit.dual_gaps.into_pyarray(py),
        fit.converged,
    ))
}

#[pyfunction]
#[pyo3(signature = (input, targets, weights, alpha, l1_ratio, fit_intercept, tolerance, max_iterations, positive))]
#[allow(clippy::too_many_arguments)]
fn linear_coordinate_fit_validated<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f64>,
    targets: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    l1_ratio: f64,
    fit_intercept: bool,
    tolerance: f64,
    max_iterations: usize,
    positive: bool,
) -> PyResult<CoordinateFitOutput<'py>> {
    let shape = input.shape();
    let target_shape = targets.shape();
    if shape[0] != target_shape[0] {
        return Err(error::CoreError::ShapeMismatch.into());
    }
    let input = input.as_slice()?;
    let targets = targets.as_slice()?;
    let weights = weights.as_slice()?;
    let fit = py.detach(|| {
        linear_model::fit_coordinate_descent_validated(
            input,
            shape[0],
            shape[1],
            targets,
            target_shape[1],
            weights,
            alpha,
            l1_ratio,
            fit_intercept,
            tolerance,
            max_iterations,
            positive,
        )
    })?;
    Ok((
        array2_output(py, fit.coefficients, target_shape[1], shape[1])?,
        fit.intercepts.into_pyarray(py),
        fit.iterations,
        fit.dual_gaps.into_pyarray(py),
        fit.converged,
    ))
}

#[pyfunction]
#[pyo3(signature = (input, labels, weights, classes, inverse_c, fit_intercept, tolerance, max_iterations))]
#[allow(clippy::too_many_arguments)]
fn logistic_fit<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f64>,
    labels: PyReadonlyArray1<'py, i64>,
    weights: PyReadonlyArray1<'py, f64>,
    classes: usize,
    inverse_c: f64,
    fit_intercept: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<LogisticFitOutput<'py>> {
    let shape = input.shape();
    let input = input.as_slice()?;
    let labels = labels.as_slice()?;
    let weights = weights.as_slice()?;
    let fit = py.detach(|| {
        linear_model::fit_logistic(
            input,
            shape[0],
            shape[1],
            labels,
            classes,
            weights,
            inverse_c,
            fit_intercept,
            tolerance,
            max_iterations,
        )
    })?;
    Ok((
        array2_output(py, fit.coefficients, classes, shape[1])?,
        fit.intercepts.into_pyarray(py),
        fit.iterations,
        fit.converged,
    ))
}

#[pyfunction]
#[pyo3(signature = (input, labels, weights, l1_regularization, l2_regularization, fit_intercept, tolerance, max_iterations))]
#[allow(clippy::too_many_arguments)]
fn logistic_fit_proximal<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f64>,
    labels: PyReadonlyArray1<'py, i64>,
    weights: PyReadonlyArray1<'py, f64>,
    l1_regularization: f64,
    l2_regularization: f64,
    fit_intercept: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<LogisticFitOutput<'py>> {
    let shape = input.shape();
    let input = input.as_slice()?;
    let labels = labels.as_slice()?;
    let weights = weights.as_slice()?;
    let fit = py.detach(|| {
        linear_model::fit_binary_logistic_proximal(
            input,
            shape[0],
            shape[1],
            labels,
            weights,
            l1_regularization,
            l2_regularization,
            fit_intercept,
            tolerance,
            max_iterations,
        )
    })?;
    Ok((
        array2_output(py, fit.coefficients, 2, shape[1])?,
        fit.intercepts.into_pyarray(py),
        fit.iterations,
        fit.converged,
    ))
}

#[pyfunction]
fn logistic_softmax<'py>(
    py: Python<'py>,
    scores: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = scores.shape();
    let scores = scores.as_slice()?;
    let output = py.detach(|| linear_model::softmax(scores, shape[0], shape[1]))?;
    array2_output(py, output, shape[0], shape[1])
}

#[pyfunction]
fn sparse_validate_i32(
    py: Python<'_>,
    indices: PyReadonlyArray1<'_, i32>,
    indptr: PyReadonlyArray1<'_, i32>,
    major_dimension: usize,
    minor_dimension: usize,
    nnz: usize,
) -> PyResult<()> {
    let indices = indices.as_slice()?;
    let indptr = indptr.as_slice()?;
    py.detach(|| {
        sparse::validate_compressed(indices, indptr, major_dimension, minor_dimension, nnz)
    })?;
    Ok(())
}

#[pyfunction]
fn sparse_validate_i64(
    py: Python<'_>,
    indices: PyReadonlyArray1<'_, i64>,
    indptr: PyReadonlyArray1<'_, i64>,
    major_dimension: usize,
    minor_dimension: usize,
    nnz: usize,
) -> PyResult<()> {
    let indices = indices.as_slice()?;
    let indptr = indptr.as_slice()?;
    py.detach(|| {
        sparse::validate_compressed(indices, indptr, major_dimension, minor_dimension, nnz)
    })?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (values, indices, scale, inverse=false))]
fn sparse_scale_csr_i32_f64(
    py: Python<'_>,
    mut values: PyReadwriteArray1<'_, f64>,
    indices: PyReadonlyArray1<'_, i32>,
    scale: PyReadonlyArray1<'_, f64>,
    inverse: bool,
) -> PyResult<()> {
    let values = values.as_slice_mut()?;
    let indices = indices.as_slice()?;
    let scale = scale.as_slice()?;
    py.detach(|| sparse::scale_csr_columns_in_place(values, indices, scale, inverse))?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (values, indices, scale, inverse=false))]
fn sparse_scale_csr_i64_f64(
    py: Python<'_>,
    mut values: PyReadwriteArray1<'_, f64>,
    indices: PyReadonlyArray1<'_, i64>,
    scale: PyReadonlyArray1<'_, f64>,
    inverse: bool,
) -> PyResult<()> {
    let values = values.as_slice_mut()?;
    let indices = indices.as_slice()?;
    let scale = scale.as_slice()?;
    py.detach(|| sparse::scale_csr_columns_in_place(values, indices, scale, inverse))?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (values, indices, scale, inverse=false))]
fn sparse_scale_csr_i32_f32(
    py: Python<'_>,
    mut values: PyReadwriteArray1<'_, f32>,
    indices: PyReadonlyArray1<'_, i32>,
    scale: PyReadonlyArray1<'_, f32>,
    inverse: bool,
) -> PyResult<()> {
    let values = values.as_slice_mut()?;
    let indices = indices.as_slice()?;
    let scale = scale.as_slice()?;
    py.detach(|| sparse::scale_csr_columns_in_place(values, indices, scale, inverse))?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (values, indices, scale, inverse=false))]
fn sparse_scale_csr_i64_f32(
    py: Python<'_>,
    mut values: PyReadwriteArray1<'_, f32>,
    indices: PyReadonlyArray1<'_, i64>,
    scale: PyReadonlyArray1<'_, f32>,
    inverse: bool,
) -> PyResult<()> {
    let values = values.as_slice_mut()?;
    let indices = indices.as_slice()?;
    let scale = scale.as_slice()?;
    py.detach(|| sparse::scale_csr_columns_in_place(values, indices, scale, inverse))?;
    Ok(())
}

#[pyfunction]
fn one_hot_csr<'py>(
    py: Python<'py>,
    codes: PyReadonlyArray2<'py, i64>,
    widths: PyReadonlyArray1<'py, i64>,
    drops: PyReadonlyArray1<'py, i64>,
) -> PyResult<(IntArray1<'py>, IntArray1<'py>)> {
    let shape = codes.shape();
    let codes = codes.as_slice()?;
    let widths = widths.as_slice()?;
    let drops = drops.as_slice()?;
    let (indices, indptr) =
        py.detach(|| sparse::one_hot_csr(codes, shape[0], shape[1], widths, drops))?;
    Ok((indices.into_pyarray(py), indptr.into_pyarray(py)))
}

#[pyfunction]
fn standard_fit<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f64>,
) -> PyResult<StandardFitOutput<'py>> {
    let shape = input.shape();
    let values = input.as_slice()?;
    let stats = py.detach(|| standard_scaler::fit(values, shape[0], shape[1]))?;
    Ok((
        stats.mean.into_pyarray(py),
        stats.variance.into_pyarray(py),
        stats.scale.into_pyarray(py),
        stats
            .counts
            .into_iter()
            .map(|value| value as i64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    ))
}

#[pyfunction]
fn standard_merge<'py>(
    py: Python<'py>,
    previous_mean: PyReadonlyArray1<'py, f64>,
    previous_variance: PyReadonlyArray1<'py, f64>,
    previous_counts: PyReadonlyArray1<'py, i64>,
    input: PyReadonlyArray2<'py, f64>,
) -> PyResult<StandardFitOutput<'py>> {
    let shape = input.shape();
    let previous_mean = previous_mean.as_slice()?;
    let previous_variance = previous_variance.as_slice()?;
    let input = input.as_slice()?;
    let previous_counts: Vec<usize> = previous_counts
        .as_slice()?
        .iter()
        .map(|&value| usize::try_from(value).map_err(|_| error::CoreError::ShapeMismatch))
        .collect::<Result<_, _>>()?;
    let stats = py.detach(|| {
        let batch = standard_scaler::fit(input, shape[0], shape[1])?;
        standard_scaler::merge(previous_mean, previous_variance, &previous_counts, &batch)
    })?;
    Ok((
        stats.mean.into_pyarray(py),
        stats.variance.into_pyarray(py),
        stats.scale.into_pyarray(py),
        stats
            .counts
            .into_iter()
            .map(|value| value as i64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (input, mean, scale, with_mean, with_std, inverse=false))]
fn standard_transform<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f64>,
    mean: PyReadonlyArray1<'py, f64>,
    scale: PyReadonlyArray1<'py, f64>,
    with_mean: bool,
    with_std: bool,
    inverse: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = input.shape();
    let values = input.as_slice()?;
    let mean = mean.as_slice()?;
    let scale = scale.as_slice()?;
    let output = py.detach(|| {
        standard_scaler::transform(
            values, shape[0], shape[1], mean, scale, with_mean, with_std, inverse,
        )
    })?;
    array2_output(py, output, shape[0], shape[1])
}

#[pyfunction]
fn minmax_fit<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f64>,
) -> PyResult<ThreeFloatArrays<'py>> {
    let shape = input.shape();
    let values = input.as_slice()?;
    let stats = py.detach(|| minmax_scaler::fit(values, shape[0], shape[1]))?;
    Ok((
        stats.data_min.into_pyarray(py),
        stats.data_max.into_pyarray(py),
        stats.data_range.into_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (input, scale, min, output_low, output_high, clip, inverse=false))]
#[allow(clippy::too_many_arguments)]
fn minmax_transform<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f64>,
    scale: PyReadonlyArray1<'py, f64>,
    min: PyReadonlyArray1<'py, f64>,
    output_low: f64,
    output_high: f64,
    clip: bool,
    inverse: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = input.shape();
    let values = input.as_slice()?;
    let scale = scale.as_slice()?;
    let min = min.as_slice()?;
    let output = py.detach(|| {
        minmax_scaler::transform(
            values,
            shape[0],
            shape[1],
            scale,
            min,
            output_low,
            output_high,
            clip,
            inverse,
        )
    })?;
    array2_output(py, output, shape[0], shape[1])
}

#[pyfunction]
fn normalize_f64<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f64>,
    norm: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = input.shape();
    let values = input.as_slice()?;
    let norm = normalizer::Norm::try_from(norm)?;
    let output = py.detach(|| normalizer::transform(values, shape[0], shape[1], norm))?;
    array2_output(py, output, shape[0], shape[1])
}

#[pyfunction]
fn normalize_f32<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f32>,
    norm: &str,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let shape = input.shape();
    let values = input.as_slice()?;
    let norm = normalizer::Norm::try_from(norm)?;
    let output = py.detach(|| normalizer::transform(values, shape[0], shape[1], norm))?;
    array2_output_f32(py, output, shape[0], shape[1])
}

#[pyfunction]
#[pyo3(signature = (input, quantile_low, quantile_high, with_centering, with_scaling))]
fn robust_fit<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f64>,
    quantile_low: f64,
    quantile_high: f64,
    with_centering: bool,
    with_scaling: bool,
) -> PyResult<(FloatArray1<'py>, FloatArray1<'py>)> {
    let shape = input.shape();
    let values = input.as_slice()?;
    let stats = py.detach(|| {
        robust_scaler::fit(
            values,
            shape[0],
            shape[1],
            quantile_low,
            quantile_high,
            with_centering,
            with_scaling,
        )
    })?;
    Ok((stats.center.into_pyarray(py), stats.scale.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (input, center, scale, with_centering, with_scaling, inverse=false))]
#[allow(clippy::too_many_arguments)]
fn robust_transform_f64<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f64>,
    center: PyReadonlyArray1<'py, f64>,
    scale: PyReadonlyArray1<'py, f64>,
    with_centering: bool,
    with_scaling: bool,
    inverse: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = input.shape();
    let input = input.as_slice()?;
    let center = center.as_slice()?;
    let scale = scale.as_slice()?;
    let output = py.detach(|| {
        robust_scaler::transform_f64(
            input,
            shape[0],
            shape[1],
            center,
            scale,
            with_centering,
            with_scaling,
            inverse,
        )
    })?;
    array2_output(py, output, shape[0], shape[1])
}

#[pyfunction]
#[pyo3(signature = (input, center, scale, with_centering, with_scaling, inverse=false))]
#[allow(clippy::too_many_arguments)]
fn robust_transform_f32<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f32>,
    center: PyReadonlyArray1<'py, f64>,
    scale: PyReadonlyArray1<'py, f64>,
    with_centering: bool,
    with_scaling: bool,
    inverse: bool,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let shape = input.shape();
    let input = input.as_slice()?;
    let center = center.as_slice()?;
    let scale = scale.as_slice()?;
    let output = py.detach(|| {
        robust_scaler::transform_f32(
            input,
            shape[0],
            shape[1],
            center,
            scale,
            with_centering,
            with_scaling,
            inverse,
        )
    })?;
    array2_output_f32(py, output, shape[0], shape[1])
}

#[pyfunction]
fn simple_imputer_fit<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f64>,
    strategy: &str,
    missing_value: f64,
    missing_is_nan: bool,
) -> PyResult<FloatArray1<'py>> {
    let shape = input.shape();
    let input = input.as_slice()?;
    let strategy = simple_imputer::Strategy::try_from(strategy)?;
    let statistics = py.detach(|| {
        simple_imputer::fit(
            input,
            shape[0],
            shape[1],
            strategy,
            missing_value,
            missing_is_nan,
        )
    })?;
    Ok(statistics.into_pyarray(py))
}

#[pyfunction]
fn simple_imputer_mean_fit<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f64>,
    missing_value: f64,
    missing_is_nan: bool,
) -> PyResult<(FloatArray1<'py>, IntArray1<'py>, IntArray1<'py>)> {
    let shape = input.shape();
    let input = input.as_slice()?;
    let stats = py.detach(|| {
        simple_imputer::fit_mean(input, shape[0], shape[1], missing_value, missing_is_nan)
    })?;
    Ok((
        stats.statistics.into_pyarray(py),
        stats.missing_features.into_pyarray(py),
        stats.empty_features.into_pyarray(py),
    ))
}

#[pyfunction]
fn simple_imputer_transform<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f64>,
    statistics: PyReadonlyArray1<'py, f64>,
    retained: PyReadonlyArray1<'py, i64>,
    missing_value: f64,
    missing_is_nan: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = input.shape();
    let input = input.as_slice()?;
    let statistics = statistics.as_slice()?;
    let retained = retained.as_slice()?;
    let output = py.detach(|| {
        simple_imputer::transform(
            input,
            shape[0],
            shape[1],
            statistics,
            retained,
            missing_value,
            missing_is_nan,
        )
    })?;
    array2_output(py, output, shape[0], retained.len())
}

#[pyfunction]
fn category_discover_f64<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<FloatCategoryOutput<'py>> {
    let values = values.as_slice()?;
    let (categories, encoded) = py.detach(|| categorical::discover_numeric(values));
    Ok((categories.into_pyarray(py), encoded.into_pyarray(py)))
}

#[pyfunction]
fn category_discover_matrix_f64<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
) -> PyResult<CategoryMatrixOutput<'py, f64>> {
    let shape = values.shape();
    let values = values.as_slice()?;
    let (categories, encoded) =
        py.detach(|| categorical::discover_numeric_matrix(values, shape[0], shape[1]))?;
    Ok((
        categories,
        array2_output_i64(py, encoded, shape[0], shape[1])?,
    ))
}

#[pyfunction]
fn category_encode_f64<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    categories: PyReadonlyArray1<'py, f64>,
) -> PyResult<IntArray1<'py>> {
    let values = values.as_slice()?;
    let categories = categories.as_slice()?;
    let encoded = py.detach(|| categorical::encode_numeric(values, categories));
    Ok(encoded.into_pyarray(py))
}

#[pyfunction]
fn category_discover_i64<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<IntLabelOutput<'py>> {
    let values = values.as_slice()?;
    let (categories, encoded) = py.detach(|| categorical::discover_ordered(values));
    Ok((categories.into_pyarray(py), encoded.into_pyarray(py)))
}

#[pyfunction]
fn category_discover_matrix_i64<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, i64>,
) -> PyResult<CategoryMatrixOutput<'py, i64>> {
    let shape = values.shape();
    let values = values.as_slice()?;
    let (categories, encoded) =
        py.detach(|| categorical::discover_ordered_matrix(values, shape[0], shape[1]))?;
    Ok((
        categories,
        array2_output_i64(py, encoded, shape[0], shape[1])?,
    ))
}

#[pyfunction]
fn category_encode_i64<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, i64>,
    categories: PyReadonlyArray1<'py, i64>,
) -> PyResult<IntArray1<'py>> {
    let values = values.as_slice()?;
    let categories = categories.as_slice()?;
    let encoded = py.detach(|| categorical::encode_ordered(values, categories));
    Ok(encoded.into_pyarray(py))
}

#[pyfunction]
fn category_discover_u64<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, u64>,
) -> PyResult<UIntLabelOutput<'py>> {
    let values = values.as_slice()?;
    let (categories, encoded) = py.detach(|| categorical::discover_ordered(values));
    Ok((categories.into_pyarray(py), encoded.into_pyarray(py)))
}

#[pyfunction]
fn category_discover_matrix_u64<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, u64>,
) -> PyResult<CategoryMatrixOutput<'py, u64>> {
    let shape = values.shape();
    let values = values.as_slice()?;
    let (categories, encoded) =
        py.detach(|| categorical::discover_ordered_matrix(values, shape[0], shape[1]))?;
    Ok((
        categories,
        array2_output_i64(py, encoded, shape[0], shape[1])?,
    ))
}

#[pyfunction]
fn category_encode_u64<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, u64>,
    categories: PyReadonlyArray1<'py, u64>,
) -> PyResult<IntArray1<'py>> {
    let values = values.as_slice()?;
    let categories = categories.as_slice()?;
    let encoded = py.detach(|| categorical::encode_ordered(values, categories));
    Ok(encoded.into_pyarray(py))
}

#[pyfunction]
fn category_discover_bool<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, bool>,
) -> PyResult<(Bound<'py, PyArray1<bool>>, IntArray1<'py>)> {
    let values = values.as_slice()?;
    let (categories, encoded) = py.detach(|| categorical::discover_ordered(values));
    Ok((categories.into_pyarray(py), encoded.into_pyarray(py)))
}

#[pyfunction]
fn category_discover_matrix_bool<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, bool>,
) -> PyResult<CategoryMatrixOutput<'py, bool>> {
    let shape = values.shape();
    let values = values.as_slice()?;
    let (categories, encoded) =
        py.detach(|| categorical::discover_ordered_matrix(values, shape[0], shape[1]))?;
    Ok((
        categories,
        array2_output_i64(py, encoded, shape[0], shape[1])?,
    ))
}

#[pyfunction]
fn category_encode_bool<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, bool>,
    categories: PyReadonlyArray1<'py, bool>,
) -> PyResult<IntArray1<'py>> {
    let values = values.as_slice()?;
    let categories = categories.as_slice()?;
    let encoded = py.detach(|| categorical::encode_ordered(values, categories));
    Ok(encoded.into_pyarray(py))
}

#[pyfunction]
fn category_discover_strings<'py>(
    py: Python<'py>,
    values: Vec<String>,
) -> (Vec<String>, IntArray1<'py>) {
    let (categories, encoded) = py.detach(|| categorical::discover_strings(&values));
    (categories, encoded.into_pyarray(py))
}

#[pyfunction]
fn category_encode_strings<'py>(
    py: Python<'py>,
    values: Vec<String>,
    categories: Vec<String>,
) -> IntArray1<'py> {
    py.detach(|| categorical::encode_strings(&values, &categories))
        .into_pyarray(py)
}

#[pyfunction]
fn category_discover_unicode<'py>(
    py: Python<'py>,
    codepoints: PyReadonlyArray2<'py, u32>,
) -> PyResult<(Vec<String>, IntArray1<'py>)> {
    let shape = codepoints.shape();
    let values = codepoints.as_slice()?;
    let (categories, encoded) =
        py.detach(|| categorical::discover_unicode(values, shape[0], shape[1]))?;
    Ok((categories, encoded.into_pyarray(py)))
}

#[pyfunction]
fn category_discover_matrix_unicode<'py>(
    py: Python<'py>,
    codepoints: PyReadonlyArray3<'py, u32>,
) -> PyResult<CategoryMatrixOutput<'py, String>> {
    let shape = codepoints.shape();
    let values = codepoints.as_slice()?;
    let (categories, encoded) =
        py.detach(|| categorical::discover_unicode_matrix(values, shape[0], shape[1], shape[2]))?;
    Ok((
        categories,
        array2_output_i64(py, encoded, shape[0], shape[1])?,
    ))
}

#[pyfunction]
fn category_encode_unicode<'py>(
    py: Python<'py>,
    codepoints: PyReadonlyArray2<'py, u32>,
    categories: Vec<String>,
) -> PyResult<IntArray1<'py>> {
    let shape = codepoints.shape();
    let values = codepoints.as_slice()?;
    let encoded =
        py.detach(|| categorical::encode_unicode(values, shape[0], shape[1], &categories))?;
    Ok(encoded.into_pyarray(py))
}

#[pyfunction]
fn label_fit_transform_numeric<'py>(
    py: Python<'py>,
    values: Vec<f64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>) {
    let (classes, encoded) = py.detach(|| label_encoder::fit_transform_numeric(&values));
    (classes.into_pyarray(py), encoded.into_pyarray(py))
}

#[pyfunction]
fn label_transform_numeric<'py>(
    py: Python<'py>,
    values: Vec<f64>,
    classes: Vec<f64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    Ok(py
        .detach(|| label_encoder::transform_numeric(&values, &classes))?
        .into_pyarray(py))
}

#[pyfunction]
fn label_inverse_numeric<'py>(
    py: Python<'py>,
    codes: Vec<i64>,
    classes: Vec<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(py
        .detach(|| label_encoder::inverse_numeric(&codes, &classes))?
        .into_pyarray(py))
}

#[pyfunction]
fn label_fit_transform_i64<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, i64>,
) -> PyResult<IntLabelOutput<'py>> {
    let values = values.as_slice()?;
    let (classes, encoded) = py.detach(|| label_encoder::fit_transform_ordered(values));
    Ok((classes.into_pyarray(py), encoded.into_pyarray(py)))
}

#[pyfunction]
fn label_transform_i64<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, i64>,
    classes: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let values = values.as_slice()?;
    let classes = classes.as_slice()?;
    Ok(py
        .detach(|| label_encoder::transform_ordered(values, classes))?
        .into_pyarray(py))
}

#[pyfunction]
fn label_inverse_i64<'py>(
    py: Python<'py>,
    codes: PyReadonlyArray1<'py, i64>,
    classes: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let codes = codes.as_slice()?;
    let classes = classes.as_slice()?;
    Ok(py
        .detach(|| label_encoder::inverse_ordered(codes, classes))?
        .into_pyarray(py))
}

#[pyfunction]
fn label_fit_transform_u64<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, u64>,
) -> PyResult<UIntLabelOutput<'py>> {
    let values = values.as_slice()?;
    let (classes, encoded) = py.detach(|| label_encoder::fit_transform_ordered(values));
    Ok((classes.into_pyarray(py), encoded.into_pyarray(py)))
}

#[pyfunction]
fn label_transform_u64<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, u64>,
    classes: PyReadonlyArray1<'py, u64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let values = values.as_slice()?;
    let classes = classes.as_slice()?;
    Ok(py
        .detach(|| label_encoder::transform_ordered(values, classes))?
        .into_pyarray(py))
}

#[pyfunction]
fn label_inverse_u64<'py>(
    py: Python<'py>,
    codes: PyReadonlyArray1<'py, i64>,
    classes: PyReadonlyArray1<'py, u64>,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let codes = codes.as_slice()?;
    let classes = classes.as_slice()?;
    Ok(py
        .detach(|| label_encoder::inverse_ordered(codes, classes))?
        .into_pyarray(py))
}

#[pyfunction]
fn label_fit_transform_strings<'py>(
    py: Python<'py>,
    values: Vec<String>,
) -> (Vec<String>, Bound<'py, PyArray1<i64>>) {
    let (classes, encoded) = py.detach(|| label_encoder::fit_transform_strings(&values));
    (classes, encoded.into_pyarray(py))
}

#[pyfunction]
fn label_fit_transform_unicode<'py>(
    py: Python<'py>,
    codepoints: PyReadonlyArray2<'py, u32>,
) -> PyResult<(Vec<String>, Bound<'py, PyArray1<i64>>)> {
    let shape = codepoints.shape();
    let values = codepoints.as_slice()?;
    let (classes, encoded) =
        py.detach(|| label_encoder::fit_transform_unicode(values, shape[0], shape[1]))?;
    Ok((classes, encoded.into_pyarray(py)))
}

#[pyfunction]
fn label_transform_strings(
    py: Python<'_>,
    values: Vec<String>,
    classes: Vec<String>,
) -> PyResult<Vec<i64>> {
    py.detach(|| label_encoder::transform_strings(&values, &classes))
        .map_err(Into::into)
}

#[pyfunction]
fn label_transform_unicode<'py>(
    py: Python<'py>,
    codepoints: PyReadonlyArray2<'py, u32>,
    classes: Vec<String>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let shape = codepoints.shape();
    let values = codepoints.as_slice()?;
    Ok(py
        .detach(|| label_encoder::transform_unicode(values, shape[0], shape[1], &classes))?
        .into_pyarray(py))
}

#[pyfunction]
fn label_inverse_strings(
    py: Python<'_>,
    codes: Vec<i64>,
    classes: Vec<String>,
) -> PyResult<Vec<String>> {
    py.detach(|| label_encoder::inverse_strings(&codes, &classes))
        .map_err(Into::into)
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(build_profile, module)?)?;
    module.add_function(wrap_pyfunction!(linear_fit, module)?)?;
    module.add_function(wrap_pyfunction!(linear_predict, module)?)?;
    module.add_function(wrap_pyfunction!(linear_all_finite, module)?)?;
    module.add_function(wrap_pyfunction!(linear_coordinate_fit, module)?)?;
    module.add_function(wrap_pyfunction!(linear_coordinate_fit_validated, module)?)?;
    module.add_function(wrap_pyfunction!(logistic_fit, module)?)?;
    module.add_function(wrap_pyfunction!(logistic_fit_proximal, module)?)?;
    module.add_function(wrap_pyfunction!(logistic_softmax, module)?)?;
    module.add_function(wrap_pyfunction!(metric_accuracy, module)?)?;
    module.add_function(wrap_pyfunction!(metric_confusion_matrix, module)?)?;
    module.add_function(wrap_pyfunction!(metric_confusion_i64, module)?)?;
    module.add_function(wrap_pyfunction!(metric_regression_reductions, module)?)?;
    module.add_function(wrap_pyfunction!(metric_regression_error, module)?)?;
    module.add_function(wrap_pyfunction!(
        metric_regression_error_unweighted,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(sparse_validate_i32, module)?)?;
    module.add_function(wrap_pyfunction!(sparse_validate_i64, module)?)?;
    module.add_function(wrap_pyfunction!(sparse_scale_csr_i32_f64, module)?)?;
    module.add_function(wrap_pyfunction!(sparse_scale_csr_i64_f64, module)?)?;
    module.add_function(wrap_pyfunction!(sparse_scale_csr_i32_f32, module)?)?;
    module.add_function(wrap_pyfunction!(sparse_scale_csr_i64_f32, module)?)?;
    module.add_function(wrap_pyfunction!(one_hot_csr, module)?)?;
    module.add_function(wrap_pyfunction!(standard_fit, module)?)?;
    module.add_function(wrap_pyfunction!(standard_merge, module)?)?;
    module.add_function(wrap_pyfunction!(standard_transform, module)?)?;
    module.add_function(wrap_pyfunction!(minmax_fit, module)?)?;
    module.add_function(wrap_pyfunction!(minmax_transform, module)?)?;
    module.add_function(wrap_pyfunction!(normalize_f64, module)?)?;
    module.add_function(wrap_pyfunction!(normalize_f32, module)?)?;
    module.add_function(wrap_pyfunction!(robust_fit, module)?)?;
    module.add_function(wrap_pyfunction!(robust_transform_f64, module)?)?;
    module.add_function(wrap_pyfunction!(robust_transform_f32, module)?)?;
    module.add_function(wrap_pyfunction!(simple_imputer_fit, module)?)?;
    module.add_function(wrap_pyfunction!(simple_imputer_mean_fit, module)?)?;
    module.add_function(wrap_pyfunction!(simple_imputer_transform, module)?)?;
    module.add_function(wrap_pyfunction!(category_discover_f64, module)?)?;
    module.add_function(wrap_pyfunction!(category_discover_matrix_f64, module)?)?;
    module.add_function(wrap_pyfunction!(category_encode_f64, module)?)?;
    module.add_function(wrap_pyfunction!(category_discover_i64, module)?)?;
    module.add_function(wrap_pyfunction!(category_discover_matrix_i64, module)?)?;
    module.add_function(wrap_pyfunction!(category_encode_i64, module)?)?;
    module.add_function(wrap_pyfunction!(category_discover_u64, module)?)?;
    module.add_function(wrap_pyfunction!(category_discover_matrix_u64, module)?)?;
    module.add_function(wrap_pyfunction!(category_encode_u64, module)?)?;
    module.add_function(wrap_pyfunction!(category_discover_bool, module)?)?;
    module.add_function(wrap_pyfunction!(category_discover_matrix_bool, module)?)?;
    module.add_function(wrap_pyfunction!(category_encode_bool, module)?)?;
    module.add_function(wrap_pyfunction!(category_discover_strings, module)?)?;
    module.add_function(wrap_pyfunction!(category_encode_strings, module)?)?;
    module.add_function(wrap_pyfunction!(category_discover_unicode, module)?)?;
    module.add_function(wrap_pyfunction!(category_discover_matrix_unicode, module)?)?;
    module.add_function(wrap_pyfunction!(category_encode_unicode, module)?)?;
    module.add_function(wrap_pyfunction!(label_fit_transform_numeric, module)?)?;
    module.add_function(wrap_pyfunction!(label_transform_numeric, module)?)?;
    module.add_function(wrap_pyfunction!(label_inverse_numeric, module)?)?;
    module.add_function(wrap_pyfunction!(label_fit_transform_i64, module)?)?;
    module.add_function(wrap_pyfunction!(label_transform_i64, module)?)?;
    module.add_function(wrap_pyfunction!(label_inverse_i64, module)?)?;
    module.add_function(wrap_pyfunction!(label_fit_transform_u64, module)?)?;
    module.add_function(wrap_pyfunction!(label_transform_u64, module)?)?;
    module.add_function(wrap_pyfunction!(label_inverse_u64, module)?)?;
    module.add_function(wrap_pyfunction!(label_fit_transform_strings, module)?)?;
    module.add_function(wrap_pyfunction!(label_fit_transform_unicode, module)?)?;
    module.add_function(wrap_pyfunction!(label_transform_strings, module)?)?;
    module.add_function(wrap_pyfunction!(label_transform_unicode, module)?)?;
    module.add_function(wrap_pyfunction!(label_inverse_strings, module)?)?;
    Ok(())
}
