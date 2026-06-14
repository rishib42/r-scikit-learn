use crate::error::CoreError;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use wide::f64x2;

#[derive(Debug, Clone)]
pub struct LinearFit {
    pub coefficients: Vec<f64>,
    pub intercepts: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct LogisticFit {
    pub coefficients: Vec<f64>,
    pub intercepts: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

#[derive(Debug, Clone)]
pub struct CoordinateFit {
    pub coefficients: Vec<f64>,
    pub intercepts: Vec<f64>,
    pub iterations: usize,
    pub dual_gaps: Vec<f64>,
    pub converged: bool,
}

fn soft_threshold(value: f64, threshold: f64) -> f64 {
    value.signum() * (value.abs() - threshold).max(0.0)
}

fn simd_dot(left: &[f64], right: &[f64]) -> f64 {
    let paired = left.len() / 2;
    let blocks = paired / 4;
    let mut sums = [f64x2::ZERO; 4];
    for block in 0..blocks {
        let offset = block * 8;
        for (lane, sum) in sums.iter_mut().enumerate() {
            let index = offset + lane * 2;
            *sum += f64x2::new([left[index], left[index + 1]])
                * f64x2::new([right[index], right[index + 1]]);
        }
    }
    let mut total = (sums[0] + sums[1] + sums[2] + sums[3]).reduce_add();
    for index in blocks * 8..left.len() {
        total += left[index] * right[index];
    }
    total
}

fn simd_add_scaled(target: &mut [f64], feature: &[f64], scale: f64) {
    let scale = f64x2::splat(scale);
    let blocks = target.len() / 8;
    for block in 0..blocks {
        let offset = block * 8;
        for lane in 0..4 {
            let index = offset + lane * 2;
            let output = f64x2::new([target[index], target[index + 1]])
                + scale * f64x2::new([feature[index], feature[index + 1]]);
            let output = output.to_array();
            target[index] = output[0];
            target[index + 1] = output[1];
        }
    }
    let scalar_scale = scale.to_array()[0];
    for index in blocks * 8..target.len() {
        target[index] += scalar_scale * feature[index];
    }
}

fn simd_squared_sum(values: &[f64]) -> f64 {
    simd_dot(values, values)
}

pub fn all_finite(input: &[f64]) -> bool {
    input.par_iter().all(|value| value.is_finite())
}

fn validate_dense(
    input: &[f64],
    rows: usize,
    columns: usize,
    targets: &[f64],
    outputs: usize,
    weights: &[f64],
) -> Result<(), CoreError> {
    if rows == 0
        || columns == 0
        || outputs == 0
        || input.len() != rows * columns
        || targets.len() != rows * outputs
        || weights.len() != rows
    {
        return Err(CoreError::ShapeMismatch);
    }
    if input.par_iter().any(|value| !value.is_finite())
        || targets.par_iter().any(|value| !value.is_finite())
        || weights
            .par_iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        || weights.iter().sum::<f64>() <= 0.0
    {
        return Err(CoreError::InputContainsNonFinite);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn fit_linear(
    input: &[f64],
    rows: usize,
    columns: usize,
    targets: &[f64],
    outputs: usize,
    weights: &[f64],
    alpha: f64,
    fit_intercept: bool,
) -> Result<LinearFit, CoreError> {
    validate_dense(input, rows, columns, targets, outputs, weights)?;
    if !alpha.is_finite() || alpha < 0.0 {
        return Err(CoreError::LinearSolverFailed);
    }
    let weight_sum = weights.iter().sum::<f64>();
    let mut x_mean = vec![0.0; columns];
    let mut y_mean = vec![0.0; outputs];
    if fit_intercept {
        for row in 0..rows {
            for column in 0..columns {
                x_mean[column] += weights[row] * input[row * columns + column];
            }
            for output in 0..outputs {
                y_mean[output] += weights[row] * targets[row * outputs + output];
            }
        }
        for value in &mut x_mean {
            *value /= weight_sum;
        }
        for value in &mut y_mean {
            *value /= weight_sum;
        }
    }
    let mut x_values = vec![0.0; rows * columns];
    let mut y_values = vec![0.0; rows * outputs];
    for row in 0..rows {
        let root_weight = weights[row].sqrt();
        for column in 0..columns {
            x_values[row * columns + column] =
                root_weight * (input[row * columns + column] - x_mean[column]);
        }
        for output in 0..outputs {
            y_values[row * outputs + output] =
                root_weight * (targets[row * outputs + output] - y_mean[output]);
        }
    }
    let x = DMatrix::from_row_slice(rows, columns, &x_values);
    let y = DMatrix::from_row_slice(rows, outputs, &y_values);
    let coefficients = if alpha > 0.0 {
        let mut gram = x.transpose() * &x;
        for column in 0..columns {
            gram[(column, column)] += alpha;
        }
        let right = x.transpose() * &y;
        match gram.clone().cholesky() {
            Some(cholesky) => cholesky.solve(&right),
            None => gram
                .svd(true, true)
                .solve(&right, f64::EPSILON * columns as f64)
                .map_err(|_| CoreError::LinearSolverFailed)?,
        }
    } else if rows >= columns {
        let gram = x.transpose() * &x;
        let right = x.transpose() * &y;
        match gram.cholesky() {
            Some(cholesky) => cholesky.solve(&right),
            None => x
                .svd(true, true)
                .solve(&y, f64::EPSILON * (rows.max(columns) as f64))
                .map_err(|_| CoreError::LinearSolverFailed)?,
        }
    } else {
        x.svd(true, true)
            .solve(&y, f64::EPSILON * (rows.max(columns) as f64))
            .map_err(|_| CoreError::LinearSolverFailed)?
    };
    if coefficients.iter().any(|value| !value.is_finite()) {
        return Err(CoreError::LinearSolverFailed);
    }
    let mut row_major = vec![0.0; outputs * columns];
    for output in 0..outputs {
        for column in 0..columns {
            row_major[output * columns + column] = coefficients[(column, output)];
        }
    }
    let intercepts = (0..outputs)
        .map(|output| {
            y_mean[output]
                - (0..columns)
                    .map(|column| row_major[output * columns + column] * x_mean[column])
                    .sum::<f64>()
        })
        .collect();
    Ok(LinearFit {
        coefficients: row_major,
        intercepts,
    })
}

pub fn predict_linear(
    input: &[f64],
    rows: usize,
    columns: usize,
    coefficients: &[f64],
    intercepts: &[f64],
) -> Result<Vec<f64>, CoreError> {
    let outputs = intercepts.len();
    if rows == 0
        || columns == 0
        || outputs == 0
        || input.len() != rows * columns
        || coefficients.len() != outputs * columns
        || input.iter().any(|value| !value.is_finite())
    {
        return Err(CoreError::ShapeMismatch);
    }
    let mut output = vec![0.0; rows * outputs];
    output
        .par_chunks_mut(outputs)
        .zip(input.par_chunks(columns))
        .for_each(|(output_row, input_row)| {
            for target in 0..outputs {
                output_row[target] = intercepts[target]
                    + input_row
                        .iter()
                        .zip(&coefficients[target * columns..(target + 1) * columns])
                        .map(|(value, coefficient)| value * coefficient)
                        .sum::<f64>();
            }
        });
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn fit_coordinate_descent(
    input: &[f64],
    rows: usize,
    columns: usize,
    targets: &[f64],
    outputs: usize,
    weights: &[f64],
    alpha: f64,
    l1_ratio: f64,
    fit_intercept: bool,
    tolerance: f64,
    max_iterations: usize,
    positive: bool,
) -> Result<CoordinateFit, CoreError> {
    validate_coordinate_shape(rows, columns, input, targets, outputs, weights)?;
    if input.par_iter().any(|value| !value.is_finite())
        || targets.par_iter().any(|value| !value.is_finite())
    {
        return Err(CoreError::InputContainsNonFinite);
    }
    fit_coordinate_descent_validated(
        input,
        rows,
        columns,
        targets,
        outputs,
        weights,
        alpha,
        l1_ratio,
        fit_intercept,
        tolerance,
        max_iterations,
        positive,
    )
}

fn validate_coordinate_shape(
    rows: usize,
    columns: usize,
    input: &[f64],
    targets: &[f64],
    outputs: usize,
    weights: &[f64],
) -> Result<(), CoreError> {
    if rows == 0
        || columns == 0
        || outputs == 0
        || input.len() != rows * columns
        || targets.len() != rows * outputs
        || weights.len() != rows
        || weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        || weights.iter().sum::<f64>() <= 0.0
    {
        return Err(CoreError::ShapeMismatch);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn fit_coordinate_descent_validated(
    input: &[f64],
    rows: usize,
    columns: usize,
    targets: &[f64],
    outputs: usize,
    weights: &[f64],
    alpha: f64,
    l1_ratio: f64,
    fit_intercept: bool,
    tolerance: f64,
    max_iterations: usize,
    positive: bool,
) -> Result<CoordinateFit, CoreError> {
    validate_coordinate_shape(rows, columns, input, targets, outputs, weights)?;
    if !alpha.is_finite()
        || alpha < 0.0
        || !l1_ratio.is_finite()
        || !(0.0..=1.0).contains(&l1_ratio)
        || !tolerance.is_finite()
        || tolerance <= 0.0
        || max_iterations == 0
    {
        return Err(CoreError::LinearSolverFailed);
    }
    let weight_sum = weights.iter().sum::<f64>();
    let mut x_mean = vec![0.0; columns];
    let mut y_mean = vec![0.0; outputs];
    if fit_intercept {
        y_mean
            .par_iter_mut()
            .enumerate()
            .for_each(|(output, mean)| {
                *mean = (0..rows)
                    .map(|row| weights[row] * targets[row * outputs + output])
                    .sum::<f64>()
                    / weight_sum;
            });
    }
    let uniform_weights = weights
        .iter()
        .all(|weight| (*weight - weights[0]).abs() <= f64::EPSILON);
    let root_weights: Vec<f64> = if uniform_weights {
        Vec::new()
    } else {
        weights
            .iter()
            .map(|weight| (weight * rows as f64 / weight_sum).sqrt())
            .collect()
    };
    // Coordinate descent repeatedly scans columns, so keep the working copy
    // column-major even though the public input is row-major.
    let mut centered = vec![0.0; rows * columns];
    centered
        .par_chunks_mut(rows)
        .zip(x_mean.par_iter_mut())
        .enumerate()
        .for_each(|(column, (feature, mean))| {
            let mut weighted_sum = 0.0;
            for row in 0..rows {
                let value = input[row * columns + column];
                feature[row] = value;
                if fit_intercept {
                    weighted_sum += weights[row] * value;
                }
            }
            if fit_intercept {
                *mean = weighted_sum / weight_sum;
            }
        });
    let mut norms = vec![0.0; columns];
    centered
        .par_chunks_mut(rows)
        .zip(norms.par_iter_mut())
        .enumerate()
        .for_each(|(column, (feature, norm))| {
            for row in 0..rows {
                let mut value = feature[row] - x_mean[column];
                if !uniform_weights {
                    value *= root_weights[row];
                }
                feature[row] = value;
            }
            *norm = simd_squared_sum(feature);
        });
    let l1 = alpha * l1_ratio;
    let l2 = alpha * (1.0 - l1_ratio);
    let l1_scaled = l1 * rows as f64;
    let l2_scaled = l2 * rows as f64;
    let mut coefficients = vec![0.0; outputs * columns];
    let mut dual_gaps = vec![0.0; outputs];
    let mut maximum_iterations = 0;
    let mut all_converged = true;
    for output in 0..outputs {
        let mut residual: Vec<f64> = (0..rows)
            .map(|row| {
                let value = targets[row * outputs + output] - y_mean[output];
                if uniform_weights {
                    value
                } else {
                    value * root_weights[row]
                }
            })
            .collect();
        let mut converged = false;
        for iteration in 0..max_iterations {
            let mut maximum_change: f64 = 0.0;
            let mut maximum_coefficient: f64 = 0.0;
            for column in 0..columns {
                let index = output * columns + column;
                let old = coefficients[index];
                let feature = &centered[column * rows..(column + 1) * rows];
                let correlation = old * norms[column] + simd_dot(feature, &residual);
                let mut new = if norms[column] == 0.0 {
                    0.0
                } else {
                    soft_threshold(correlation, l1_scaled) / (norms[column] + l2_scaled)
                };
                if positive {
                    new = new.max(0.0);
                }
                coefficients[index] = new;
                let change = old - new;
                if change != 0.0 {
                    simd_add_scaled(&mut residual, feature, change);
                }
                maximum_change = maximum_change.max((new - old).abs());
                maximum_coefficient = maximum_coefficient.max(new.abs());
            }
            maximum_iterations = maximum_iterations.max(iteration + 1);
            if maximum_change <= tolerance * maximum_coefficient.max(1.0) {
                converged = true;
                break;
            }
        }
        let coefficient_row = &coefficients[output * columns..(output + 1) * columns];
        let mut dual_norm: f64 = 0.0;
        for column in 0..columns {
            let feature = &centered[column * rows..(column + 1) * rows];
            let correlation = simd_dot(feature, &residual) - l2_scaled * coefficient_row[column];
            dual_norm = if positive {
                dual_norm.max(correlation)
            } else {
                dual_norm.max(correlation.abs())
            };
        }
        let residual_norm = simd_squared_sum(&residual);
        let coefficient_norm = coefficient_row
            .iter()
            .map(|coefficient| coefficient * coefficient)
            .sum::<f64>();
        let coefficient_l1: f64 = coefficient_row
            .iter()
            .map(|coefficient| coefficient.abs())
            .sum();
        let scale = if dual_norm > l1_scaled && dual_norm > 0.0 {
            l1_scaled / dual_norm
        } else {
            1.0
        };
        let residual_target = residual
            .iter()
            .enumerate()
            .map(|row| {
                let target = targets[row.0 * outputs + output] - y_mean[output];
                row.1
                    * if uniform_weights {
                        target
                    } else {
                        target * root_weights[row.0]
                    }
            })
            .sum::<f64>();
        dual_gaps[output] = (0.5 * residual_norm * (1.0 + scale * scale)
            + l1_scaled * coefficient_l1
            - scale * residual_target
            + 0.5 * l2_scaled * (1.0 + scale * scale) * coefficient_norm)
            .max(0.0)
            / rows as f64;
        all_converged &= converged;
    }
    let intercepts = (0..outputs)
        .map(|output| {
            y_mean[output]
                - (0..columns)
                    .map(|column| coefficients[output * columns + column] * x_mean[column])
                    .sum::<f64>()
        })
        .collect();
    Ok(CoordinateFit {
        coefficients,
        intercepts,
        iterations: maximum_iterations,
        dual_gaps,
        converged: all_converged,
    })
}

fn binary_row_reduction(
    input: &[f64],
    labels: &[i64],
    weights: &[f64],
    columns: usize,
    parameters: &[f64],
    fit_intercept: bool,
) -> Result<(f64, Vec<f64>), CoreError> {
    let width = columns + usize::from(fit_intercept);
    let rows = labels.len();
    let block_rows = 1024;
    let blocks = rows.div_ceil(block_rows);
    (0..blocks)
        .into_par_iter()
        .map(|block| {
            let start = block * block_rows;
            let end = (start + block_rows).min(rows);
            let mut loss = 0.0;
            let mut gradient = vec![0.0; width];
            for row in start..end {
                let input_row = &input[row * columns..(row + 1) * columns];
                let label = labels[row];
                let weight = weights[row];
                if label != 0 && label != 1 {
                    return Err(CoreError::InvalidMetricCode);
                }
                let score = input_row
                    .iter()
                    .zip(&parameters[..columns])
                    .map(|(value, parameter)| value * parameter)
                    .sum::<f64>()
                    + if fit_intercept {
                        parameters[columns]
                    } else {
                        0.0
                    };
                let probability = if score >= 0.0 {
                    1.0 / (1.0 + (-score).exp())
                } else {
                    let exponential = score.exp();
                    exponential / (1.0 + exponential)
                };
                loss += weight * softplus(if label == 1 { -score } else { score });
                let error = weight * (probability - label as f64);
                for column in 0..columns {
                    gradient[column] += error * input_row[column];
                }
                if fit_intercept {
                    gradient[columns] += error;
                }
            }
            Ok((loss, gradient))
        })
        .try_reduce(
            || (0.0, vec![0.0; width]),
            |(left_loss, mut left_gradient), (right_loss, right_gradient)| {
                for (left, right) in left_gradient.iter_mut().zip(right_gradient) {
                    *left += right;
                }
                Ok((left_loss + right_loss, left_gradient))
            },
        )
}

fn binary_smooth_loss(
    input: &[f64],
    labels: &[i64],
    weights: &[f64],
    columns: usize,
    parameters: &[f64],
    fit_intercept: bool,
    l2_regularization: f64,
) -> Result<f64, CoreError> {
    let rows = labels.len();
    let block_rows = 1024;
    let blocks = rows.div_ceil(block_rows);
    let loss = (0..blocks)
        .into_par_iter()
        .map(|block| {
            let start = block * block_rows;
            let end = (start + block_rows).min(rows);
            let mut loss = 0.0;
            for row in start..end {
                let label = labels[row];
                if label != 0 && label != 1 {
                    return Err(CoreError::InvalidMetricCode);
                }
                let score = input[row * columns..(row + 1) * columns]
                    .iter()
                    .zip(&parameters[..columns])
                    .map(|(value, parameter)| value * parameter)
                    .sum::<f64>()
                    + if fit_intercept {
                        parameters[columns]
                    } else {
                        0.0
                    };
                loss += weights[row] * softplus(if label == 1 { -score } else { score });
            }
            Ok(loss)
        })
        .try_reduce(|| 0.0, |left, right| Ok(left + right))?;
    let weight_sum = weights.iter().sum::<f64>();
    Ok(loss / weight_sum
        + 0.5
            * l2_regularization
            * parameters[..columns]
                .iter()
                .map(|parameter| parameter * parameter)
                .sum::<f64>())
}

fn softplus(value: f64) -> f64 {
    if value > 0.0 {
        value + (-value).exp().ln_1p()
    } else {
        value.exp().ln_1p()
    }
}

fn add_binary_regularization(
    mut loss: f64,
    mut gradient: Vec<f64>,
    weight_sum: f64,
    parameters: &[f64],
    columns: usize,
    regularization: f64,
) -> (f64, DVector<f64>) {
    loss /= weight_sum;
    for value in &mut gradient {
        *value /= weight_sum;
    }
    for column in 0..columns {
        loss += 0.5 * regularization * parameters[column] * parameters[column];
        gradient[column] += regularization * parameters[column];
    }
    (loss, DVector::from_vec(gradient))
}

#[allow(clippy::too_many_arguments)]
fn stable_binary_loss_gradient(
    input: &[f64],
    rows: usize,
    columns: usize,
    labels: &[i64],
    weights: &[f64],
    weight_sum: f64,
    parameters: &DVector<f64>,
    fit_intercept: bool,
    regularization: f64,
) -> Result<(f64, DVector<f64>), CoreError> {
    if input.len() != rows * columns {
        return Err(CoreError::ShapeMismatch);
    }
    let (loss, gradient) = binary_row_reduction(
        input,
        labels,
        weights,
        columns,
        parameters.as_slice(),
        fit_intercept,
    )?;
    Ok(add_binary_regularization(
        loss,
        gradient,
        weight_sum,
        parameters.as_slice(),
        columns,
        regularization,
    ))
}

fn should_use_parallel_binary(rows: usize, columns: usize) -> bool {
    rows.saturating_mul(columns) >= 50_000
}

#[allow(clippy::too_many_arguments)]
fn dispatch_binary_loss_gradient(
    input: &[f64],
    rows: usize,
    columns: usize,
    labels: &[i64],
    weights: &[f64],
    weight_sum: f64,
    parameters: &DVector<f64>,
    fit_intercept: bool,
    regularization: f64,
) -> Result<(f64, DVector<f64>), CoreError> {
    if should_use_parallel_binary(rows, columns) {
        stable_binary_loss_gradient(
            input,
            rows,
            columns,
            labels,
            weights,
            weight_sum,
            parameters,
            fit_intercept,
            regularization,
        )
    } else {
        binary_loss_gradient_serial(
            input,
            rows,
            columns,
            labels,
            weights,
            weight_sum,
            parameters,
            fit_intercept,
            regularization,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn binary_loss_gradient_serial(
    input: &[f64],
    rows: usize,
    columns: usize,
    labels: &[i64],
    weights: &[f64],
    weight_sum: f64,
    parameters: &DVector<f64>,
    fit_intercept: bool,
    regularization: f64,
) -> Result<(f64, DVector<f64>), CoreError> {
    let mut loss = 0.0;
    let mut gradient = vec![0.0; parameters.len()];
    for row in 0..rows {
        let label = labels[row];
        if label != 0 && label != 1 {
            return Err(CoreError::InvalidMetricCode);
        }
        let input_row = &input[row * columns..(row + 1) * columns];
        let score = input_row
            .iter()
            .zip(&parameters.as_slice()[..columns])
            .map(|(value, parameter)| value * parameter)
            .sum::<f64>()
            + if fit_intercept {
                parameters[columns]
            } else {
                0.0
            };
        let probability = if score >= 0.0 {
            1.0 / (1.0 + (-score).exp())
        } else {
            let exponential = score.exp();
            exponential / (1.0 + exponential)
        };
        loss += weights[row] * softplus(if label == 1 { -score } else { score });
        let error = weights[row] * (probability - label as f64);
        for column in 0..columns {
            gradient[column] += error * input_row[column];
        }
        if fit_intercept {
            gradient[columns] += error;
        }
    }
    Ok(add_binary_regularization(
        loss,
        gradient,
        weight_sum,
        parameters.as_slice(),
        columns,
        regularization,
    ))
}

#[allow(clippy::too_many_arguments)]
fn logistic_loss_gradient(
    input: &[f64],
    rows: usize,
    columns: usize,
    labels: &[i64],
    classes: usize,
    weights: &[f64],
    parameters: &[f64],
    fit_intercept: bool,
    regularization: f64,
) -> Result<(f64, Vec<f64>), CoreError> {
    let width = columns + usize::from(fit_intercept);
    let mut loss = 0.0;
    let mut gradient = vec![0.0; classes * width];
    let mut scores = vec![0.0; classes];
    let weight_sum = weights.iter().sum::<f64>();
    for row in 0..rows {
        let label = usize::try_from(labels[row]).map_err(|_| CoreError::InvalidMetricCode)?;
        if label >= classes {
            return Err(CoreError::InvalidMetricCode);
        }
        for class in 0..classes {
            scores[class] = (0..columns)
                .map(|column| input[row * columns + column] * parameters[class * width + column])
                .sum::<f64>()
                + if fit_intercept {
                    parameters[class * width + columns]
                } else {
                    0.0
                };
        }
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let denominator = scores
            .iter_mut()
            .map(|score| {
                *score = (*score - maximum).exp();
                *score
            })
            .sum::<f64>();
        loss -= weights[row] * (scores[label] / denominator).ln();
        for class in 0..classes {
            let error = weights[row] * (scores[class] / denominator - f64::from(class == label));
            for column in 0..columns {
                gradient[class * width + column] += error * input[row * columns + column];
            }
            if fit_intercept {
                gradient[class * width + columns] += error;
            }
        }
    }
    loss /= weight_sum;
    for class in 0..classes {
        for column in 0..columns {
            let index = class * width + column;
            loss += 0.5 * regularization * parameters[index] * parameters[index];
            gradient[index] = gradient[index] / weight_sum + regularization * parameters[index];
        }
        if fit_intercept {
            gradient[class * width + columns] /= weight_sum;
        }
    }
    Ok((loss, gradient))
}

#[allow(clippy::too_many_arguments)]
fn fit_binary_logistic(
    input: &[f64],
    rows: usize,
    columns: usize,
    labels: &[i64],
    weights: &[f64],
    regularization: f64,
    fit_intercept: bool,
    tolerance: f64,
    max_iterations: usize,
) -> Result<LogisticFit, CoreError> {
    let width = columns + usize::from(fit_intercept);
    let mut parameters = DVector::zeros(width);
    let mut inverse_hessian: DMatrix<f64> = DMatrix::identity(width, width) * 4.0;
    let weight_sum = weights.iter().sum::<f64>();
    let mut converged = false;
    let mut completed = 0;
    let (mut loss, mut gradient) = dispatch_binary_loss_gradient(
        input,
        rows,
        columns,
        labels,
        weights,
        weight_sum,
        &parameters,
        fit_intercept,
        regularization,
    )?;
    for iteration in 0..max_iterations {
        let gradient_norm = gradient.iter().copied().map(f64::abs).fold(0.0, f64::max);
        completed = iteration + 1;
        if gradient_norm <= tolerance {
            converged = true;
            break;
        }
        let direction = -(&inverse_hessian * &gradient);
        let slope = gradient.dot(&direction);
        let mut step_size = 1.0;
        let (candidate, candidate_loss, candidate_gradient) = loop {
            let candidate = &parameters + step_size * &direction;
            let (candidate_loss, candidate_gradient) = dispatch_binary_loss_gradient(
                input,
                rows,
                columns,
                labels,
                weights,
                weight_sum,
                &candidate,
                fit_intercept,
                regularization,
            )?;
            if candidate_loss <= loss + 1e-4 * step_size * slope || step_size <= 1e-12 {
                break (candidate, candidate_loss, candidate_gradient);
            }
            step_size *= 0.5;
        };
        let displacement = &candidate - &parameters;
        let gradient_change = &candidate_gradient - &gradient;
        let curvature = gradient_change.dot(&displacement);
        if curvature > 1e-12 {
            let rho = 1.0 / curvature;
            let identity = DMatrix::identity(width, width);
            inverse_hessian = (&identity - rho * &displacement * gradient_change.transpose())
                * inverse_hessian
                * (&identity - rho * &gradient_change * displacement.transpose())
                + rho * &displacement * displacement.transpose();
        } else {
            inverse_hessian = DMatrix::identity(width, width) * 4.0;
        }
        parameters = candidate;
        loss = candidate_loss;
        gradient = candidate_gradient;
    }
    let mut coefficients = vec![0.0; 2 * columns];
    let mut intercepts = vec![0.0; 2];
    for column in 0..columns {
        coefficients[column] = -0.5 * parameters[column];
        coefficients[columns + column] = 0.5 * parameters[column];
    }
    if fit_intercept {
        intercepts[0] = -0.5 * parameters[columns];
        intercepts[1] = 0.5 * parameters[columns];
    }
    Ok(LogisticFit {
        coefficients,
        intercepts,
        iterations: completed,
        converged,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn fit_binary_logistic_proximal(
    input: &[f64],
    rows: usize,
    columns: usize,
    labels: &[i64],
    weights: &[f64],
    l1_regularization: f64,
    l2_regularization: f64,
    fit_intercept: bool,
    tolerance: f64,
    max_iterations: usize,
) -> Result<LogisticFit, CoreError> {
    if rows == 0
        || columns == 0
        || labels.len() != rows
        || weights.len() != rows
        || input.len() != rows * columns
        || input.iter().chain(weights).any(|value| !value.is_finite())
        || weights.iter().any(|weight| *weight < 0.0)
        || weights.iter().sum::<f64>() <= 0.0
        || !l1_regularization.is_finite()
        || l1_regularization < 0.0
        || !l2_regularization.is_finite()
        || l2_regularization < 0.0
        || !tolerance.is_finite()
        || tolerance <= 0.0
        || max_iterations == 0
    {
        return Err(CoreError::ShapeMismatch);
    }
    let width = columns + usize::from(fit_intercept);
    let weight_sum = weights.iter().sum::<f64>();
    let mut parameters = DVector::zeros(width);
    let mut accelerated = parameters.clone();
    let mut momentum = 1.0;
    let mut step = 1.0;
    let mut completed = 0;
    let mut converged = false;
    for iteration in 0..max_iterations {
        let (smooth_loss, gradient) = dispatch_binary_loss_gradient(
            input,
            rows,
            columns,
            labels,
            weights,
            weight_sum,
            &accelerated,
            fit_intercept,
            l2_regularization,
        )?;
        let candidate = loop {
            let mut candidate = &accelerated - step * &gradient;
            for column in 0..columns {
                candidate[column] = soft_threshold(candidate[column], step * l1_regularization);
            }
            let candidate_smooth_loss = binary_smooth_loss(
                input,
                labels,
                weights,
                columns,
                candidate.as_slice(),
                fit_intercept,
                l2_regularization,
            )?;
            let displacement = &candidate - &accelerated;
            let bound = smooth_loss
                + gradient.dot(&displacement)
                + displacement.norm_squared() / (2.0 * step);
            if candidate_smooth_loss <= bound + 1e-12 || step <= 1e-12 {
                break candidate;
            }
            step *= 0.5;
        };
        completed = iteration + 1;
        let maximum_change = candidate
            .iter()
            .zip(parameters.iter())
            .map(|(new, old)| (new - old).abs())
            .fold(0.0, f64::max);
        let scale = candidate.iter().copied().map(f64::abs).fold(1.0, f64::max);
        if maximum_change <= tolerance * scale {
            parameters = candidate;
            converged = true;
            break;
        }
        let next_momentum = (1.0_f64 + (1.0_f64 + 4.0 * momentum * momentum).sqrt()) / 2.0;
        accelerated = &candidate + ((momentum - 1.0) / next_momentum) * (&candidate - &parameters);
        parameters = candidate;
        momentum = next_momentum;
        step = (step * 1.2).min(1.0);
    }
    let mut coefficients = vec![0.0; 2 * columns];
    let mut intercepts = vec![0.0; 2];
    for column in 0..columns {
        coefficients[column] = -0.5 * parameters[column];
        coefficients[columns + column] = 0.5 * parameters[column];
    }
    if fit_intercept {
        intercepts[0] = -0.5 * parameters[columns];
        intercepts[1] = 0.5 * parameters[columns];
    }
    Ok(LogisticFit {
        coefficients,
        intercepts,
        iterations: completed,
        converged,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn fit_logistic(
    input: &[f64],
    rows: usize,
    columns: usize,
    labels: &[i64],
    classes: usize,
    weights: &[f64],
    inverse_c: f64,
    fit_intercept: bool,
    tolerance: f64,
    max_iterations: usize,
) -> Result<LogisticFit, CoreError> {
    if classes < 2 {
        return Err(CoreError::InsufficientClasses);
    }
    if rows == 0
        || columns == 0
        || labels.len() != rows
        || input.len() != rows * columns
        || weights.len() != rows
        || input.iter().chain(weights).any(|value| !value.is_finite())
        || weights.iter().any(|weight| *weight < 0.0)
        || weights.iter().sum::<f64>() <= 0.0
    {
        return Err(CoreError::ShapeMismatch);
    }
    if !inverse_c.is_finite() || inverse_c < 0.0 || !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(CoreError::LinearSolverFailed);
    }
    let regularization = inverse_c / weights.iter().sum::<f64>();
    if classes == 2 {
        return fit_binary_logistic(
            input,
            rows,
            columns,
            labels,
            weights,
            regularization,
            fit_intercept,
            tolerance,
            max_iterations,
        );
    }
    let width = columns + usize::from(fit_intercept);
    let mut parameters = vec![0.0; classes * width];
    let mut converged = false;
    let mut completed = 0;
    for iteration in 0..max_iterations {
        let (loss, gradient) = logistic_loss_gradient(
            input,
            rows,
            columns,
            labels,
            classes,
            weights,
            &parameters,
            fit_intercept,
            regularization,
        )?;
        let gradient_norm = gradient
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        completed = iteration + 1;
        if gradient_norm <= tolerance {
            converged = true;
            break;
        }
        let mut step = 1.0;
        let mut accepted = false;
        while step >= 1e-12 {
            let candidate: Vec<f64> = parameters
                .iter()
                .zip(&gradient)
                .map(|(parameter, gradient)| parameter - step * gradient)
                .collect();
            let (candidate_loss, _) = logistic_loss_gradient(
                input,
                rows,
                columns,
                labels,
                classes,
                weights,
                &candidate,
                fit_intercept,
                regularization,
            )?;
            if candidate_loss <= loss - 1e-4 * step * gradient_norm * gradient_norm {
                parameters = candidate;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    let mut coefficients = vec![0.0; classes * columns];
    let mut intercepts = vec![0.0; classes];
    for class in 0..classes {
        coefficients[class * columns..(class + 1) * columns]
            .copy_from_slice(&parameters[class * width..class * width + columns]);
        if fit_intercept {
            intercepts[class] = parameters[class * width + columns];
        }
    }
    Ok(LogisticFit {
        coefficients,
        intercepts,
        iterations: completed,
        converged,
    })
}

pub fn softmax(scores: &[f64], rows: usize, classes: usize) -> Result<Vec<f64>, CoreError> {
    if rows == 0 || classes < 2 || scores.len() != rows * classes {
        return Err(CoreError::ShapeMismatch);
    }
    let mut output = scores.to_vec();
    for row in output.chunks_exact_mut(classes) {
        let maximum = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let denominator = row
            .iter_mut()
            .map(|score| {
                *score = (*score - maximum).exp();
                *score
            })
            .sum::<f64>();
        for score in row {
            *score /= denominator;
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_coordinate_helpers_handle_full_vectors_and_remainders() {
        let left = [1.0, 2.0, 3.0, 4.0, 5.0];
        let right = [5.0, 4.0, 3.0, 2.0, 1.0];
        assert_eq!(simd_dot(&left, &right), 35.0);
        assert_eq!(simd_squared_sum(&left), 55.0);
        let mut output = left;
        simd_add_scaled(&mut output, &right, -0.5);
        assert_eq!(output, [-1.5, 0.0, 1.5, 3.0, 4.5]);
    }

    #[test]
    fn linear_fit_recovers_multioutput_coefficients() {
        let x = [1.0, 2.0, 2.0, 0.0, 3.0, 1.0, 4.0, 3.0];
        let y = [6.0, 7.0, 5.0, 2.0, 8.5, 3.0, 13.5, 6.0];
        let fit = fit_linear(&x, 4, 2, &y, 2, &[1.0; 4], 0.0, true).unwrap();
        assert!((fit.intercepts[0] - 1.0).abs() < 1e-10);
        assert!((fit.intercepts[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn logistic_fit_separates_simple_classes() {
        let x = [-2.0, -1.0, 1.0, 2.0];
        let fit =
            fit_logistic(&x, 4, 1, &[0, 0, 1, 1], 2, &[1.0; 4], 1.0, true, 1e-6, 500).unwrap();
        assert!(fit.coefficients[0] < fit.coefficients[1]);
    }

    #[test]
    fn coordinate_descent_recovers_sparse_coefficients() {
        let x = [1.0, 0.0, 2.0, 1.0, 3.0, -1.0, 4.0, 2.0];
        let y = [3.0, 5.0, 7.0, 9.0];
        let fit = fit_coordinate_descent(
            &x, 4, 2, &y, 1, &[1.0; 4], 0.1, 1.0, true, 1e-10, 1000, false,
        )
        .unwrap();
        assert!(fit.coefficients[0] > 1.5);
        assert_eq!(fit.coefficients[1], 0.0);
        assert!(fit.dual_gaps[0] < 1e-8);
        assert!(fit.converged);
    }

    #[test]
    fn proximal_logistic_produces_sparse_binary_model() {
        let x = [-3.0, 1.0, -2.0, -1.0, 2.0, 1.0, 3.0, -1.0];
        let fit = fit_binary_logistic_proximal(
            &x,
            4,
            2,
            &[0, 0, 1, 1],
            &[1.0; 4],
            0.1,
            0.0,
            true,
            1e-7,
            1000,
        )
        .unwrap();
        assert!(fit.coefficients[2] > 0.0);
        assert_eq!(fit.coefficients[1], 0.0);
        assert_eq!(fit.coefficients[3], 0.0);
        assert!(fit.converged);
    }
}
