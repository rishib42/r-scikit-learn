use crate::error::CoreError;
use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Debug, Clone)]
pub struct RegressionReductions {
    pub absolute_error: Vec<f64>,
    pub squared_error: Vec<f64>,
    pub true_sum: Vec<f64>,
    pub true_squared_sum: Vec<f64>,
    pub weight_sum: f64,
}

fn validate_weights(weights: &[f64], rows: usize) -> Result<(), CoreError> {
    if weights.len() != rows
        || weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err(CoreError::InvalidSampleWeight);
    }
    Ok(())
}

pub fn accuracy(
    expected: &[i64],
    predicted: &[i64],
    weights: &[f64],
) -> Result<(f64, f64), CoreError> {
    if expected.len() != predicted.len() || expected.is_empty() {
        return Err(CoreError::ShapeMismatch);
    }
    validate_weights(weights, expected.len())?;
    let mut correct = 0.0;
    let mut total = 0.0;
    for ((&expected, &predicted), &weight) in expected.iter().zip(predicted).zip(weights) {
        total += weight;
        if expected == predicted {
            correct += weight;
        }
    }
    Ok((correct, total))
}

pub fn confusion_matrix(
    expected: &[i64],
    predicted: &[i64],
    weights: &[f64],
    classes: usize,
) -> Result<Vec<f64>, CoreError> {
    if expected.len() != predicted.len() || expected.is_empty() || classes == 0 {
        return Err(CoreError::ShapeMismatch);
    }
    validate_weights(weights, expected.len())?;
    let mut output = vec![0.0; classes * classes];
    for ((&expected, &predicted), &weight) in expected.iter().zip(predicted).zip(weights) {
        let expected = usize::try_from(expected).map_err(|_| CoreError::InvalidMetricCode)?;
        let predicted = usize::try_from(predicted).map_err(|_| CoreError::InvalidMetricCode)?;
        if expected >= classes || predicted >= classes {
            return Err(CoreError::InvalidMetricCode);
        }
        output[expected * classes + predicted] += weight;
    }
    Ok(output)
}

pub fn confusion_i64(
    expected: &[i64],
    predicted: &[i64],
    weights: &[f64],
) -> Result<(Vec<i64>, Vec<f64>), CoreError> {
    if expected.len() != predicted.len() || expected.is_empty() {
        return Err(CoreError::ShapeMismatch);
    }
    validate_weights(weights, expected.len())?;
    let mut unique = FxHashSet::default();
    unique.extend(expected.iter().copied());
    unique.extend(predicted.iter().copied());
    let mut classes: Vec<i64> = unique.into_iter().collect();
    classes.sort();
    let dense_codes = classes
        .iter()
        .enumerate()
        .all(|(index, &value)| value == index as i64);
    if dense_codes {
        let mut output = vec![0.0; classes.len() * classes.len()];
        for ((&expected, &predicted), &weight) in expected.iter().zip(predicted).zip(weights) {
            let row = usize::try_from(expected).map_err(|_| CoreError::InvalidMetricCode)?;
            let column = usize::try_from(predicted).map_err(|_| CoreError::InvalidMetricCode)?;
            output[row * classes.len() + column] += weight;
        }
        return Ok((classes, output));
    }
    let mapping: FxHashMap<i64, usize> = classes
        .iter()
        .enumerate()
        .map(|(index, &value)| (value, index))
        .collect();
    let mut output = vec![0.0; classes.len() * classes.len()];
    for ((expected, predicted), &weight) in expected.iter().zip(predicted).zip(weights) {
        let row = mapping
            .get(expected)
            .copied()
            .ok_or(CoreError::InvalidMetricCode)?;
        let column = mapping
            .get(predicted)
            .copied()
            .ok_or(CoreError::InvalidMetricCode)?;
        output[row * classes.len() + column] += weight;
    }
    Ok((classes, output))
}

pub fn regression_reductions(
    expected: &[f64],
    predicted: &[f64],
    weights: &[f64],
    rows: usize,
    columns: usize,
) -> Result<RegressionReductions, CoreError> {
    if rows == 0
        || columns == 0
        || expected.len() != rows * columns
        || predicted.len() != rows * columns
    {
        return Err(CoreError::ShapeMismatch);
    }
    validate_weights(weights, rows)?;
    if expected
        .iter()
        .chain(predicted)
        .any(|value| !value.is_finite())
    {
        return Err(CoreError::InputContainsNonFinite);
    }
    let mut absolute_error = vec![0.0; columns];
    let mut squared_error = vec![0.0; columns];
    let mut true_sum = vec![0.0; columns];
    let mut true_squared_sum = vec![0.0; columns];
    let weight_sum = weights.iter().sum();
    for (row, &weight) in weights.iter().enumerate() {
        for column in 0..columns {
            let index = row * columns + column;
            let actual = expected[index];
            let difference = actual - predicted[index];
            absolute_error[column] += weight * difference.abs();
            squared_error[column] += weight * difference * difference;
            true_sum[column] += weight * actual;
            true_squared_sum[column] += weight * actual * actual;
        }
    }
    Ok(RegressionReductions {
        absolute_error,
        squared_error,
        true_sum,
        true_squared_sum,
        weight_sum,
    })
}

pub fn regression_error(
    expected: &[f64],
    predicted: &[f64],
    weights: &[f64],
    rows: usize,
    columns: usize,
    squared: bool,
) -> Result<(Vec<f64>, f64), CoreError> {
    if rows == 0
        || columns == 0
        || expected.len() != rows * columns
        || predicted.len() != rows * columns
    {
        return Err(CoreError::ShapeMismatch);
    }
    validate_weights(weights, rows)?;
    if expected
        .iter()
        .chain(predicted)
        .any(|value| !value.is_finite())
    {
        return Err(CoreError::InputContainsNonFinite);
    }
    let mut error = vec![0.0; columns];
    if squared {
        for ((expected_row, predicted_row), &weight) in expected
            .chunks_exact(columns)
            .zip(predicted.chunks_exact(columns))
            .zip(weights)
        {
            for ((output, &expected), &predicted) in
                error.iter_mut().zip(expected_row).zip(predicted_row)
            {
                let difference = expected - predicted;
                *output += weight * difference * difference;
            }
        }
    } else {
        for ((expected_row, predicted_row), &weight) in expected
            .chunks_exact(columns)
            .zip(predicted.chunks_exact(columns))
            .zip(weights)
        {
            for ((output, &expected), &predicted) in
                error.iter_mut().zip(expected_row).zip(predicted_row)
            {
                *output += weight * (expected - predicted).abs();
            }
        }
    }
    Ok((error, weights.iter().sum()))
}

pub fn regression_error_unweighted(
    expected: &[f64],
    predicted: &[f64],
    rows: usize,
    columns: usize,
    squared: bool,
) -> Result<Vec<f64>, CoreError> {
    if rows == 0
        || columns == 0
        || expected.len() != rows * columns
        || predicted.len() != rows * columns
    {
        return Err(CoreError::ShapeMismatch);
    }
    let mut error = vec![0.0; columns];
    if squared {
        for (expected_row, predicted_row) in expected
            .chunks_exact(columns)
            .zip(predicted.chunks_exact(columns))
        {
            for ((output, &expected), &predicted) in
                error.iter_mut().zip(expected_row).zip(predicted_row)
            {
                if !expected.is_finite() || !predicted.is_finite() {
                    return Err(CoreError::InputContainsNonFinite);
                }
                let difference = expected - predicted;
                *output += difference * difference;
            }
        }
    } else {
        for (expected_row, predicted_row) in expected
            .chunks_exact(columns)
            .zip(predicted.chunks_exact(columns))
        {
            for ((output, &expected), &predicted) in
                error.iter_mut().zip(expected_row).zip(predicted_row)
            {
                if !expected.is_finite() || !predicted.is_finite() {
                    return Err(CoreError::InputContainsNonFinite);
                }
                *output += (expected - predicted).abs();
            }
        }
    }
    Ok(error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulates_weighted_classification_metrics() {
        let weights = [1.0, 2.0, 3.0];
        assert_eq!(
            accuracy(&[0, 1, 1], &[0, 0, 1], &weights).unwrap(),
            (4.0, 6.0)
        );
        assert_eq!(
            confusion_matrix(&[0, 1, 1], &[0, 0, 1], &weights, 2).unwrap(),
            vec![1.0, 0.0, 2.0, 3.0]
        );
        assert_eq!(
            confusion_i64(&[2, 1, 2], &[1, 1, 2], &weights).unwrap(),
            (vec![1, 2], vec![2.0, 0.0, 1.0, 3.0])
        );
    }

    #[test]
    fn computes_multioutput_regression_reductions() {
        let stats = regression_reductions(
            &[1.0, 2.0, 3.0, 4.0],
            &[0.0, 2.0, 5.0, 3.0],
            &[1.0, 2.0],
            2,
            2,
        )
        .unwrap();
        assert_eq!(stats.absolute_error, vec![5.0, 2.0]);
        assert_eq!(stats.squared_error, vec![9.0, 2.0]);
        assert_eq!(stats.true_sum, vec![7.0, 10.0]);
        assert_eq!(stats.weight_sum, 3.0);
        assert_eq!(
            regression_error(&[1.0, 3.0], &[0.0, 5.0], &[1.0, 2.0], 2, 1, true).unwrap(),
            (vec![9.0], 3.0)
        );
        assert_eq!(
            regression_error_unweighted(&[1.0, 3.0], &[0.0, 5.0], 2, 1, true).unwrap(),
            vec![5.0]
        );
    }
}
