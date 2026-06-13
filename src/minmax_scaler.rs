use crate::error::CoreError;

#[derive(Debug, Clone)]
pub struct MinMaxStats {
    pub data_min: Vec<f64>,
    pub data_max: Vec<f64>,
    pub data_range: Vec<f64>,
}

pub fn fit(data: &[f64], rows: usize, cols: usize) -> Result<MinMaxStats, CoreError> {
    if rows == 0 || cols == 0 {
        return Err(CoreError::EmptyInput);
    }
    if data.len() != rows * cols {
        return Err(CoreError::ShapeMismatch);
    }
    let mut data_min = vec![f64::INFINITY; cols];
    let mut data_max = vec![f64::NEG_INFINITY; cols];
    let mut counts = vec![0; cols];
    for row in data.chunks_exact(cols) {
        for (column, &value) in row.iter().enumerate() {
            if value.is_nan() {
                continue;
            }
            counts[column] += 1;
            data_min[column] = data_min[column].min(value);
            data_max[column] = data_max[column].max(value);
        }
    }
    for column in 0..cols {
        if counts[column] == 0 {
            data_min[column] = f64::NAN;
            data_max[column] = f64::NAN;
        }
    }
    let data_range = data_max
        .iter()
        .zip(&data_min)
        .map(|(maximum, minimum)| maximum - minimum)
        .collect();
    Ok(MinMaxStats {
        data_min,
        data_max,
        data_range,
    })
}

// Explicit range and operation values keep Python policy out of the Rust algorithm.
#[allow(clippy::too_many_arguments)]
pub fn transform(
    data: &[f64],
    rows: usize,
    cols: usize,
    scale: &[f64],
    min: &[f64],
    output_low: f64,
    output_high: f64,
    clip: bool,
    inverse: bool,
) -> Result<Vec<f64>, CoreError> {
    if data.len() != rows * cols || scale.len() != cols || min.len() != cols {
        return Err(CoreError::ShapeMismatch);
    }
    let mut output = data.to_vec();
    for row in output.chunks_exact_mut(cols) {
        for (column, value) in row.iter_mut().enumerate() {
            if inverse {
                *value = (*value - min[column]) / scale[column];
            } else {
                *value = *value * scale[column] + min[column];
                if clip {
                    *value = value.clamp(output_low, output_high);
                }
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_ranges() {
        let stats = fit(&[1.0, -2.0, 3.0, 4.0], 2, 2).unwrap();
        assert_eq!(stats.data_min, vec![1.0, -2.0]);
        assert_eq!(stats.data_max, vec![3.0, 4.0]);
        assert_eq!(stats.data_range, vec![2.0, 6.0]);
    }

    #[test]
    fn ignores_nan_values() {
        let stats = fit(&[1.0, f64::NAN, 3.0, 4.0], 2, 2).unwrap();
        assert_eq!(stats.data_min, vec![1.0, 4.0]);
        assert_eq!(stats.data_max, vec![3.0, 4.0]);
    }
}
