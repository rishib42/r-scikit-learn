use crate::error::CoreError;

#[derive(Debug, Clone)]
pub struct RobustStats {
    pub center: Vec<f64>,
    pub scale: Vec<f64>,
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let index = quantile / 100.0 * (sorted.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    let fraction = index - lower as f64;
    sorted[lower] + (sorted[upper] - sorted[lower]) * fraction
}

pub fn fit(
    data: &[f64],
    rows: usize,
    cols: usize,
    quantile_low: f64,
    quantile_high: f64,
    with_centering: bool,
    with_scaling: bool,
) -> Result<RobustStats, CoreError> {
    if rows == 0 || cols == 0 {
        return Err(CoreError::EmptyInput);
    }
    if data.len() != rows * cols {
        return Err(CoreError::ShapeMismatch);
    }
    if !(0.0..=100.0).contains(&quantile_low)
        || !(0.0..=100.0).contains(&quantile_high)
        || quantile_low > quantile_high
    {
        return Err(CoreError::InvalidQuantileRange(quantile_low, quantile_high));
    }

    let mut center = Vec::with_capacity(if with_centering { cols } else { 0 });
    let mut scale = Vec::with_capacity(if with_scaling { cols } else { 0 });
    let mut column = Vec::with_capacity(rows);
    for column_index in 0..cols {
        column.clear();
        column.extend(
            data.iter()
                .skip(column_index)
                .step_by(cols)
                .copied()
                .filter(|value| !value.is_nan()),
        );
        column.sort_unstable_by(f64::total_cmp);
        if with_centering {
            center.push(percentile(&column, 50.0));
        }
        if with_scaling {
            let low = percentile(&column, quantile_low);
            let high = percentile(&column, quantile_high);
            let difference = high - low;
            scale.push(if difference < 10.0 * f64::EPSILON {
                1.0
            } else {
                difference
            });
        }
    }
    Ok(RobustStats { center, scale })
}

#[allow(clippy::too_many_arguments)]
pub fn transform_f64(
    data: &[f64],
    rows: usize,
    cols: usize,
    center: &[f64],
    scale: &[f64],
    with_centering: bool,
    with_scaling: bool,
    inverse: bool,
) -> Result<Vec<f64>, CoreError> {
    if data.len() != rows * cols
        || (with_centering && center.len() != cols)
        || (with_scaling && scale.len() != cols)
    {
        return Err(CoreError::ShapeMismatch);
    }
    let mut output = data.to_vec();
    for row in output.chunks_exact_mut(cols) {
        for (column, value) in row.iter_mut().enumerate() {
            if inverse {
                if with_scaling {
                    *value *= scale[column];
                }
                if with_centering {
                    *value += center[column];
                }
            } else {
                if with_centering {
                    *value -= center[column];
                }
                if with_scaling {
                    *value /= scale[column];
                }
            }
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn transform_f32(
    data: &[f32],
    rows: usize,
    cols: usize,
    center: &[f64],
    scale: &[f64],
    with_centering: bool,
    with_scaling: bool,
    inverse: bool,
) -> Result<Vec<f32>, CoreError> {
    if data.len() != rows * cols
        || (with_centering && center.len() != cols)
        || (with_scaling && scale.len() != cols)
    {
        return Err(CoreError::ShapeMismatch);
    }
    let mut output = data.to_vec();
    for row in output.chunks_exact_mut(cols) {
        for (column, value) in row.iter_mut().enumerate() {
            let mut transformed = f64::from(*value);
            if inverse {
                if with_scaling {
                    transformed *= scale[column];
                }
                if with_centering {
                    transformed += center[column];
                }
            } else {
                if with_centering {
                    transformed -= center[column];
                }
                if with_scaling {
                    transformed /= scale[column];
                }
            }
            *value = transformed as f32;
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_nan_aware_median_and_interquartile_range() {
        let stats = fit(
            &[1.0, f64::NAN, 2.0, 4.0, 100.0, 8.0],
            3,
            2,
            25.0,
            75.0,
            true,
            true,
        )
        .unwrap();
        assert_eq!(stats.center, vec![2.0, 6.0]);
        assert_eq!(stats.scale, vec![49.5, 2.0]);
    }

    #[test]
    fn handles_constant_and_all_nan_features() {
        let stats = fit(
            &[4.0, f64::NAN, 4.0, f64::NAN],
            2,
            2,
            25.0,
            75.0,
            true,
            true,
        )
        .unwrap();
        assert_eq!(stats.center[0], 4.0);
        assert_eq!(stats.scale[0], 1.0);
        assert!(stats.center[1].is_nan());
        assert!(stats.scale[1].is_nan());
    }

    #[test]
    fn transforms_and_roundtrips_f64_and_f32() {
        let center = [2.0, 6.0];
        let scale = [2.0, 2.0];
        let data = [1.0, 4.0, 5.0, 8.0];
        let transformed = transform_f64(&data, 2, 2, &center, &scale, true, true, false).unwrap();
        assert_eq!(transformed, vec![-0.5, -1.0, 1.5, 1.0]);
        assert_eq!(
            transform_f64(&transformed, 2, 2, &center, &scale, true, true, true).unwrap(),
            data
        );
        let data_f32 = [1.0_f32, 4.0, 5.0, 8.0];
        assert_eq!(
            transform_f32(&data_f32, 2, 2, &center, &scale, true, true, false).unwrap(),
            vec![-0.5_f32, -1.0, 1.5, 1.0]
        );
    }
}
