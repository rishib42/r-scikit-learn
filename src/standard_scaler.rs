use crate::error::CoreError;

#[derive(Debug, Clone)]
pub struct StandardStats {
    pub mean: Vec<f64>,
    pub variance: Vec<f64>,
    pub scale: Vec<f64>,
}

pub fn fit(data: &[f64], rows: usize, cols: usize) -> Result<StandardStats, CoreError> {
    if rows == 0 || cols == 0 {
        return Err(CoreError::EmptyInput);
    }
    if data.len() != rows * cols {
        return Err(CoreError::ShapeMismatch);
    }

    let mut mean = vec![0.0; cols];
    let mut m2 = vec![0.0; cols];
    for (row_index, row) in data.chunks_exact(cols).enumerate() {
        let count = (row_index + 1) as f64;
        for (column, &value) in row.iter().enumerate() {
            let delta = value - mean[column];
            mean[column] += delta / count;
            let delta2 = value - mean[column];
            m2[column] += delta * delta2;
        }
    }

    let variance: Vec<f64> = m2.into_iter().map(|value| value / rows as f64).collect();
    let scale = variance
        .iter()
        .map(|&value| if value == 0.0 { 1.0 } else { value.sqrt() })
        .collect();
    Ok(StandardStats {
        mean,
        variance,
        scale,
    })
}

// Explicit operation flags keep the algorithm reusable by both forward and inverse bindings.
#[allow(clippy::too_many_arguments)]
pub fn transform(
    data: &[f64],
    rows: usize,
    cols: usize,
    mean: &[f64],
    scale: &[f64],
    with_mean: bool,
    with_std: bool,
    inverse: bool,
) -> Result<Vec<f64>, CoreError> {
    if data.len() != rows * cols || mean.len() != cols || scale.len() != cols {
        return Err(CoreError::ShapeMismatch);
    }
    let mut output = data.to_vec();
    for row in output.chunks_exact_mut(cols) {
        for (column, value) in row.iter_mut().enumerate() {
            if inverse {
                if with_std {
                    *value *= scale[column];
                }
                if with_mean {
                    *value += mean[column];
                }
            } else {
                if with_mean {
                    *value -= mean[column];
                }
                if with_std {
                    *value /= scale[column];
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
    fn computes_population_statistics_and_roundtrips() {
        let data = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
        let stats = fit(&data, 3, 2).unwrap();
        assert_eq!(stats.mean, vec![2.0, 20.0]);
        assert!((stats.variance[0] - 2.0 / 3.0).abs() < 1e-12);
        let scaled = transform(&data, 3, 2, &stats.mean, &stats.scale, true, true, false).unwrap();
        let restored =
            transform(&scaled, 3, 2, &stats.mean, &stats.scale, true, true, true).unwrap();
        assert_eq!(restored, data);
    }

    #[test]
    fn constant_feature_has_unit_scale() {
        let stats = fit(&[2.0, 2.0, 2.0], 3, 1).unwrap();
        assert_eq!(stats.scale, vec![1.0]);
    }
}
