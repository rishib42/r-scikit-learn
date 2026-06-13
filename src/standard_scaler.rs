use crate::error::CoreError;

#[derive(Debug, Clone)]
pub struct StandardStats {
    pub mean: Vec<f64>,
    pub variance: Vec<f64>,
    pub scale: Vec<f64>,
    pub counts: Vec<usize>,
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
    let mut counts = vec![0; cols];
    for row in data.chunks_exact(cols) {
        for (column, &value) in row.iter().enumerate() {
            if value.is_nan() {
                continue;
            }
            counts[column] += 1;
            let count = counts[column] as f64;
            let delta = value - mean[column];
            mean[column] += delta / count;
            let delta2 = value - mean[column];
            m2[column] += delta * delta2;
        }
    }

    for (value, &count) in mean.iter_mut().zip(&counts) {
        if count == 0 {
            *value = f64::NAN;
        }
    }
    let variance: Vec<f64> = m2
        .into_iter()
        .zip(&counts)
        .map(|(value, &count)| {
            if count == 0 {
                f64::NAN
            } else {
                value / count as f64
            }
        })
        .collect();
    let scale = variance
        .iter()
        .map(|&value| if value == 0.0 { 1.0 } else { value.sqrt() })
        .collect();
    Ok(StandardStats {
        mean,
        variance,
        scale,
        counts,
    })
}

pub fn merge(
    previous_mean: &[f64],
    previous_variance: &[f64],
    previous_counts: &[usize],
    batch: &StandardStats,
) -> Result<StandardStats, CoreError> {
    let cols = previous_mean.len();
    if previous_variance.len() != cols
        || previous_counts.len() != cols
        || batch.mean.len() != cols
        || batch.variance.len() != cols
        || batch.counts.len() != cols
    {
        return Err(CoreError::ShapeMismatch);
    }
    let mut mean = vec![f64::NAN; cols];
    let mut variance = vec![f64::NAN; cols];
    let mut counts = vec![0; cols];
    for column in 0..cols {
        let old_count = previous_counts[column];
        let batch_count = batch.counts[column];
        counts[column] = old_count + batch_count;
        match (old_count, batch_count) {
            (0, 0) => {}
            (0, _) => {
                // Match scikit-learn incremental behavior: an all-NaN first
                // batch leaves the running mean NaN even if later batches
                // contain valid values, while variance can still recover.
                mean[column] = previous_mean[column];
                variance[column] = batch.variance[column];
            }
            (_, 0) => {
                mean[column] = previous_mean[column];
                variance[column] = previous_variance[column];
            }
            _ => {
                let total = counts[column] as f64;
                let delta = batch.mean[column] - previous_mean[column];
                mean[column] = previous_mean[column] + delta * batch_count as f64 / total;
                let old_m2 = previous_variance[column] * old_count as f64;
                let batch_m2 = batch.variance[column] * batch_count as f64;
                let correction = delta * delta * old_count as f64 * batch_count as f64 / total;
                variance[column] = (old_m2 + batch_m2 + correction) / total;
            }
        }
    }
    let scale = variance
        .iter()
        .map(|&value| if value == 0.0 { 1.0 } else { value.sqrt() })
        .collect();
    Ok(StandardStats {
        mean,
        variance,
        scale,
        counts,
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

    #[test]
    fn ignores_nan_and_tracks_counts() {
        let stats = fit(&[1.0, f64::NAN, 3.0, 4.0], 2, 2).unwrap();
        assert_eq!(stats.counts, vec![2, 1]);
        assert_eq!(stats.mean, vec![2.0, 4.0]);
    }

    #[test]
    fn merges_batches() {
        let first = fit(&[1.0, 2.0], 2, 1).unwrap();
        let second = fit(&[3.0, 4.0], 2, 1).unwrap();
        let merged = merge(&first.mean, &first.variance, &first.counts, &second).unwrap();
        let complete = fit(&[1.0, 2.0, 3.0, 4.0], 4, 1).unwrap();
        assert_eq!(merged.mean, complete.mean);
        assert_eq!(merged.variance, complete.variance);
        assert_eq!(merged.counts, complete.counts);
    }
}
