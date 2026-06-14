use crate::error::CoreError;

#[derive(Clone, Copy)]
pub enum Strategy {
    Mean,
    Median,
    MostFrequent,
}

#[derive(Debug, Clone)]
pub struct MeanStats {
    pub statistics: Vec<f64>,
    pub missing_features: Vec<i64>,
    pub empty_features: Vec<i64>,
}

impl TryFrom<&str> for Strategy {
    type Error = CoreError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "mean" => Ok(Self::Mean),
            "median" => Ok(Self::Median),
            "most_frequent" => Ok(Self::MostFrequent),
            _ => Err(CoreError::InvalidImputationStrategy(value.to_owned())),
        }
    }
}

fn is_missing(value: f64, missing_value: f64, missing_is_nan: bool) -> bool {
    if missing_is_nan {
        value.is_nan()
    } else {
        value == missing_value
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn most_frequent(values: &mut [f64]) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    let mut best = values[0];
    let mut best_count = 0;
    let mut index = 0;
    while index < values.len() {
        let value = values[index];
        let mut end = index + 1;
        while end < values.len() && values[end] == value {
            end += 1;
        }
        let count = end - index;
        if count > best_count {
            best = value;
            best_count = count;
        }
        index = end;
    }
    best
}

pub fn fit_mean(
    data: &[f64],
    rows: usize,
    cols: usize,
    missing_value: f64,
    missing_is_nan: bool,
) -> Result<MeanStats, CoreError> {
    if rows == 0 || cols == 0 {
        return Err(CoreError::EmptyInput);
    }
    if data.len() != rows * cols {
        return Err(CoreError::ShapeMismatch);
    }
    let mut sums = vec![0.0; cols];
    let mut counts = vec![0_usize; cols];
    let mut has_missing = vec![false; cols];
    for row in data.chunks_exact(cols) {
        for (column, &value) in row.iter().enumerate() {
            if value.is_infinite() {
                return Err(CoreError::InputContainsInfinity);
            }
            if is_missing(value, missing_value, missing_is_nan) {
                has_missing[column] = true;
            } else {
                if value.is_nan() {
                    return Err(CoreError::UnexpectedNaN);
                }
                sums[column] += value;
                counts[column] += 1;
            }
        }
    }
    let statistics = sums
        .into_iter()
        .zip(&counts)
        .map(|(sum, &count)| {
            if count == 0 {
                f64::NAN
            } else {
                sum / count as f64
            }
        })
        .collect();
    let missing_features = has_missing
        .iter()
        .enumerate()
        .filter_map(|(column, &missing)| missing.then_some(column as i64))
        .collect();
    let empty_features = counts
        .iter()
        .enumerate()
        .filter_map(|(column, &count)| (count == 0).then_some(column as i64))
        .collect();
    Ok(MeanStats {
        statistics,
        missing_features,
        empty_features,
    })
}

pub fn fit(
    data: &[f64],
    rows: usize,
    cols: usize,
    strategy: Strategy,
    missing_value: f64,
    missing_is_nan: bool,
) -> Result<Vec<f64>, CoreError> {
    if rows == 0 || cols == 0 {
        return Err(CoreError::EmptyInput);
    }
    if data.len() != rows * cols {
        return Err(CoreError::ShapeMismatch);
    }
    let mut statistics = Vec::with_capacity(cols);
    let mut values = Vec::with_capacity(rows);
    for column in 0..cols {
        values.clear();
        values.extend(
            data.iter()
                .skip(column)
                .step_by(cols)
                .copied()
                .filter(|value| !is_missing(*value, missing_value, missing_is_nan)),
        );
        let statistic = if values.is_empty() {
            f64::NAN
        } else {
            match strategy {
                Strategy::Mean => values.iter().sum::<f64>() / values.len() as f64,
                Strategy::Median => median(&mut values),
                Strategy::MostFrequent => most_frequent(&mut values),
            }
        };
        statistics.push(statistic);
    }
    Ok(statistics)
}

pub fn transform(
    data: &[f64],
    rows: usize,
    cols: usize,
    statistics: &[f64],
    retained: &[i64],
    missing_value: f64,
    missing_is_nan: bool,
) -> Result<Vec<f64>, CoreError> {
    if data.len() != rows * cols || statistics.len() != cols {
        return Err(CoreError::ShapeMismatch);
    }
    let retained: Vec<usize> = retained
        .iter()
        .map(|&value| usize::try_from(value).map_err(|_| CoreError::ShapeMismatch))
        .collect::<Result<_, _>>()?;
    if retained.iter().any(|&column| column >= cols) {
        return Err(CoreError::ShapeMismatch);
    }
    let mut output = Vec::with_capacity(rows * retained.len());
    for row in data.chunks_exact(cols) {
        for &column in &retained {
            let value = row[column];
            output.push(if is_missing(value, missing_value, missing_is_nan) {
                statistics[column]
            } else {
                value
            });
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_standard_statistics_and_tie_breaks_to_smallest() {
        let data = [1.0, 3.0, f64::NAN, 1.0, 5.0, 2.0, 3.0, 2.0];
        assert_eq!(
            fit(&data, 4, 2, Strategy::Mean, f64::NAN, true).unwrap(),
            vec![3.0, 2.0]
        );
        assert_eq!(
            fit(&data, 4, 2, Strategy::Median, f64::NAN, true).unwrap(),
            vec![3.0, 2.0]
        );
        assert_eq!(
            fit(&data, 4, 2, Strategy::MostFrequent, f64::NAN, true).unwrap(),
            vec![1.0, 2.0]
        );
    }

    #[test]
    fn fused_mean_fit_returns_compact_missing_and_empty_metadata() {
        let stats = fit_mean(
            &[1.0, f64::NAN, 3.0, f64::NAN, f64::NAN, 5.0],
            2,
            3,
            f64::NAN,
            true,
        )
        .unwrap();
        assert_eq!(stats.statistics[0], 1.0);
        assert!(stats.statistics[1].is_nan());
        assert_eq!(stats.statistics[2], 4.0);
        assert_eq!(stats.missing_features, vec![0, 1]);
        assert_eq!(stats.empty_features, vec![1]);
    }

    #[test]
    fn fused_mean_fit_rejects_unexpected_non_finite_values() {
        assert!(matches!(
            fit_mean(&[f64::INFINITY], 1, 1, f64::NAN, true),
            Err(CoreError::InputContainsInfinity)
        ));
        assert!(matches!(
            fit_mean(&[f64::NAN], 1, 1, -1.0, false),
            Err(CoreError::UnexpectedNaN)
        ));
    }

    #[test]
    fn transforms_and_can_drop_empty_features() {
        let data = [1.0, f64::NAN, f64::NAN, f64::NAN];
        let output = transform(&data, 2, 2, &[1.0, 0.0], &[0], f64::NAN, true).unwrap();
        assert_eq!(output, vec![1.0, 1.0]);
    }
}
