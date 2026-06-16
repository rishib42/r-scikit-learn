use crate::error::CoreError;

pub fn fit(data: &[f64], rows: usize, cols: usize) -> Result<Vec<f64>, CoreError> {
    if rows == 0 || cols == 0 {
        return Err(CoreError::EmptyInput);
    }
    if data.len() != rows * cols {
        return Err(CoreError::ShapeMismatch);
    }
    let mut max_abs = vec![0.0_f64; cols];
    for row in data.chunks_exact(cols) {
        for (column, &value) in row.iter().enumerate() {
            if !value.is_nan() {
                max_abs[column] = max_abs[column].max(value.abs());
            }
        }
    }
    Ok(max_abs)
}

pub fn transform_f64(
    data: &[f64],
    rows: usize,
    cols: usize,
    scale: &[f64],
    inverse: bool,
) -> Result<Vec<f64>, CoreError> {
    if data.len() != rows * cols || scale.len() != cols {
        return Err(CoreError::ShapeMismatch);
    }
    let mut output = data.to_vec();
    for row in output.chunks_exact_mut(cols) {
        for (column, value) in row.iter_mut().enumerate() {
            if inverse {
                *value *= scale[column];
            } else {
                *value /= scale[column];
            }
        }
    }
    Ok(output)
}

pub fn transform_f32(
    data: &[f32],
    rows: usize,
    cols: usize,
    scale: &[f64],
    inverse: bool,
) -> Result<Vec<f32>, CoreError> {
    if data.len() != rows * cols || scale.len() != cols {
        return Err(CoreError::ShapeMismatch);
    }
    let mut output = data.to_vec();
    for row in output.chunks_exact_mut(cols) {
        for (column, value) in row.iter_mut().enumerate() {
            let factor = scale[column] as f32;
            if inverse {
                *value *= factor;
            } else {
                *value /= factor;
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_max_abs_without_materializing_absolute_matrix() {
        let values = [0.0, -2.0, f64::NAN, 4.0, 0.0, 0.0];
        assert_eq!(fit(&values, 2, 3).unwrap(), vec![4.0, 2.0, 0.0]);
    }

    #[test]
    fn transforms_and_roundtrips() {
        let values = [0.0, -2.0, 4.0, 0.0];
        let scale = [4.0, 2.0];
        let scaled = transform_f64(&values, 2, 2, &scale, false).unwrap();
        assert_eq!(scaled, vec![0.0, -1.0, 1.0, 0.0]);
        assert_eq!(transform_f64(&scaled, 2, 2, &scale, true).unwrap(), values);
    }
}
