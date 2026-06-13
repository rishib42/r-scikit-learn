use crate::error::CoreError;

pub fn validate_compressed<T>(
    indices: &[T],
    indptr: &[T],
    major_dimension: usize,
    minor_dimension: usize,
    nnz: usize,
) -> Result<(), CoreError>
where
    T: Copy + TryInto<usize>,
{
    if indices.len() != nnz || indptr.len() != major_dimension + 1 {
        return Err(CoreError::InvalidSparseStructure);
    }
    let mut previous = 0;
    for (position, &value) in indptr.iter().enumerate() {
        let pointer = value
            .try_into()
            .map_err(|_| CoreError::InvalidSparseStructure)?;
        if (position == 0 && pointer != 0) || pointer < previous || pointer > nnz {
            return Err(CoreError::InvalidSparseStructure);
        }
        previous = pointer;
    }
    if previous != nnz {
        return Err(CoreError::InvalidSparseStructure);
    }
    for value in indices {
        let index = (*value)
            .try_into()
            .map_err(|_| CoreError::InvalidSparseStructure)?;
        if index >= minor_dimension {
            return Err(CoreError::SparseIndexOutOfBounds(index, minor_dimension));
        }
    }
    Ok(())
}

pub fn scale_csr_columns_in_place<T, I>(
    values: &mut [T],
    indices: &[I],
    scale: &[T],
    inverse: bool,
) -> Result<(), CoreError>
where
    T: Copy + std::ops::Div<Output = T> + std::ops::Mul<Output = T>,
    I: Copy + TryInto<usize>,
{
    if values.len() != indices.len() {
        return Err(CoreError::InvalidSparseStructure);
    }
    for (value, &index) in values.iter_mut().zip(indices) {
        let index = index
            .try_into()
            .map_err(|_| CoreError::InvalidSparseStructure)?;
        let factor = scale
            .get(index)
            .ok_or(CoreError::SparseIndexOutOfBounds(index, scale.len()))?;
        *value = if inverse {
            *value * *factor
        } else {
            *value / *factor
        };
    }
    Ok(())
}

pub fn one_hot_csr(
    codes: &[i64],
    rows: usize,
    features: usize,
    widths: &[i64],
    drops: &[i64],
) -> Result<(Vec<i64>, Vec<i64>), CoreError> {
    if codes.len() != rows * features || widths.len() != features || drops.len() != features {
        return Err(CoreError::ShapeMismatch);
    }
    let mut offsets = Vec::with_capacity(features);
    let mut offset = 0_i64;
    for (&width, &drop) in widths.iter().zip(drops) {
        if width < 0 || drop >= width {
            return Err(CoreError::InvalidCode(drop));
        }
        offsets.push(offset);
        offset += width - i64::from(drop >= 0);
    }
    let mut indices = Vec::with_capacity(codes.len());
    let mut indptr = Vec::with_capacity(rows + 1);
    indptr.push(0);
    for row in codes.chunks_exact(features) {
        for (feature, &code) in row.iter().enumerate() {
            if code < 0 || code == drops[feature] {
                continue;
            }
            if code >= widths[feature] {
                return Err(CoreError::InvalidCode(code));
            }
            let shifted = code - i64::from(drops[feature] >= 0 && code > drops[feature]);
            indices.push(offsets[feature] + shifted);
        }
        indptr.push(indices.len() as i64);
    }
    Ok((indices, indptr))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_compressed_structure_and_bounds() {
        validate_compressed(&[1_i32, 0], &[0_i32, 1, 2], 2, 2, 2).unwrap();
        assert!(validate_compressed(&[2_i32], &[0_i32, 1, 1], 2, 2, 1).is_err());
        assert!(validate_compressed(&[0_i32], &[1_i32, 1, 1], 2, 2, 1).is_err());
    }

    #[test]
    fn scales_csr_values_by_column() {
        let mut values = vec![6.0, 8.0];
        scale_csr_columns_in_place(&mut values, &[1_i32, 0], &[2.0, 3.0], false).unwrap();
        assert_eq!(values, vec![2.0, 4.0]);
        scale_csr_columns_in_place(&mut values, &[1_i64, 0], &[2.0, 3.0], true).unwrap();
        assert_eq!(values, vec![6.0, 8.0]);
    }

    #[test]
    fn builds_one_hot_csr_with_unknowns_and_drops() {
        let (indices, indptr) = one_hot_csr(&[0, 1, 1, -1, 2, 0], 3, 2, &[3, 2], &[-1, 0]).unwrap();
        assert_eq!(indices, vec![0, 3, 1, 2]);
        assert_eq!(indptr, vec![0, 2, 3, 4]);
    }
}
