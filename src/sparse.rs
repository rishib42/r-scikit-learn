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
}
