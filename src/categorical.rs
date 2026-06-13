use crate::error::CoreError;
use rustc_hash::FxHashMap;

fn decode_unicode_row(row: &[u32]) -> Result<String, CoreError> {
    row.iter()
        .take_while(|&&codepoint| codepoint != 0)
        .map(|&codepoint| char::from_u32(codepoint).ok_or(CoreError::InvalidUnicode(codepoint)))
        .collect()
}

pub fn discover_ordered<T: Ord + Copy>(values: &[T]) -> (Vec<T>, Vec<i64>) {
    let mut categories = values.to_vec();
    categories.sort();
    categories.dedup();
    let encoded = values
        .iter()
        .map(|value| categories.binary_search(value).unwrap_or_default() as i64)
        .collect();
    (categories, encoded)
}

pub fn encode_ordered<T: Ord + Copy>(values: &[T], categories: &[T]) -> Vec<i64> {
    values
        .iter()
        .map(|value| {
            categories
                .binary_search(value)
                .map_or(-1, |index| index as i64)
        })
        .collect()
}

pub fn discover_ordered_matrix<T: Ord + Copy>(
    values: &[T],
    rows: usize,
    columns: usize,
) -> Result<(Vec<Vec<T>>, Vec<i64>), CoreError> {
    if values.len() != rows * columns {
        return Err(CoreError::ShapeMismatch);
    }
    let mut categories = Vec::with_capacity(columns);
    let mut encoded = vec![0; values.len()];
    for column in 0..columns {
        let column_values: Vec<T> = values
            .chunks_exact(columns)
            .map(|row| row[column])
            .collect();
        let (feature_categories, feature_encoded) = discover_ordered(&column_values);
        for (row, code) in feature_encoded.into_iter().enumerate() {
            encoded[row * columns + column] = code;
        }
        categories.push(feature_categories);
    }
    Ok((categories, encoded))
}

pub fn discover_numeric(values: &[f64]) -> (Vec<f64>, Vec<i64>) {
    let mut categories: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| !value.is_nan())
        .collect();
    categories.sort_by(f64::total_cmp);
    categories.dedup_by(|left, right| left == right);
    let has_nan = values.iter().any(|value| value.is_nan());
    if has_nan {
        categories.push(f64::NAN);
    }
    let encoded = encode_numeric(values, &categories);
    (categories, encoded)
}

pub fn encode_numeric(values: &[f64], categories: &[f64]) -> Vec<i64> {
    let nan_index = categories
        .last()
        .filter(|value| value.is_nan())
        .map(|_| categories.len() as i64 - 1);
    let non_nan_categories = if nan_index.is_some() {
        &categories[..categories.len() - 1]
    } else {
        categories
    };
    values
        .iter()
        .map(|value| {
            if value.is_nan() {
                nan_index.unwrap_or(-1)
            } else {
                non_nan_categories
                    .binary_search_by(|candidate| candidate.total_cmp(value))
                    .map_or(-1, |index| index as i64)
            }
        })
        .collect()
}

pub fn discover_numeric_matrix(
    values: &[f64],
    rows: usize,
    columns: usize,
) -> Result<(Vec<Vec<f64>>, Vec<i64>), CoreError> {
    if values.len() != rows * columns {
        return Err(CoreError::ShapeMismatch);
    }
    let mut categories = Vec::with_capacity(columns);
    let mut encoded = vec![0; values.len()];
    for column in 0..columns {
        let column_values: Vec<f64> = values
            .chunks_exact(columns)
            .map(|row| row[column])
            .collect();
        let (feature_categories, feature_encoded) = discover_numeric(&column_values);
        for (row, code) in feature_encoded.into_iter().enumerate() {
            encoded[row * columns + column] = code;
        }
        categories.push(feature_categories);
    }
    Ok((categories, encoded))
}

pub fn discover_strings(values: &[String]) -> (Vec<String>, Vec<i64>) {
    let mut categories = values.to_vec();
    categories.sort();
    categories.dedup();
    let mapping: FxHashMap<&str, i64> = categories
        .iter()
        .enumerate()
        .map(|(index, category)| (category.as_str(), index as i64))
        .collect();
    let encoded = values.iter().map(|value| mapping[value.as_str()]).collect();
    (categories, encoded)
}

pub fn encode_strings(values: &[String], categories: &[String]) -> Vec<i64> {
    let mapping: FxHashMap<&str, i64> = categories
        .iter()
        .enumerate()
        .map(|(index, category)| (category.as_str(), index as i64))
        .collect();
    values
        .iter()
        .map(|value| mapping.get(value.as_str()).copied().unwrap_or(-1))
        .collect()
}

pub fn discover_unicode(
    codepoints: &[u32],
    rows: usize,
    width: usize,
) -> Result<(Vec<String>, Vec<i64>), CoreError> {
    if codepoints.len() != rows * width {
        return Err(CoreError::ShapeMismatch);
    }
    let values = codepoints
        .chunks_exact(width)
        .map(decode_unicode_row)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(discover_strings(&values))
}

pub fn discover_unicode_matrix(
    codepoints: &[u32],
    rows: usize,
    columns: usize,
    width: usize,
) -> Result<(Vec<Vec<String>>, Vec<i64>), CoreError> {
    if codepoints.len() != rows * columns * width {
        return Err(CoreError::ShapeMismatch);
    }
    let mut categories = Vec::with_capacity(columns);
    let mut encoded = vec![0; rows * columns];
    for column in 0..columns {
        let mut unique_rows: Vec<&[u32]> = (0..rows)
            .map(|row| {
                let start = (row * columns + column) * width;
                &codepoints[start..start + width]
            })
            .collect();
        unique_rows.sort();
        unique_rows.dedup();
        for row in 0..rows {
            let start = (row * columns + column) * width;
            encoded[row * columns + column] = unique_rows
                .binary_search(&&codepoints[start..start + width])
                .unwrap_or_default() as i64;
        }
        categories.push(
            unique_rows
                .iter()
                .map(|row| decode_unicode_row(row))
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    Ok((categories, encoded))
}

pub fn encode_unicode(
    codepoints: &[u32],
    rows: usize,
    width: usize,
    categories: &[String],
) -> Result<Vec<i64>, CoreError> {
    if codepoints.len() != rows * width {
        return Err(CoreError::ShapeMismatch);
    }
    let values = codepoints
        .chunks_exact(width)
        .map(decode_unicode_row)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(encode_strings(&values, categories))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_and_encodes_numeric_with_nan_last() {
        let (categories, encoded) = discover_numeric(&[2.0, f64::NAN, 1.0, 2.0]);
        assert_eq!(&categories[..2], &[1.0, 2.0]);
        assert!(categories[2].is_nan());
        assert_eq!(encoded, vec![1, 2, 0, 1]);
        assert_eq!(encode_numeric(&[3.0, f64::NAN], &categories), vec![-1, 2]);
    }

    #[test]
    fn discovers_and_marks_unknown_ordered_values() {
        let (categories, encoded) = discover_ordered(&[true, false, true]);
        assert_eq!(categories, vec![false, true]);
        assert_eq!(encoded, vec![1, 0, 1]);
        assert_eq!(encode_ordered(&[true], &[false]), vec![-1]);
    }

    #[test]
    fn discovers_and_marks_unknown_strings() {
        let values = vec!["東京".to_string(), "café".to_string()];
        let (categories, encoded) = discover_strings(&values);
        assert_eq!(categories, vec!["café".to_string(), "東京".to_string()]);
        assert_eq!(encoded, vec![1, 0]);
        assert_eq!(
            encode_strings(&["other".to_string()], &categories),
            vec![-1]
        );
    }

    #[test]
    fn discovers_fixed_width_unicode() {
        let (categories, encoded) =
            discover_unicode(&[26481, 20140, 0, 0, 99, 97, 102, 233], 2, 4).unwrap();
        assert_eq!(categories, vec!["café".to_string(), "東京".to_string()]);
        assert_eq!(encoded, vec![1, 0]);
    }

    #[test]
    fn discovers_ordered_matrix_by_feature() {
        let (categories, encoded) = discover_ordered_matrix(&[3, 1, 1, 2, 3, 1], 3, 2).unwrap();
        assert_eq!(categories, vec![vec![1, 3], vec![1, 2]]);
        assert_eq!(encoded, vec![1, 0, 0, 1, 1, 0]);
    }

    #[test]
    fn discovers_unicode_matrix_without_decoding_every_value() {
        let codepoints = [98, 0, 121, 0, 97, 0, 120, 0, 98, 0, 120, 0];
        let (categories, encoded) = discover_unicode_matrix(&codepoints, 3, 2, 2).unwrap();
        assert_eq!(
            categories,
            vec![
                vec!["a".to_string(), "b".to_string()],
                vec!["x".to_string(), "y".to_string()]
            ]
        );
        assert_eq!(encoded, vec![1, 1, 0, 0, 1, 0]);
    }
}
