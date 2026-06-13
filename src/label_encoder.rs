use crate::error::CoreError;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::BTreeMap;

fn decode_unicode_row(row: &[u32]) -> Result<String, CoreError> {
    row.iter()
        .take_while(|&&codepoint| codepoint != 0)
        .map(|&codepoint| char::from_u32(codepoint).ok_or(CoreError::InvalidUnicode(codepoint)))
        .collect()
}

pub fn fit_transform_unicode(
    codepoints: &[u32],
    rows: usize,
    width: usize,
) -> Result<(Vec<String>, Vec<i64>), CoreError> {
    if codepoints.len() != rows * width {
        return Err(CoreError::ShapeMismatch);
    }
    let mut unique = FxHashSet::default();
    for row in codepoints.chunks_exact(width) {
        unique.insert(decode_unicode_row(row)?);
    }
    let mut classes: Vec<String> = unique.into_iter().collect();
    classes.sort();
    let mapping: FxHashMap<&str, i64> = classes
        .iter()
        .enumerate()
        .map(|(index, value)| (value.as_str(), index as i64))
        .collect();
    let encoded = codepoints
        .chunks_exact(width)
        .map(|row| {
            let value = decode_unicode_row(row)?;
            mapping
                .get(value.as_str())
                .copied()
                .ok_or(CoreError::UnknownLabel(value))
        })
        .collect::<Result<_, _>>()?;
    Ok((classes, encoded))
}

pub fn transform_unicode(
    codepoints: &[u32],
    rows: usize,
    width: usize,
    classes: &[String],
) -> Result<Vec<i64>, CoreError> {
    if codepoints.len() != rows * width {
        return Err(CoreError::ShapeMismatch);
    }
    let mapping: FxHashMap<&str, i64> = classes
        .iter()
        .enumerate()
        .map(|(index, value)| (value.as_str(), index as i64))
        .collect();
    codepoints
        .chunks_exact(width)
        .map(|row| {
            let value = decode_unicode_row(row)?;
            mapping
                .get(value.as_str())
                .copied()
                .ok_or(CoreError::UnknownLabel(value))
        })
        .collect()
}

pub fn fit_transform_numeric(values: &[f64]) -> (Vec<f64>, Vec<i64>) {
    let mut classes = values.to_vec();
    classes.sort_by(f64::total_cmp);
    classes.dedup_by(|left, right| left.total_cmp(right).is_eq());
    let encoded = values
        .iter()
        .map(|value| classes.partition_point(|candidate| candidate.total_cmp(value).is_lt()) as i64)
        .collect();
    (classes, encoded)
}

pub fn fit_transform_ordered<T: Ord + Copy>(values: &[T]) -> (Vec<T>, Vec<i64>) {
    let mut classes = values.to_vec();
    classes.sort();
    classes.dedup();
    let encoded = values
        .iter()
        .map(|value| classes.partition_point(|candidate| candidate < value) as i64)
        .collect();
    (classes, encoded)
}

pub fn transform_ordered<T: Ord + Copy + ToString>(
    values: &[T],
    classes: &[T],
) -> Result<Vec<i64>, CoreError> {
    values
        .iter()
        .map(|value| {
            classes
                .binary_search(value)
                .map(|index| index as i64)
                .map_err(|_| CoreError::UnknownLabel(value.to_string()))
        })
        .collect()
}

pub fn inverse_ordered<T: Copy>(codes: &[i64], classes: &[T]) -> Result<Vec<T>, CoreError> {
    codes
        .iter()
        .map(|&code| {
            usize::try_from(code)
                .ok()
                .and_then(|index| classes.get(index).copied())
                .ok_or(CoreError::InvalidCode(code))
        })
        .collect()
}

pub fn transform_numeric(values: &[f64], classes: &[f64]) -> Result<Vec<i64>, CoreError> {
    values
        .iter()
        .map(|value| {
            classes
                .binary_search_by(|candidate| candidate.total_cmp(value))
                .map(|index| index as i64)
                .map_err(|_| CoreError::UnknownLabel(value.to_string()))
        })
        .collect()
}

pub fn fit_transform_strings(values: &[String]) -> (Vec<String>, Vec<i64>) {
    let mut classes = values.to_vec();
    classes.sort();
    classes.dedup();
    let mapping: BTreeMap<&str, i64> = classes
        .iter()
        .enumerate()
        .map(|(index, value)| (value.as_str(), index as i64))
        .collect();
    let encoded = values.iter().map(|value| mapping[value.as_str()]).collect();
    (classes, encoded)
}

pub fn transform_strings(values: &[String], classes: &[String]) -> Result<Vec<i64>, CoreError> {
    values
        .iter()
        .map(|value| {
            classes
                .binary_search(value)
                .map(|index| index as i64)
                .map_err(|_| CoreError::UnknownLabel(value.clone()))
        })
        .collect()
}

pub fn inverse_numeric(codes: &[i64], classes: &[f64]) -> Result<Vec<f64>, CoreError> {
    codes
        .iter()
        .map(|&code| {
            usize::try_from(code)
                .ok()
                .and_then(|index| classes.get(index).copied())
                .ok_or(CoreError::InvalidCode(code))
        })
        .collect()
}

pub fn inverse_strings(codes: &[i64], classes: &[String]) -> Result<Vec<String>, CoreError> {
    codes
        .iter()
        .map(|&code| {
            usize::try_from(code)
                .ok()
                .and_then(|index| classes.get(index).cloned())
                .ok_or(CoreError::InvalidCode(code))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorts_and_encodes_unicode() {
        let values = vec!["東京".to_string(), "café".to_string(), "東京".to_string()];
        let (classes, encoded) = fit_transform_strings(&values);
        assert_eq!(classes, vec!["café".to_string(), "東京".to_string()]);
        assert_eq!(encoded, vec![1, 0, 1]);
    }

    #[test]
    fn rejects_unknown_numeric_label() {
        assert!(transform_numeric(&[3.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn preserves_large_ordered_integer_labels() {
        let values = [9_007_199_254_740_993_i64, 9_007_199_254_740_992_i64];
        let (classes, encoded) = fit_transform_ordered(&values);
        assert_eq!(
            classes,
            vec![9_007_199_254_740_992_i64, 9_007_199_254_740_993_i64]
        );
        assert_eq!(encoded, vec![1, 0]);
    }

    #[test]
    fn decodes_fixed_width_unicode_row() {
        assert_eq!(
            decode_unicode_row(&[99, 97, 102, 233]).unwrap(),
            "café".to_string()
        );
    }

    #[test]
    fn encodes_fixed_width_unicode_without_materializing_all_labels() {
        let (classes, encoded) = fit_transform_unicode(
            &[26481, 20140, 0, 0, 99, 97, 102, 233, 26481, 20140, 0, 0],
            3,
            4,
        )
        .unwrap();
        assert_eq!(classes, vec!["café".to_string(), "東京".to_string()]);
        assert_eq!(encoded, vec![1, 0, 1]);
    }
}
