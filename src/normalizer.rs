use crate::error::CoreError;

#[derive(Debug, Clone, Copy)]
pub enum Norm {
    L1,
    L2,
    Max,
}

impl TryFrom<&str> for Norm {
    type Error = CoreError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "l1" => Ok(Self::L1),
            "l2" => Ok(Self::L2),
            "max" => Ok(Self::Max),
            _ => Err(CoreError::InvalidNorm(value.to_owned())),
        }
    }
}

pub trait NormalizeFloat:
    Copy
    + PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::DivAssign
    + std::ops::Mul<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;
    const EPSILON: Self;

    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn from_u8(value: u8) -> Self;
}

impl NormalizeFloat for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const EPSILON: Self = f32::EPSILON;

    fn abs(self) -> Self {
        self.abs()
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn from_u8(value: u8) -> Self {
        value as Self
    }
}

impl NormalizeFloat for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const EPSILON: Self = f64::EPSILON;

    fn abs(self) -> Self {
        self.abs()
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn from_u8(value: u8) -> Self {
        value as Self
    }
}

fn row_norm<T: NormalizeFloat>(row: &[T], norm: Norm) -> T {
    match norm {
        Norm::L1 => row
            .iter()
            .fold(T::ZERO, |total, &value| total + value.abs()),
        Norm::L2 => row
            .iter()
            .fold(T::ZERO, |total, &value| total + value * value)
            .sqrt(),
        Norm::Max => row.iter().fold(T::ZERO, |maximum, &value| {
            let absolute = value.abs();
            if absolute > maximum {
                absolute
            } else {
                maximum
            }
        }),
    }
}

pub fn transform<T: NormalizeFloat>(
    data: &[T],
    rows: usize,
    cols: usize,
    norm: Norm,
) -> Result<Vec<T>, CoreError> {
    if rows == 0 || cols == 0 {
        return Err(CoreError::EmptyInput);
    }
    if data.len() != rows * cols {
        return Err(CoreError::ShapeMismatch);
    }

    let mut output = data.to_vec();
    let minimum_norm = T::from_u8(10) * T::EPSILON;
    for row in output.chunks_exact_mut(cols) {
        let mut divisor = row_norm(row, norm);
        if divisor < minimum_norm {
            divisor = T::ONE;
        }
        for value in row {
            *value /= divisor;
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_all_supported_norms() {
        let data = [3.0_f64, 4.0, -3.0, 4.0];
        assert_eq!(
            transform(&data, 2, 2, Norm::L1).unwrap(),
            vec![3.0 / 7.0, 4.0 / 7.0, -3.0 / 7.0, 4.0 / 7.0]
        );
        assert_eq!(
            transform(&data, 2, 2, Norm::L2).unwrap(),
            vec![0.6, 0.8, -0.6, 0.8]
        );
        assert_eq!(
            transform(&data, 2, 2, Norm::Max).unwrap(),
            vec![0.75, 1.0, -0.75, 1.0]
        );
    }

    #[test]
    fn leaves_zero_and_tiny_rows_unchanged() {
        let data = [0.0_f64, 0.0, 1e-308, 1e-308];
        assert_eq!(transform(&data, 2, 2, Norm::L2).unwrap(), data);
    }

    #[test]
    fn supports_native_float32() {
        let output = transform(&[3.0_f32, 4.0], 1, 2, Norm::L2).unwrap();
        assert_eq!(output, vec![0.6_f32, 0.8]);
    }

    #[test]
    fn rejects_unknown_norms() {
        assert!(Norm::try_from("unknown").is_err());
    }
}
