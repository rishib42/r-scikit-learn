use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("input must contain at least one sample and one feature")]
    EmptyInput,
    #[error("input length does not match its shape")]
    ShapeMismatch,
    #[error("unknown label: {0}")]
    UnknownLabel(String),
    #[error("encoded label {0} is outside the valid range")]
    InvalidCode(i64),
    #[error("string labels contain an invalid Unicode code point: {0}")]
    InvalidUnicode(u32),
    #[error("unsupported normalization norm: {0}")]
    InvalidNorm(String),
    #[error("invalid quantile range: ({0}, {1})")]
    InvalidQuantileRange(f64, f64),
    #[error("unsupported imputation strategy: {0}")]
    InvalidImputationStrategy(String),
    #[error("input contains infinity")]
    InputContainsInfinity,
    #[error("input contains NaN, but the configured missing value is not NaN")]
    UnexpectedNaN,
    #[error("invalid compressed sparse matrix structure")]
    InvalidSparseStructure,
    #[error("sparse index {0} is outside dimension {1}")]
    SparseIndexOutOfBounds(usize, usize),
}

impl From<CoreError> for PyErr {
    fn from(error: CoreError) -> Self {
        PyValueError::new_err(error.to_string())
    }
}
