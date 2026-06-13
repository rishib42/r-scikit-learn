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
}

impl From<CoreError> for PyErr {
    fn from(error: CoreError) -> Self {
        PyValueError::new_err(error.to_string())
    }
}
