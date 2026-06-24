# Changelog

All notable changes to r-scikit-learn are documented here. Release tags and
published package versions are immutable.

## Unreleased

## 0.1.3 - 2026-06-24

- Added dense brute-force `KNeighborsRegressor` with Rust-backed single-output
  and multi-output prediction.
- Optimized dense Euclidean nearest-neighbor search with a lazily cached
  transposed fit matrix and operation-specific block sizes.

## 0.1.2 - 2026-06-24

- Added dense brute-force `KNeighborsClassifier` with Rust-backed neighbor
  search, class voting, `predict`, `predict_proba`, and `kneighbors`.
- Added scikit-learn parity tests and benchmarks for nearest-neighbor
  classification.
- Optimized the dense Euclidean neighbor search path with blocked dot products,
  reusable work buffers, and macOS Accelerate/CBLAS acceleration with a portable
  `matrixmultiply` fallback.
- Added sparse `StandardScaler(with_mean=False)` and `MaxAbsScaler` with
  Rust-backed CSR/CSC reductions and column scaling.

## 0.1.1 - 2026-06-15

- Added wheel and source-distribution installation testing across supported
  operating systems and Python versions.
- Added a numerical-safety fallback for ill-conditioned tall least-squares
  problems.
- Added TestPyPI, cross-platform benchmark, and immutable manual release
  workflows.

## 0.1.0

- Added Rust-powered preprocessing, categorical encoding, sparse
  infrastructure, composition, metrics, model selection, and linear models.
- Added Linux, macOS, and Windows wheel builds for Python 3.10 through 3.13.
- Added Rust-native tall-matrix least squares and multinomial logistic
  optimization.
