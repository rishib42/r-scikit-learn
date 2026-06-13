# r-scikit-learn

`r-scikit-learn` provides scikit-learn-style preprocessing with a safe Rust
computational core and lightweight Python estimator classes. Version 0.1.0
provides `StandardScaler`, `MinMaxScaler`, `Normalizer`, and `LabelEncoder`.

This project is not affiliated with or endorsed by scikit-learn.

The installable distribution is named `r-scikit-learn`. Its Python import
package remains `rsklearn` because Python module names cannot contain hyphens.

## Installation

Published wheels, once available:

```bash
python -m pip install r-scikit-learn
```

Before publishing, verify that the `r-scikit-learn` distribution name is
available on PyPI.

Build from source on macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip maturin
maturin develop
pytest
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Building
requires a stable Rust toolchain and Python 3.10 or newer.

## Usage

```python
import numpy as np
from rsklearn.preprocessing import StandardScaler

X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_original = scaler.inverse_transform(X_scaled)
```

```python
from rsklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1.0, 1.0), clip=True)
X_scaled = scaler.fit_transform([[1, 10], [2, 20], [3, 30]])
```

```python
from rsklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoded = encoder.fit_transform(["café", "東京", "café"])
labels = encoder.inverse_transform(encoded)
```

```python
from rsklearn.preprocessing import Normalizer

X_normalized = Normalizer(norm="l2").fit_transform([[3.0, 4.0], [0.0, 0.0]])
```

## Supported Inputs

Numeric preprocessors accept non-empty two-dimensional NumPy arrays and
array-like numeric input. Inputs are converted to contiguous float64 arrays for
the Rust core. Transforming float32 input returns float32 output; other numeric
input returns float64 output. NaNs are ignored while fitting and preserved
while transforming. Infinity is rejected. Inputs are not mutated.

`StandardScaler` and `MinMaxScaler` support incremental updates through
`partial_fit`. `Normalizer` supports L1, L2, and max row normalization with
native float32 and float64 Rust kernels. Its `copy=False` behavior is
best-effort, matching scikit-learn's documented contract.

`LabelEncoder` accepts one-dimensional signed integer, unsigned integer,
floating-point, boolean, or UTF-8 string labels. Empty labels, NaN, infinity,
and values across the full int64/uint64 ranges are supported. Integer class
values preserve their input dtype.

Public estimator-author APIs are available from `rsklearn.base` and
`rsklearn.utils.validation`. They include `BaseEstimator`, `TransformerMixin`,
`ClassifierMixin`, `RegressorMixin`, `clone`, `check_array`, `check_X_y`,
`check_is_fitted`, and `validate_data`. Scalers use these APIs for fitted-state,
feature-count, and string feature-name validation. The numeric preprocessors
pass scikit-learn's official estimator checks.

Contiguous NumPy Unicode arrays are exposed to safe Rust as fixed-width
codepoint rows, avoiding per-label Python string conversion in the hot path.

For `StandardScaler`, `mean_` follows scikit-learn's practical behavior: it is
available when either centering or standard-deviation scaling needs it, and is
`None` only when both options are disabled. `var_` and `scale_` are `None`
when `with_std=False`.

## Comparison With scikit-learn

The supported behavior is differential-tested against scikit-learn, including
population variance, constant features, non-default feature ranges, clipping,
round trips, and sorted label classes. `r-scikit-learn` is intentionally much
smaller and does not yet claim complete estimator API compatibility.

## Current Production Gaps

The core implemented behavior is tested and packaged across Linux, macOS, and
Windows, but the project remains alpha software. Before a stable 1.0 release,
the following compatibility and operational work remains:

- Sparse matrix support, including non-centering `StandardScaler` operation.
- `sample_weight` support for `StandardScaler.partial_fit`.
- `get_feature_names_out` and configurable output containers.
- Estimator-check compliance for future classifier and regressor types.
- Broader `copy=False` support and native float32 Rust kernels for scalers.
- Broader fuzz, property, memory-pressure, and long-running benchmark coverage.

No performance claim is made. Run:

```bash
python benches/benchmark_preprocessing.py
python benches/benchmark_preprocessing.py --include-largest
```

The benchmark warms up each operation and reports multiple repetitions for
fit, transform, and end-to-end calls. Public `r-scikit-learn` timings include
Python-side validation and any required contiguous float64 conversion.

## Development

```bash
maturin develop --extras dev
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
ruff format --check python tests benches
ruff check python tests benches
pytest
maturin build --release
maturin sdist --out dist
python -c "from rsklearn.preprocessing import StandardScaler"
```

The Rust binding accepts contiguous NumPy arrays through `rust-numpy`. Public
Python validation may copy non-contiguous or non-float64 input. Rust produces
new owned output arrays so transformations never mutate caller input.
Substantial numerical loops release the Python GIL.

## Release

1. Run all development checks and build a release wheel.
2. Install the wheel into a clean virtual environment and run the import smoke
   test.
3. Verify the distribution name on PyPI.
4. Tag the release as `v0.1.0` and push the tag.
5. Approve the GitHub Actions Trusted Publishing environment.

The release workflow uses PyPI Trusted Publishing and contains no API token.

## Roadmap

- Close the remaining production gaps listed above.
- Add robust scaling, encoding, and discretization estimators.
- Publish reproducible benchmark reports from release wheels.

## License

MIT
