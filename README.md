# r-scikit-learn

Fast, familiar machine-learning building blocks powered by safe Rust. 🦀

`r-scikit-learn` combines a Rust computational core with lightweight,
scikit-learn-style Python estimators. Version 0.1.1 includes:

- Preprocessing, categorical encoding, and missing-value imputation
- Pipelines and column transformers
- Classification and regression metrics
- Dataset splitting and cross-validation
- Rust-powered linear models

This project is not affiliated with or endorsed by scikit-learn.

The installable distribution is named `r-scikit-learn`. Its Python import
package is `rsklearn`.

## Quick Start 🚀

After the first PyPI release, install with:

```bash
python -m pip install r-scikit-learn
```

Or build from source on macOS/Linux:

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

```python
from rsklearn.preprocessing import RobustScaler

X_robust = RobustScaler(quantile_range=(25.0, 75.0)).fit_transform(X)
```

```python
from rsklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1,
)
X_encoded = encoder.fit_transform([["small"], ["large"], ["small"]])
```

```python
from rsklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown="ignore")
X_one_hot = encoder.fit_transform([["small"], ["large"], ["small"]])
```

```python
from rsklearn.preprocessing import MaxAbsScaler, StandardScaler

X_sparse_scaled = StandardScaler(with_mean=False).fit_transform(X_one_hot)
X_sparse_maxabs = MaxAbsScaler().fit_transform(X_one_hot)
```

```python
import numpy as np
from rsklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median", add_indicator=True)
X_imputed = imputer.fit_transform([[1.0, np.nan], [3.0, 4.0]])
```

```python
from rsklearn.impute import SimpleImputer
from rsklearn.pipeline import make_pipeline
from rsklearn.preprocessing import StandardScaler

pipeline = make_pipeline(SimpleImputer(), StandardScaler())
X_prepared = pipeline.fit_transform([[1.0, np.nan], [3.0, 4.0]])
```

```python
from rsklearn.compose import ColumnTransformer
from rsklearn.impute import SimpleImputer
from rsklearn.pipeline import make_pipeline
from rsklearn.preprocessing import OneHotEncoder
from rsklearn.preprocessing import StandardScaler

preprocessor = ColumnTransformer(
    [
        ("numeric", make_pipeline(SimpleImputer(), StandardScaler()), ["age"]),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), ["city"]),
    ],
    remainder="drop",
)
X_prepared = preprocessor.fit_transform(table)
```

```python
from rsklearn.metrics import accuracy_score, mean_squared_error

accuracy = accuracy_score(y_true, y_pred)
mse = mean_squared_error(y_true_regression, y_pred_regression)
```

```python
from rsklearn.model_selection import cross_val_score, train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scores = cross_val_score(estimator, X, y, cv=5, scoring="accuracy")
```

```python
from rsklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge

regressor = Ridge(alpha=1.0).fit(X_train, y_train)
predictions = regressor.predict(X_test)
sparse_regressor = Lasso(alpha=0.1).fit(X_train, y_train)

classifier = LogisticRegression(max_iter=500).fit(X_train, class_labels)
probabilities = classifier.predict_proba(X_test)
```

## Highlights ✨

### Numeric Preprocessing

- Accepts non-empty 2D NumPy arrays and numeric array-like input.
- Uses float64 fitted statistics and native float32 kernels where supported.
- Ignores NaNs while fitting, preserves them while transforming, and rejects
  infinity.
- Supports incremental `partial_fit` for `StandardScaler`, `MaxAbsScaler`, and
  `MinMaxScaler`.
- Supports CSR/CSC sparse `StandardScaler(with_mean=False)` and `MaxAbsScaler`
  without densifying input.
- Supports L1, L2, and max row normalization.
- Provides quantile-based `RobustScaler` fitting and inverse transforms.

### Labels And Categories

- `LabelEncoder` supports integers, floats, booleans, and UTF-8 strings.
- `OrdinalEncoder` supports discovered or explicit categories, unknown values,
  missing values, and infrequent-category grouping.
- `OneHotEncoder` provides native Rust CSR construction, sparse or dense
  output, category dropping, inverse transforms, and feature names.
- Contiguous NumPy Unicode arrays use a fixed-width Rust codepoint pathway,
  avoiding per-label Python string conversion in the hot path.

### Imputation And Composition

- `SimpleImputer` supports dense numeric and categorical input, standard and
  callable strategies, missing indicators, inverse transforms, and feature
  names.
- Numeric imputation statistics and replacement use native Rust kernels.
- `Pipeline` and `make_pipeline` support nested parameters, passthrough steps,
  prediction, scoring, inverse transforms, and feature-name propagation.
- `ColumnTransformer` supports named or positional column selection, remainder
  estimators, transformer weights, and density-based dense or CSR output.

### Metrics And Model Selection

- Classification metrics: `accuracy_score`, `confusion_matrix`,
  `precision_score`, `recall_score`, and `f1_score`.
- Regression metrics: `mean_squared_error`, `mean_absolute_error`, and
  `r2_score`.
- Model selection: `train_test_split`, `KFold`, `StratifiedKFold`, and
  `cross_val_score`.
- Large reductions, weighted confusion matrices, and common split operations
  use safe Rust kernels.

### Linear Models

- Dense `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, and
  `LogisticRegression`.
- Optimized LAPACK least-squares fitting, Rust regularized solvers, and NumPy's
  BLAS path for dense prediction.
- Sample weights, intercepts, rank-deficient input, and multi-output
  regression.
- Shared Rust cyclic coordinate descent for `Lasso` and `ElasticNet`.
- Binary Rust logistic solvers and BLAS-backed multiclass L-BFGS, including
  binary L1 and elastic-net fitting.

### Estimator And Sparse Foundations

Public estimator-author APIs are available from `rsklearn.base` and
`rsklearn.utils.validation`. They include `BaseEstimator`, `TransformerMixin`,
`ClassifierMixin`, `RegressorMixin`, `clone`, `check_array`, `check_X_y`,
`check_is_fitted`, and `validate_data`. Scalers use these APIs for fitted-state,
feature-count, and string feature-name validation. The numeric preprocessors
pass scikit-learn's official estimator checks.

Shared sparse infrastructure is available from `rsklearn.utils`. It validates
and converts SciPy sparse formats, exposes canonical CSR/CSC components to safe
Rust kernels, reconstructs validated sparse output, and provides native
float32/float64 sparse column scaling. Existing estimators remain dense-only
until their sparse-specific behavior is implemented.

For `StandardScaler`, `mean_` follows scikit-learn's practical behavior: it is
available when either centering or standard-deviation scaling needs it, and is
`None` only when both options are disabled. `var_` and `scale_` are `None`
when `with_std=False`.

## Compatibility

The supported behavior is differential-tested against scikit-learn, including
population variance, constant features, non-default feature ranges, clipping,
round trips, and sorted label classes. `r-scikit-learn` is intentionally much
smaller and does not yet claim complete estimator API compatibility.

## Current Production Gaps 🛠️

The core implemented behavior is tested and packaged across Linux, macOS, and
Windows, but the project remains alpha software. Before a stable 1.0 release,
the following compatibility and operational work remains:

- `sample_weight` support for `StandardScaler.partial_fit`.
- Comprehensive `get_feature_names_out` support and configurable output
  containers across estimators.
- Estimator-check compliance for future classifier and regressor types.
- Broader `copy=False` support and native float32 Rust kernels for scalers.
- Further multiclass logistic solver optimization and broader parallel-kernel
  tuning.
- Broader fuzz, property, memory-pressure, and long-running benchmark coverage.

## Benchmarks ⚡

Performance depends on workload, hardware, input layout, and build mode. Run
the benchmarks locally:

```bash
maturin develop --release
python benches/benchmark_preprocessing.py
python benches/benchmark_preprocessing.py --include-largest
python benches/benchmark_metrics.py
python benches/benchmark_linear_models.py
```

The benchmark warms up each operation and reports multiple repetitions for
fit, transform, and end-to-end calls. Public `r-scikit-learn` timings include
Python-side validation and any required contiguous float64 conversion.
Performance benchmarks must use a release Rust extension. `maturin develop`
without `--release` intentionally builds an unoptimized debug extension for
development and can be tens of times slower.

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

1. Update the matching versions in `pyproject.toml`, `Cargo.toml`, and
   `python/rsklearn/__init__.py`, then update `CHANGELOG.md`.
2. Push the release commit and wait for CI, including manylinux and sdist
   installation checks, to pass.
3. Run the manual TestPyPI workflow and verify its distributions.
4. Run the manual Release workflow with the version number without a `v`
   prefix.
5. Approve the PyPI environment if required.

The release workflow refuses existing versions, installs every wheel on
Python 3.10-3.13 across Linux, macOS, and Windows, verifies sdist installation,
publishes through PyPI Trusted Publishing, creates the immutable GitHub tag and
release, attaches artifacts, and verifies installation from PyPI. No API token
is stored in the repository. Configure separate `pypi` and `testpypi` GitHub
environments and matching Trusted Publishers for `release.yml` and
`test-pypi.yml`, respectively.

## Roadmap

- Close the remaining production gaps listed above.
- Add sparse-aware behavior to compatible existing estimators.
- Add further categorical encoding and discretization estimators.
- Publish reproducible benchmark reports from release wheels.

## License

MIT
