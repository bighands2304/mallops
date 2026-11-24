# `std/train.smartcore`

Native classic machine-learning training block in Rust backed by [`smartcore`](https://smartcorelib.org/)
`0.7` (pure Rust, Apache-2.0, no BLAS). Behind the optional `train-smartcore` Cargo feature of
`mallops-std` (not in `default`).
See [SPEC FR-TRN-1](../SPEC.md#34-training).

```yaml
blocks:
  model:
    use: std/train.smartcore
    in: { train: features.train }
    with:
      algorithm: logistic_regression
      target: churned
      seed: 42
      hyperparameters:
        alpha: 1.0
```

## Configuration (`with:`)

Unknown fields are rejected (`deny_unknown_fields`). `validate_config` uses the same parser as
`run`, so `mallops validate` catches configuration errors before execution.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `algorithm` | `logistic_regression` \| `linear_regression` | required | Training algorithm (`snake_case`). |
| `target` | non-empty string | none | Target column name. Falls back to `metadata["target"]` of the `train` input artifact if omitted; if both are set and differ, the run fails. |
| `seed` | `u64` | none | Optional seed recorded in the output manifest `metadata`. Neither v0.1 algorithm uses randomness; distinct from the node-level `seed:` ([SPEC §5](../SPEC.md#5-pipeline-definition-mallopsyaml)). |
| `hyperparameters` | object | `{}` | Algorithm-specific parameters (deserialized after `algorithm`; unknown keys rejected). |

### `hyperparameters` by algorithm

**`logistic_regression`** (binary classification, fixed LBFGS solver):

| Key | Type | Default | Constraints |
|---|---|---|---|
| `alpha` | `f64` | `1.0` | Finite number `>= 0.0`. |

**`linear_regression`** (ordinary least squares):

| Key | Type | Default | Constraints |
|---|---|---|---|
| `solver` | `svd` \| `qr` | `svd` | Matrix factorization solver. |

Validation rules:

- Using a `logistic_regression` hyperparameter key with `linear_regression`, or vice versa, is an
  error.
- Unknown `algorithm` or `solver` values, negative or non-finite `alpha`, and empty `target` are
  rejected.
- Strings containing `${secret:` anywhere in the config are rejected (as in
  [`std/train.external`](train-external.md)).

## Ports

- **Input:** `train` (required), `kind: features`, `format: parquet`.
- **Output:** `model`, `kind: model`, `format: json`.
  - The static `BlockSpec` advertises both flavors: `smartcore-logistic` and `smartcore-linear`.
  - `resolve_ports` narrows the output flavor to the single flavor matching `with.algorithm`:
    `smartcore-logistic` for `logistic_regression` and `smartcore-linear` for `linear_regression`.

## Input data rules

- **Target resolution:** uses `with.target` when set, otherwise `metadata["target"]` (string) from
  the input artifact manifest. Missing in both is an error; **present in both with different values
  is an error** (`` with.target `<x>` differs from the input's metadata target `<y>` ``). The target
  column must exist in the Parquet schema.
- **Feature columns:** every column other than the target, in the input schema's order. Allowed
  Arrow types are `Boolean`, `Int64`, and `Float64` (cast to `f64`). Any other type fails with an
  error suggesting preprocessing with `encode` or `select`.
- **Values and shape:**
  - Null values fail with an error naming the column.
  - Non-finite values (`NaN`, `±inf`) fail with an error naming the column and row.
  - Empty datasets (`0` rows) or datasets with no feature columns fail.
  - `linear_regression` requires `rows >= features + 1`.
- **Target types:**
  - `logistic_regression` (binary only): `Int64`, `Boolean`, or `Utf8` (`Float64` is rejected).
    Distinct values are sorted (`Int64` numerically, `false < true` for `Boolean`, byte order for
    `Utf8`). Anything other than **2** distinct classes fails. Internally the model fits on `0`/`1`
    labels (`0 = classes[0]`, `1 = classes[1]`), so positive coefficients correspond to
    `classes[1]`.
  - `linear_regression`: `Int64` or `Float64` (cast to `f64`).
- **Post-fit checks and cancellation:** non-finite intercepts or coefficients fail after fitting.
  A singular-matrix error from `smartcore` reports
  `"features are linearly dependent (e.g. one_hot with all categories); drop a column"`.
  Cooperative cancellation (`ctx.cancel()`) is checked at run start, after fetching inputs, during
  Parquet reading, before and after fitting, and before writing the output; the fit itself is not
  interruptible.

## Model format (`model.json`)

The output artifact payload is a single file, `model.json`, serialized from a struct with fixed
field order (no `HashMap`) and `serde_json` float formatting:

```json
{
  "v": 1,
  "algorithm": "logistic_regression",
  "features": ["age", "plan_a"],
  "target": "churned",
  "intercept": -2.71,
  "coefficients": [1.11, -0.13],
  "classes": [false, true],
  "threshold": 0.5
}
```

- `classes` and `threshold` are included **only** for `logistic_regression` and omitted for
  `linear_regression`.
- `classes` keeps the JSON type of the target column: `[0, 1]` (`int64`), `[false, true]` (`bool`),
  or `["no", "yes"]` (`string`), sorted ascending.

### Prediction rule

- **`logistic_regression`:** compute `p = 1 / (1 + exp(-(intercept + Σ coefficients[i] * x[i])))`;
  predict `classes[1]` when `p >= threshold`, otherwise `classes[0]`.
- **`linear_regression`:** compute `y = intercept + Σ coefficients[i] * x[i]`.

## Manifest and `io_schema`

Output manifest fields:

- `kind`: `model`
- `format`: `json`
- `flavor`: `smartcore-logistic` or `smartcore-linear` (pure data formats; do not execute code on
  load, allowed by default in `mallops serve`)
- `metadata`:
  - `algorithm`: `"logistic_regression"` or `"linear_regression"`
  - `hyperparameters`: **effective** hyperparameters including defaults (e.g. `{"alpha": 1.0}` or
    `{"solver": "svd"}`)
  - `target`: target column name
  - `seed`: present only when `with.seed` is configured
  - `framework`: `"smartcore"`
  - `framework_version`: `"0.7"`
  - No timestamps, run IDs, or block-written `parents` (the runtime injects `parents` on commit).

`io_schema` uses the same shape as the `transform` artifact from `std/preprocess.tabular`:

```json
{
  "input": [
    { "name": "age", "type": "int64" },
    { "name": "plan_a", "type": "float64" }
  ],
  "output": [
    { "name": "churned", "type": "bool" }
  ]
}
```

- `input`: feature columns with their **original** input types (`bool`, `int64`, or `float64`), in
  schema order.
- `output`: single entry for the target column — original target type (`int64`, `bool`, or `string`)
  for `logistic_regression`, and `float64` for `linear_regression`.

## Cache

`CachePolicy::Deterministic { extra_hash: None }`.

- Same inputs, config, node `seed:`, and block version hit the cache and produce the same
  `ArtifactId` on the same machine.
- Changing `with.seed` changes the manifest metadata and cache key, producing a different
  `ArtifactId`.
- Upgrading `smartcore` or changing the `model.json` format requires bumping `BlockSpec.version`.

## Errors

| Situation | Error |
|---|---|
| Unknown field, invalid `algorithm`/`solver`, cross-algorithm hyperparameter key, negative or non-finite `alpha`, empty `target`, `${secret:` in config | `Error::Config` (in `mallops validate`, aggregated into `Error::Validation` under `blocks.<id>.with`) |
| Missing `target`, `with.target` differing from input `metadata["target"]`, or `target` absent from schema | `Error::Runtime` naming the target |
| Unsupported feature column type (not `Boolean`, `Int64`, or `Float64`) | `Error::Runtime` suggesting `encode`/`select` in preprocess |
| Null value in features or target | `Error::Runtime` naming the column |
| Non-finite (`NaN`/`±inf`) value | `Error::Runtime` naming the column and row |
| 0 rows, 0 features, or `linear_regression` with `rows < features + 1` | `Error::Runtime` |
| `logistic_regression` with `Float64` target or class count `!= 2` | `Error::Runtime` |
| `linear_regression` with target type other than `Int64`/`Float64` | `Error::Runtime` |
| Collinear / singular feature matrix | `Error::Runtime` (`"features are linearly dependent (e.g. one_hot with all categories); drop a column"`) |
| Non-finite intercept or coefficients after fit | `Error::Runtime` |
| Cancellation via `ctx.cancel()` | `Error::Cancelled` |

## Known limitations

- **In-memory dense matrix:** the entire training dataset is loaded into memory as a dense `n × k`
  `f64` matrix.
- **Per-machine determinism:** an identical `ArtifactId` is guaranteed on the same machine, not
  bit-for-bit across different CPUs or operating systems.
- **No convergence signal for logistic regression:** LBFGS runs up to 1000 iterations and does not
  report non-convergence.
- **Binary logistic regression only:** `logistic_regression` requires exactly 2 distinct target
  classes.

## Algorithms not in v0.1

Additional `smartcore` algorithms planned as potential follow-ups (not in v0.1):

- Multiclass logistic regression
- Regularized linear regression: Ridge, Lasso, ElasticNet
- Decision trees and random forests
- k-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)
- Naive Bayes
