# `std/evaluate` (`std/evaluate.classification` and `std/evaluate.regression`)

Model evaluation and quality-gate blocks for classification and regression. Behind the optional
`evaluate` Cargo feature of `mallops-std` (`evaluate = ["predictor", "_ingest", "dep:arrow-select"]`,
not in `default`, no new crate dependency; enabled in `mallops-cli` together with `predict-smartcore`
and `predict-external`).
See [SPEC FR-EVL-1 / FR-EVL-2](../SPEC.md#36-model-evaluation).

```yaml
blocks:
  eval:
    use: std/evaluate.classification
    in: { model: model.model, test: features.test }
    with:
      target: churned
      positive: true
      gate:
        metric: f1
        min: 0.80
        on_fail: skip
```

For regression:

```yaml
blocks:
  eval:
    use: std/evaluate.regression
    in: { model: model.model, test: features.test }
    with:
      gate:
        metric: rmse
        max: 0.25
        on_fail: fail
```

## Configuration (`with:`)

Unknown fields are rejected (`deny_unknown_fields`). `validate_config` uses the same parser as
`run`, so `mallops validate` catches configuration errors before execution.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `target` | non-empty string | none | Target column name. Falls back to `metadata["target"]` of the `test` input artifact if omitted; if absent in both, or if both are set and differ, the run fails. |
| `positive` | JSON scalar (`int64`, `bool`, or `string`) | none | **Classification only.** Explicit positive class label for binary classification. Rejected on `std/evaluate.regression` and on multiclass (`> 2` classes) classification. |
| `gate` | object | none | Optional quality gate: `{ metric, min?, max?, on_fail? }`. |

### `gate` fields

| Key | Type | Default | Constraints |
|---|---|---|---|
| `metric` | string | required | Must be a metric of the block's task (`accuracy`, `precision`, `recall`, `f1`, `roc_auc` for classification; `rmse`, `mae`, `r2` for regression). |
| `min` | `f64` | none | Finite inclusive lower bound (`value >= min`). |
| `max` | `f64` | none | Finite inclusive upper bound (`value <= max`). |
| `on_fail` | `skip` \| `fail` | `skip` | Action when the gate check fails. |

Validation rules (`validate_config`):

- A gate `metric` not belonging to the block's task is rejected.
- `gate` must specify at least one of `min` or `max`. Specifying both is allowed (closed interval
  `[min, max]`, endpoints inclusive) as long as `min <= max`.
- Non-finite `min` or `max` values (`NaN`, `±inf`) are rejected.
- Setting `positive` on `std/evaluate.regression` is rejected.
- Strings containing `${secret:` anywhere in the config are rejected (as in
  [`std/external`](external.md)).

## Ports

Both blocks have `version: "0.1.0"` and `Stage::Evaluate`:

- **Block IDs:** `std/evaluate.classification`, `std/evaluate.regression`.
- **Inputs:**
  - `model` (required): `kind: model`, any `format`, any `flavor` (dispatched at runtime through
    [`PredictorRegistry`](predictor.md#predictorregistry)).
  - `test` (required): `kind: features`, `format: parquet`.
- **Output:**
  - `metrics`: `kind: metrics`, `format: json`.

## Execution and input data rules

- **Predictor resolution:** `model` is fetched into `model/` and `test` into `test/` inside
  `ctx.workdir()`. The predictor is resolved via `predictors.get(flavor, true)`: `mallops run`
  executes trusted pipeline definitions ([SPEC §4.1](../SPEC.md#41-trust-model)), so code-executing
  flavors such as `sklearn-pickle` are allowed during evaluation. A missing `flavor` in the model
  manifest is an error. Without any `predict-*` feature enabled in `mallops-std`, the blocks still
  register but fail at runtime with `"no predictor for flavor \`<flavor>\`"`.
- **Target resolution:** uses `with.target` when set, otherwise `metadata["target"]` (string) from
  the `test` input artifact manifest. Missing in both is an error; present in both with different
  values is an error.
- **Test dataset:** the entire `test` Parquet file is loaded into memory. Empty datasets (`0` rows)
  fail with `Error::Runtime`.
- **Feature columns sent to the predictor:**
  - If `model.manifest.io_schema.input` is a JSON array of strings, exactly those columns are sent
    (a missing column fails with `Error::Runtime`).
  - Otherwise, every column in the `test` Parquet schema except the target column is sent.
  - Supported Arrow feature column types are `Int64`, `Float64`, `Boolean`, and `Utf8`. Nulls are
    encoded as JSON `null`; non-finite `Float64` values (`NaN`, `±inf`) are also encoded as JSON
    `null` (leaving handling to the predictor). Any other Arrow type (including `Date32`) fails with
    `Error::Runtime`.
- **Prediction call:** each row is encoded as `PredictInput::Json` (`{ "<col>": <val>, ... }`) and
  evaluated in a single `LoadedPredictor::predict_batch` call.
  - `outputs.len()` must equal the number of input rows, otherwise `Error::Runtime`.
  - Every output must be a `PredictInput::Json` object containing `prediction` (and optional
    `probability`), otherwise `Error::Runtime`.
- **Cancellation:** `ctx.cancel()` is checked at run start, after fetching inputs, and after
  `predict_batch`.

## Metrics semantics

Every computed metric must be finite; a non-finite metric (`NaN`, `±inf`, e.g. from floating-point
overflow) fails with `Error::Runtime` regardless of `gate.on_fail`.

### Classification (`std/evaluate.classification`)

- **Target (`y_true`):** Arrow `Int64`, `Boolean`, or `Utf8`. `Float64` and null values are
  `Error::Runtime`.
- **Label typing and class set:** labels are parsed into a typed enum (`int64`, `bool`, or `string`).
  Every predicted `prediction` value and `with.positive` (if configured) must have the same type as
  `y_true`, otherwise `Error::Runtime`. The class set is the sorted set of distinct values in
  `y_true ∪ y_pred` (plus `{with.positive}` when set).
- **Binary (`<= 2` classes):**
  - `positive` is `with.positive` when configured, otherwise the greater class (`classes[1]`,
    matching `smartcore` and scikit-learn). If only one class is present and `with.positive` is
    omitted, the run fails with `Error::Runtime` (cannot infer the positive class).
  - `accuracy`, `precision`, `recall`, `f1` are computed with respect to `positive`; a division by
    zero yields `0.0` (matching scikit-learn's `zero_division=0`).
  - `roc_auc` is computed by Mann-Whitney ranks (average ranks for ties) **only when** all of the
    following hold: the task is binary, **every** output carries a valid `probability`, and `y_true`
    contains both classes. Per the [Predictor contract](predictor.md#input-output-representations),
    `probability` is `P(classes[1])` (the greater class); when `positive` is the smaller class
    (`classes[0]`), `1 - p` is used. If `probability` is absent from all rows or `y_true` contains
    only one class, `roc_auc` is omitted. If `probability` is present in only some rows, or is
    non-numeric, non-finite, or outside `[0.0, 1.0]`, the run fails with `Error::Runtime`.
- **Multiclass (`> 2` classes):**
  - Setting `with.positive` fails with `Error::Runtime`.
  - `accuracy` plus macro-averaged `precision`, `recall`, `f1` (division by zero yields `0.0`).
  - `roc_auc` is omitted.

### Regression (`std/evaluate.regression`)

- **Target (`y_true`):** Arrow `Int64` or `Float64` cast to `f64` (nulls and non-finite values fail
  with `Error::Runtime`).
- **Predictions (`y_pred`):** `prediction` must be a finite JSON number (nulls and non-finite values
  fail with `Error::Runtime`).
- **Metrics:** `rmse`, `mae`, `r2`.
- **Constant `y_true`:** when `y_true` has zero variance, `r2` is `1.0` if all residuals are `0.0`
  and `0.0` otherwise (matching scikit-learn's `force_finite=True`).

## Output (`metrics.json`) and manifest

The output artifact payload is a single file, `metrics.json`, containing a flat JSON object mapping
computed metric names to finite `f64` values with keys sorted lexicographically:

```json
{
  "accuracy": 0.92,
  "f1": 0.89,
  "precision": 0.91,
  "recall": 0.87,
  "roc_auc": 0.95
}
```

Output manifest fields:

- `kind`: `metrics`
- `format`: `json`
- `metadata`:
  - `task`: `"classification"` or `"regression"`
  - `target`: target column name
  - `rows`: number of evaluated rows
  - `positive`: positive class label (present only for binary classification)
  - `gate`: present only when `with.gate` is configured (`{ "metric", "min"?, "max"?, "on_fail", "passed" }`)
  - No timestamps, run IDs, or block-written `parents` (the runtime injects `parents` on commit).

## Quality gate semantics

When `with.gate` is configured:

- If the gate's `metric` was not computed at runtime (for example `roc_auc` in multiclass
  classification or when the predictor does not emit probabilities), the block returns
  `Error::Config` regardless of `on_fail`.
- The gate passes iff `min.is_none_or(|m| v >= m) && max.is_none_or(|m| v <= m)` (inclusive
  endpoints; non-finite metrics fail before the gate check).
- **Passed:** returns `BlockOutcome::Produced` with `metrics`.
- **Failed with `on_fail: skip` (default):**
  - Returns `BlockOutcome::Skipped { reason, outputs }` (e.g. `"gate failed: f1 = 0.75 < min 0.8"`).
    The `metrics` artifact is still committed to the store so the failed evaluation can be inspected.
  - **Downstream blocking:** the runtime skips only downstream nodes that consume `eval.metrics` as
    a **required** input. For example, `std/package` must wire `metrics: eval.metrics` (as in
    [SPEC §5](../SPEC.md#5-pipeline-definition-mallopsyaml)); a consumer without that required edge
    still runs.
  - **`runs/latest.json` and serving:** when any node is `Skipped`, `mallops run` exits `0` but
    leaves `runs/latest.json` unchanged, so `mallops serve` continues serving the previous accepted
    package.
  - **Caching:** `Skipped` nodes are never cached by the runtime, so a failed gate is re-evaluated
    on the next run.
- **Failed with `on_fail: fail`:**
  - Returns `Error::Block { id, source: reason }`. The runtime marks the node `Failed`, cancels all
    descendants, does **not** commit `metrics` (outputs are only committed on `Ok`), leaves
    `runs/latest.json` unchanged, and `mallops run` exits `1`.

## Cache

`CachePolicy::Deterministic { extra_hash: None }`.

- Same inputs, config, node `seed:`, and block version hit the cache when the previous run
  `Succeeded`. `Skipped` and `Failed` runs are never cached.
- Changing metric computation semantics requires bumping `BlockSpec.version`.

## Errors

| Situation | Error |
|---|---|
| Unknown field, invalid gate `metric` for task, neither `min` nor `max` set, `min > max`, non-finite `min`/`max`, `positive` on regression, `${secret:` in config | `Error::Config` (in `mallops validate`, aggregated into `Error::Validation` under `blocks.<id>.with`) |
| Missing predictor for model `flavor` (including when no `predict-*` feature is enabled) | `Error::Config` |
| Gate `metric` not available at runtime (e.g. `roc_auc` omitted) | `Error::Config` |
| Missing `flavor` in model manifest, missing `target`, `with.target` differing from `metadata["target"]`, or missing column | `Error::Runtime` |
| Unsupported Arrow column type, `0` rows, invalid target value (`Float64`/null in classification, null/non-finite in regression) | `Error::Runtime` |
| `outputs.len() != rows`, non-object predictor output, missing `prediction`, type mismatch in labels/`positive`, `positive` set on multiclass, single class without `with.positive` | `Error::Runtime` |
| `probability` present in only some rows, non-numeric, non-finite, or outside `[0.0, 1.0]` | `Error::Runtime` |
| Non-finite computed metric (e.g. overflow in regression) | `Error::Runtime` |
| Gate failed with `on_fail: fail` | `Error::Block` |
| Cancellation via `ctx.cancel()` | `Error::Cancelled` |

## Known limitations

- **In-memory test set:** the entire `test` Parquet artifact is loaded into memory and sent in a
  single `predict_batch` call.
- **Cancellation during `predict_batch`:** `ctx.cancel()` is checked at run start, after fetching
  inputs, and after `predict_batch`, but does not interrupt an in-flight `predict_batch` call. For
  [`ExternalPythonPredictor`](predictor.md#externalpythonpredictor), in-flight calls are already
  bounded by the worker's per-request timeout (default 30 s).
- **No production-model comparison in v0.1:** comparing metrics against the current production model
  in a registry is planned for v0.2 ([SPEC §3.6](../SPEC.md#36-model-evaluation)).
- **Shared standard `PredictorRegistry` in v0.1:** `register_std` builds a single
  `PredictorRegistry` with `register_std_predictors` and shares it between both evaluate blocks;
  injecting custom predictors from library code is out of scope for v0.1.
