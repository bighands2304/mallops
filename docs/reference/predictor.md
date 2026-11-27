# Predictor contract

The `Predictor` contract ([ADR-0007 D6/D7](../adr/0007-core-contracts-revision.md)) turns a model
artifact into an active inference component that answers prediction requests. `mallops-core` remains
strictly model-agnostic and free of machine learning dependencies; consumers pick a `Predictor` by
the artifact's `flavor` via a local `PredictorRegistry`.

In v0.1, the minimal contract lives in `mallops-std` (shared by `evaluate` and `serve`) before its
promotion to `mallops-core` ahead of 1.0.

See [SPEC FR-SRV-3](../SPEC.md#310-model-serving) and [ARCHITECTURE §7 Serving](../ARCHITECTURE.md#7-serving).

## Cargo features

All predictor components in `mallops-std` are opt-in (not in `default`):

| Feature | Dependencies | What it enables |
|---|---|---|
| `predictor` | `mallops-core`, `serde`, `serde_json`, `tokio` | Base types (`PredictInput`, `Predictor`, `LoadedPredictor`), `PredictorRegistry`, `register_std_predictors`, and the conformance suite `conformance::check`. |
| `predict-smartcore` | `predictor` | `SmartcorePredictor` for `smartcore-logistic` and `smartcore-linear`. Evaluates models in-process without depending on the `smartcore` crate. |
| `predict-external` | `predictor`, `tokio/process`, `tokio/io-util`, `tokio/time` | `ExternalPythonPredictor` for `sklearn-pickle`. Evaluates models via an external Python worker subprocess communicating over NDJSON v1. |

## Core types and traits

### `PredictInput`

Prediction request and response payloads:

```rust
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum PredictInput {
    /// A JSON document.
    Json(serde_json::Value),
    /// Opaque bytes with their media type.
    Bytes {
        /// Media type of `data`.
        content_type: String,
        /// The raw payload.
        data: Vec<u8>,
    },
}
```

- `PredictInput::Json` wraps any `serde_json::Value` (objects, numbers, arrays).
- `PredictInput::Bytes` carries unparsed raw payloads with an explicit MIME `content_type`.
- Marked `#[non_exhaustive]` to allow future zero-copy buffer variants.

### `Predictor`

Factory trait that loads model artifacts for one or more flavor identifiers:

```rust
pub trait Predictor: Send + Sync {
    /// The artifact `flavor` values this predictor can load (non-empty).
    fn flavors(&self) -> &[&str];

    /// Whether loading a model runs code shipped in the artifact (pickle, joblib, …).
    /// Such predictors are refused by [`PredictorRegistry::get`] unless trust is given.
    fn executes_code(&self) -> bool;

    /// Loads the model whose payload files are in `payload_dir`.
    ///
    /// # Errors
    ///
    /// Any [`Error`] if the payload is missing or malformed.
    fn load<'a>(
        &'a self,
        artifact: &'a Artifact,
        payload_dir: &'a Path,
    ) -> BoxFuture<'a, Result<Box<dyn LoadedPredictor>, Error>>;
}
```

- `flavors()`: slice of supported artifact flavor strings (must not be empty).
- `executes_code()`: returns `true` if loading the model executes code contained in the artifact
  (e.g., Python `pickle` / `joblib`). Governed by the [trust model](../SPEC.md#41-trust-model).
- `load(&'a self, artifact: &'a Artifact, payload_dir: &'a Path)`: loads the model from files
  in `payload_dir` against `artifact.manifest()`. Returns an asynchronous `BoxFuture` resolving
  to a trait object `Box<dyn LoadedPredictor>`.

### `LoadedPredictor`

A model instantiated in memory, ready to answer inference requests:

```rust
pub trait LoadedPredictor: Send + Sync {
    /// Predicts one input.
    ///
    /// # Errors
    ///
    /// Any [`Error`] if the input is malformed or the model fails.
    fn predict(&self, input: PredictInput) -> BoxFuture<'_, Result<PredictInput, Error>>;

    /// Predicts a batch; outputs are in input order.
    ///
    /// # Errors
    ///
    /// Any [`Error`] if any input is malformed or the model fails.
    fn predict_batch(
        &self,
        inputs: Vec<PredictInput>,
    ) -> BoxFuture<'_, Result<Vec<PredictInput>, Error>>;
}
```

- `predict`: answers a single prediction request.
- `predict_batch`: answers multiple prediction requests. The output vector must strictly match the
  length and order of `inputs`. Passing an empty `Vec` must succeed and return an empty `Vec`.

## `PredictorRegistry`

`PredictorRegistry` resolves predictors by artifact flavor. It is an explicit local instance
(no global state, no linker magic).

```rust
let mut registry = PredictorRegistry::new();
register_std_predictors(&mut registry)?;

// Resolving a predictor:
let predictor = registry.get("smartcore-logistic", false)?;
let loaded = predictor.load(&artifact, payload_dir).await?;
```

### Registration: `register`

```rust
pub fn register(&mut self, predictor: Arc<dyn Predictor>) -> Result<(), Error>
```

Registers `predictor` for all its declared flavors **atomically**.

- If `predictor.flavors()` is empty, returns `Error::Validation` (`"predictor declares no flavors"`).
- If `flavors()` contains duplicate entries, or if any flavor is already registered, returns
  `Error::Validation` (`"duplicate predictor flavor <flavor>"`).
- **Atomic rollback:** if any flavor conflicts, the registry is left completely unmodified.

### Resolution: `get`

```rust
pub fn get(&self, flavor: &str, trust_unsafe_flavors: bool) -> Result<Arc<dyn Predictor>, Error>
```

Retrieves the predictor registered for `flavor`, enforcing the code-execution trust policy:

- If no predictor handles `flavor`: returns `Error::Config("no predictor for flavor `<flavor>`")`.
- If `predictor.executes_code()` is `true` and `trust_unsafe_flavors` is `false`: returns
  `Error::Config("flavor `<flavor>` executes code from the artifact; pass --trust-unsafe-flavors to serve it")`.

### Standard predictors: `register_std_predictors`

```rust
pub fn register_std_predictors(registry: &mut PredictorRegistry) -> Result<(), Error>
```

Registers all standard predictors enabled by active Cargo feature flags:

- `SmartcorePredictor` when `predict-smartcore` is enabled (`smartcore-logistic`, `smartcore-linear`).
- `ExternalPythonPredictor` when `predict-external` is enabled (`sklearn-pickle`).

Fails with `Error::Validation` if a standard flavor is already present in `registry`.

## Conformance suite (`conformance::check`)

`mallops_std::predictor::conformance::check` is a reusable compliance test suite for any `Predictor`
implementation:

```rust
pub async fn check(
    predictor: &dyn Predictor,
    artifact: &Artifact,
    payload_dir: &Path,
    cases: &[(PredictInput, PredictInput)],
    malformed: PredictInput,
) -> Result<(), String>
```

It validates compliance without panicking, returning `Err(String)` on the first rule violation:

1. **Flavor advertisement:** if `artifact.manifest().flavor` is `Some(flavor)`, checks that
   `predictor.flavors()` contains `flavor`.
2. **Model loading:** calls `predictor.load(artifact, payload_dir).await` and verifies success.
3. **Single prediction correctness:** evaluates each `(input, expected)` pair in `cases` via
   `loaded.predict(input)` and confirms `got == expected`.
4. **Batch prediction preservation:** evaluates `loaded.predict_batch(cases.inputs)` and verifies
   that outputs match `expected` in exact input order.
5. **Empty batch handling:** evaluates `loaded.predict_batch(vec![])` and verifies that it returns
   an empty `Vec`.
6. **Error handling:** verifies that `loaded.predict(malformed)` returns an `Err`.

---

## `SmartcorePredictor`

Feature: `predict-smartcore`. Serves models trained by
[`std/train.smartcore`](train-smartcore.md) in-process without linking to `smartcore` (pure Rust, no BLAS,
zero unsafe code).

- **Flavors:** `smartcore-logistic`, `smartcore-linear`
- **`executes_code()`:** `false` (safe data format, permitted by default without `--trust-unsafe-flavors`)

### Load validation rules

During `load(artifact, payload_dir)`, `SmartcorePredictor` performs exhaustive validation:

1. **Artifact kind:** `artifact.manifest().kind` must be `ArtifactKind::Model`.
2. **Artifact format:** `artifact.manifest().format` must be `"json"`.
3. **Artifact flavor:** `artifact.manifest().flavor` must be present and match one of `smartcore-logistic`
   or `smartcore-linear`.
4. **Payload file:** `payload_dir/model.json` must exist, be valid JSON, and reject unknown fields
   (`deny_unknown_fields`).
5. **Schema version:** `v` must equal `1` (`MODEL_SCHEMA_VERSION`).
6. **Flavor & algorithm alignment:**
   - `smartcore-logistic` requires `algorithm: "logistic_regression"`.
   - `smartcore-linear` requires `algorithm: "linear_regression"`.
   - Any mismatch fails with an error.
7. **Features:** `features` list must be non-empty and must not contain duplicate feature names.
8. **Target:** `target` column name must not be empty.
9. **Coefficients:** `coefficients.len()` must equal `features.len()`, and all coefficients must be finite
   floating-point numbers.
10. **Intercept:** `intercept` must be a finite floating-point number.
11. **Algorithm-specific fields:**
    - For `logistic_regression`:
      - `classes` is required and must contain exactly **2** distinct classes (`int64`, `bool`, or `string`).
      - `threshold` is required and must be a finite floating-point number.
    - For `linear_regression`:
      - `classes` and `threshold` must be omitted (`null` / absent).

### Input and output shapes

- **Input:** `PredictInput::Json` containing a JSON object.
  - The object keys must match `model.json`'s `features` list **exactly** (missing or extra keys are
    rejected).
  - Each value must be a finite JSON number or a boolean (`true` maps to `1.0`, `false` maps to `0.0`).
  - Non-object JSON and `PredictInput::Bytes` are rejected with `Error::Runtime`.
- **Output (`smartcore-logistic`):**
  - Returns `PredictInput::Json` with shape:
    ```json
    {
      "prediction": <class>,
      "probability": <f64>
    }
    ```
  - Probability calculation: `p = 1.0 / (1.0 + exp(-(intercept + Σ coefficients[i] * x[i])))`.
  - Prediction decision: `classes[1]` when `p >= threshold`, otherwise `classes[0]`.
  - Preserves the original JSON type of `classes` (`int64`, `bool`, or `string`).
- **Output (`smartcore-linear`):**
  - Returns `PredictInput::Json` with shape:
    ```json
    {
      "prediction": <f64>
    }
    ```
  - Calculated as: `prediction = intercept + Σ coefficients[i] * x[i]`.

---

## `ExternalPythonPredictor`

Feature: `predict-external`. Serves models through an external Python worker process speaking NDJSON
protocol v1 over stdin/stdout.

- **Flavors:** `sklearn-pickle`
- **`executes_code()`:** `true` (unpickling arbitrary models executes arbitrary code; requires
  explicit `--trust-unsafe-flavors` opt-in)

### Command and configuration

- **Default command:** runs the embedded reference worker (`worker.py`) directly with Python:
  - Unix: `["python3", "-c", "<worker.py>"]`
  - Windows: `["python", "-c", "<worker.py>"]`
- **`with_command(Vec<String>)`:** overrides worker command arguments (executed directly, never through a shell).
- **`with_timeout(Duration)`:** sets per-request and handshake timeout (default: 30 seconds).

### Model file selection

During `load`, `ExternalPythonPredictor` locates the model file inside `payload_dir`:

1. Uses `payload_dir/model.pkl` if it exists.
2. If `model.pkl` is absent, searches `payload_dir` for regular files (subdirectories are ignored).
   If there is **exactly one** regular file (e.g. `model.joblib`), it is selected.
3. If `payload_dir` has 0 files or more than 1 file without `model.pkl`, loading fails with `Error::Runtime`.

### Worker environment and directory hygiene

The worker child process is spawned with an isolated environment and working directory:

- **Working directory:** set to `payload_dir` (resolved via `std::path::absolute` without UNC `\\?\` prefix
  on Windows to avoid Python import path confusion).
- **Environment isolation:** host environment is completely cleared (`env_clear`). The worker inherits only:
  - `PATH` (plus `SYSTEMROOT` on Windows).
  - `MALLOPS_MODEL_FILE`: absolute path to the resolved model file.
  - `MALLOPS_FEATURES`: JSON array string of feature names extracted in order from
    `artifact.manifest().io_schema.input[*].name` (omitted if empty or absent).
- **`sys.path` hygiene:** on startup, `worker.py` purges `''` (the cwd) and the absolute payload path from
  `sys.path` before any other imports. This prevents untrusted payload files from shadowing standard Python
  modules (such as a malicious `json.py` or `pickle.py`).
- **Stream separation:** `worker.py` immediately redirects `sys.stdout = sys.stderr`. All user/model print
  statements go to stderr, keeping stdout clean for the NDJSON protocol.
- **Stderr ring buffer:** an asynchronous background task drains the worker's stderr, keeping the most recent
  **4 KiB** (`STDERR_TAIL`). If the worker crashes or closes its pipes, the task drains stderr for up to
  1 second (`STDERR_DRAIN`), and the tail is included in `Error::Runtime`.
- **Protocol line limit:** protocol lines are bounded by **64 MiB** (`MAX_LINE = 64 << 20`). Exceeding this
  limit triggers a protocol violation error.

### NDJSON Protocol v1

Communication occurs over standard streams with one UTF-8 JSON object per line (`\n` terminated):

#### 1. Handshake

Upon loading the model into memory, the worker writes:

```json
{"v": 1, "type": "ready"}
```

The host must receive this within the configured timeout (default 30 s).

#### 2. Single prediction (`predict`)

**Host request:**

```json
{"id": 0, "type": "predict", "input": {"age": 42.0, "plan_a": 1.0}}
```

**Worker success response:**

```json
{"id": 0, "ok": true, "output": {"prediction": 1, "probability": 0.875}}
```

#### 3. Batch prediction (`predict_batch`)

**Host request:**

```json
{"id": 1, "type": "predict_batch", "inputs": [{"age": 25.0, "plan_a": 0.0}, {"age": 50.0, "plan_a": 1.0}]}
```

**Worker success response:**

```json
{"id": 1, "ok": true, "outputs": [{"prediction": 0, "probability": 0.12}, {"prediction": 1, "probability": 0.91}]}
```

#### 4. Error response

If a prediction fails (e.g. missing features or invalid input):

**Worker error response:**

```json
{"id": 2, "ok": false, "error": "ValueError: missing features ['plan_a']"}
```

The host maps this to `Error::Runtime("worker error: ValueError: missing features ['plan_a']")`. **A single failed request does not crash or terminate the worker.**

#### Input & output representations

- **Inputs:** `worker.py` accepts either an object `{"feature": value}` (ordered using `MALLOPS_FEATURES` or
  the model's `feature_names_in_`) or a list of values already in column order.
- **Outputs:** `{"prediction": <val>}`. If the model provides `predict_proba` and has exactly two classes
  (`len(classes_) == 2`), `"probability": <p>` is included — the probability of the greater class
  (`classes[1]` in sorted ascending order, matching scikit-learn's `classes_[1]` and
  `SmartcorePredictor`). Consumers such as [`std/evaluate.classification`](evaluate.md) rely on this
  contract to compute `roc_auc` (using `1 - p` when `with.positive` is `classes[0]`).
- NumPy scalars and arrays are automatically converted to standard JSON primitives via `.tolist()` / `.item()`.

### Lifecycle, failure recovery, and process management

- **Serialized requests:** inference requests are serialized through an asynchronous `Mutex`.
- **Failure recovery:** if a request times out, receives an invalid reply, finds the worker dead, or if the
  request future is cancelled/dropped mid-flight, the worker process is killed (`kill_on_drop(true)`). The
  worker slot is vacated.
- **Automatic restart:** on the subsequent request, a fresh worker child process is spawned automatically. The
  failed request itself is not retried.
- **Drop cleanup:** dropping `LoadedPredictor` drops the active worker child process, killing it immediately.
- **Platform note (Windows):** process termination on drop or crash terminates only the direct child process.
  There is no process-group kill on Windows, so Python workers must not spawn detached subprocesses.

---

## Errors reference

| Situation | Error variant | Behavior / Message format |
|---|---|---|
| Registering predictor declaring no flavors | `Error::Validation` | `"predictor declares no flavors"` (registry unchanged) |
| Duplicate flavor in `register` | `Error::Validation` | `"duplicate predictor flavor `<flavor>`"` (registry unchanged) |
| Standard predictor registered twice | `Error::Validation` | `"duplicate predictor flavor `<flavor>`"` |
| Unknown flavor requested in `get` | `Error::Config` | `"no predictor for flavor `<flavor>`"` |
| Code-executing flavor without trust opt-in | `Error::Config` | `"flavor `<flavor>` executes code from the artifact; pass --trust-unsafe-flavors to serve it"` |
| Manifest kind is not `model` | `Error::Config` | `"expected a model artifact, got \`<kind>\`"` |
| Manifest format is not `json` (`SmartcorePredictor`) | `Error::Config` | `"expected model format \`json\`, got \`<format>\`"` |
| Missing manifest flavor (`SmartcorePredictor`) | `Error::Config` | `"model artifact manifest is missing \`flavor\`"` |
| Unsupported manifest flavor | `Error::Config` | `"unsupported smartcore flavor \`<flavor>\`"` / `"unsupported flavor <flavor:?>"` (external, missing flavor included: `None`) |
| Missing or invalid `model.json` / `model.pkl` | `Error::Runtime` | `<path>: <io error>` or `"invalid model.json: <serde error>"` |
| Unsupported `model.json` schema version | `Error::Runtime` | `"unsupported model schema version <v>, expected 1"` |
| Algorithm / flavor mismatch | `Error::Runtime` | `"manifest flavor \`<flavor>\` does not match model algorithm \`<alg>\`"` |
| Empty or duplicate `features` in `model.json` | `Error::Runtime` | `"model \`features\` list must not be empty"` / `"duplicate feature \`<f>\` in model payload"` |
| Empty `target` in `model.json` | `Error::Runtime` | `"model \`target\` must not be empty"` |
| Coefficient length mismatch | `Error::Runtime` | `"model has <N> coefficients for <M> features"` |
| Non-finite intercept, coefficients, or threshold | `Error::Runtime` | `"model \`intercept\` must be finite"` / `"model \`coefficients\` must all be finite"` / `"logistic_regression \`threshold\` must be finite"` |
| Logistic model with class count != 2, or equal classes | `Error::Runtime` | `"logistic_regression model requires exactly 2 classes, got <N>"` / `"logistic_regression model requires 2 distinct classes"` |
| Linear model specifying classes or threshold | `Error::Runtime` | `"linear_regression model must not specify \`<field>\`"` |
| Input missing expected features or extra keys | `Error::Runtime` | `"missing feature \`<name>\`"` / `"unexpected feature \`<name>\`"` |
| Input feature value is not finite number or bool | `Error::Runtime` | `"feature \`<name>\` must be a finite number or boolean"` |
| Non-JSON input passed to smartcore/external | `Error::Runtime` | `"smartcore predictor expects a JSON object of feature values"` (non-object JSON) / `"smartcore predictor expects JSON input"` (bytes) / `"sklearn-pickle takes JSON input, got bytes"` |
| Ambiguous payload directory in external predictor | `Error::Runtime` | `"<dir>: expected model.pkl or exactly one file, found <N> files"` |
| Worker handshake timed out | `Error::Runtime` | `"worker handshake timed out after <duration>"` |
| Request timed out | `Error::Runtime` | `"timed out after <duration>"` (worker killed, restarts on next request) |
| Worker crashed or closed pipes | `Error::Runtime` | `"worker exited unexpectedly; stderr: <tail>"` |
| Worker returned error reply (`ok: false`) | `Error::Runtime` | `"worker error: <error message>"` (worker stays alive) |
| Protocol line exceeds 64 MiB limit | `Error::Runtime` | `"worker line longer than 67108864 bytes"` |
