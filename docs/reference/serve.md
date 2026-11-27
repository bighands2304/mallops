# `mallops serve`

HTTP/JSON inference server that serves a self-contained `package` artifact produced by `mallops run`
([`std/package`](package.md)). Serving runs as a native process (`mallops` binary) outside the batch
DAG; container packaging (`FR-PKG-2`) and Docker/Kubernetes deploy targets (`FR-DEP-1`) are deferred
to v0.2. Flavors backed by [`ExternalPythonPredictor`](predictor.md#externalpythonpredictor)
(`sklearn-pickle`) spawn a local Python worker subprocess, not a container.

In `mallops-std`, the HTTP server lives in `mallops_std::serve` behind the optional `serve-http`
Cargo feature (not in `default`; enabled in `mallops-cli`).
See [SPEC FR-SRV-1](../SPEC.md#310-model-serving), [ARCHITECTURE §7](../ARCHITECTURE.md#7-serving),
and the [Predictor contract](predictor.md).

## CLI usage and configuration

```bash
mallops serve [PACKAGE_ID] \
  [--host <HOST>] \
  [--port <PORT>] \
  [--trust-unsafe-flavors] \
  [--max-body-bytes <N>] \
  [--timeout <SECS>] \
  [--max-concurrency <N>] \
  [--auth-token-secret <NAME>]
```

Global flags (`-f` / `--file <PATH>`, default `mallops.yaml`; `-v` / `--verbose`) apply as usual.
Unlike other subcommands (which default to `WARN`), `mallops serve` initializes stderr logging at
`INFO` so per-request metadata logs are visible by default (`-v` raises the level to `DEBUG`).

### Precedence and defaults

Configuration is resolved with precedence:

**CLI flags > `serve:` in `mallops.yaml` > defaults**

The top-level `serve:` section of `mallops.yaml` supports `package`, `host`, and `port`:

```yaml
serve:
  package: package.package
  host: 127.0.0.1
  port: 8080
```

| Setting | CLI flag | `mallops.yaml` | Default |
|---|---|---|---|
| Bind host | `--host <HOST>` | `serve.host` | `127.0.0.1` |
| Bind port | `--port <PORT>` | `serve.port` | `8080` (`0` = OS-assigned) |
| Max request body | `--max-body-bytes <N>` | — | `10485760` (10 MiB) |
| Request timeout | `--timeout <SECS>` | — | `30` s |
| Max concurrent requests | `--max-concurrency <N>` | — | `64` |
| Allow code-executing flavors | `--trust-unsafe-flavors` | — | `false` |
| Bearer auth secret name | `--auth-token-secret <NAME>` | — | none (unauthenticated) |

### Package resolution and store requirement

- **Without `PACKAGE_ID`:** requires a `serve:` section in `mallops.yaml` and a `runs/latest.json`
  record in the store whose `pipeline` matches `name` in `mallops.yaml`. The server resolves
  `serve.package` (`<block>.<port>`) from `latest.json`'s `outputs`. If no matching successful run
  exists, startup fails with:
  ```text
  error: no successful run: run `mallops run` first
  ```
- **With explicit `PACKAGE_ID`:** loads that `ArtifactId` directly from the store configured in
  `mallops.yaml`. `mallops.yaml` is still required to locate and open the store.
- **Persistent store required:** `mallops serve` requires a persistent store (`store.kind: local`
  in v0.1). Configuring `store.kind: memory` fails with:
  ```text
  error: serve needs a persistent store
  ```

### Startup and shutdown

- Once the package and predictor are loaded and the listener is bound, the CLI prints:
  ```text
  serving <artifact-id> on http://<addr>
  ```
- Graceful shutdown is triggered by `Ctrl-C` (`SIGINT`) on all platforms and `SIGTERM` on Unix.
  In-flight requests finish before the server drops the active `LoadedPredictor` (terminating any
  external Python worker) and removes its temporary package directory.

## HTTP endpoints

### `GET /health`

Liveness and readiness probe. Always unauthenticated and exempt from the request timeout and
concurrency limiter so health checks do not fail with `503` when `/predict` is saturated.

**Response (`200 OK`):**

```json
{
  "status": "ok",
  "package": "<64-hex-artifact-id>",
  "flavor": "smartcore-logistic"
}
```

After a hot-swap, `package` (and `flavor`) immediately report the newly active package.

### `POST /predict`

Runs inference against the active package's optional fitted transform and `Predictor`.

#### Request and response shapes

- **Single input:** a JSON object `{...}` returns a single JSON prediction object `{...}`:
  ```bash
  curl -s -X POST http://127.0.0.1:8080/predict \
    -H 'Content-Type: application/json' \
    -d '{"age": 42, "tenure": 12}'
  ```
  ```json
  {"prediction": 1, "probability": 0.87}
  ```
- **Batch input:** a JSON array of objects `[{...}, ...]` returns a JSON array of predictions in
  the exact same order. An empty array `[]` returns `200 OK` with `[]`.
  ```bash
  curl -s -X POST http://127.0.0.1:8080/predict \
    -H 'Content-Type: application/json' \
    -d '[{"age": 42, "tenure": 12}, {"age": 25, "tenure": 2}]'
  ```
  ```json
  [{"prediction": 1, "probability": 0.87}, {"prediction": 0, "probability": 0.14}]
  ```

#### Preprocessing pipeline

- **Package with a bundled transform (`transform.json`):** input rows (including a single object,
  wrapped as a 1-element array) are passed through `FittedTransform::apply_json`, converted to JSON
  row objects containing the columns declared in `model.io_schema.input`, and evaluated via
  `LoadedPredictor::predict_batch`. Loading a package that bundles a transform fails with
  `Error::Config` if `mallops-std` was compiled without the `preprocess-tabular` feature.
- **Package without a transform:** rows are passed directly to the predictor.

#### Middleware order

Every `POST /predict` request passes through the following layers in order:

1. **Metadata logging** (`tracing::info!`)
2. **Request timeout** (`--timeout`, returns `504`)
3. **Bearer authentication** (when configured, returns `401` before reading the body)
4. **Concurrency permit** (`--max-concurrency`, returns `503` if saturated; held until response completion)
5. **Body size limit** (`--max-body-bytes`, returns `413`)
6. **JSON decode, transform, and prediction**

#### Status codes

All error responses — including `400` and `413` — are returned as JSON `{"error": "<message>"}`.

| Status | Meaning |
|---|---|
| `200 OK` | Prediction succeeded (JSON object for single input, JSON array for batch input, `[]` for empty batch). |
| `400 Bad Request` | Malformed JSON, payload is neither an object nor an array of objects, or fitted transform validation failed (`apply_json`). |
| `401 Unauthorized` | Missing `Authorization` header, scheme other than `Bearer`, or invalid token. Includes `WWW-Authenticate: Bearer` and a generic error message. |
| `413 Payload Too Large` | Request body exceeds `--max-body-bytes` (default 10 MiB). |
| `500 Internal Server Error` | Predictor runtime error, or predictor returned non-JSON bytes (`PredictInput::Bytes`; the detail `"non-JSON prediction"` goes to the `warn` log only). The client always receives `{"error":"prediction failed"}`; the request body is never echoed. |
| `503 Service Unavailable` | Concurrency limit (`--max-concurrency`, default 64) reached. |
| `504 Gateway Timeout` | Request exceeded `--timeout` (default 30 s). |

## Security and hardening

### Bearer authentication

Pass `--auth-token-secret <NAME>` to require `Authorization: Bearer <token>` on `POST /predict`.

- The CLI reads the token from the `MALLOPS_SECRET_<NAME>` environment variable via
  `EnvSecretProvider`.
- A missing environment variable fails at startup with `Error::Secret`; an empty or
  whitespace-only token fails with `Error::Config`.
- The server hashes the token with BLAKE3 on load and retains only the `blake3::Hash` digest (not
  the raw `Secret`), comparing incoming tokens in constant time via `blake3::Hash` equality.
- Missing header, non-`Bearer` scheme, and wrong token all return `401 Unauthorized` with
  `WWW-Authenticate: Bearer` and the same generic error message.
- `GET /health` remains unauthenticated for liveness/readiness probes.
- The server binds to `127.0.0.1` by default. **Whenever you bind to `0.0.0.0` or another
  non-loopback address, enable `--auth-token-secret`** and terminate TLS at a reverse proxy.

### Code-executing model flavors (`--trust-unsafe-flavors`)

Model formats that execute arbitrary code on deserialization (such as `sklearn-pickle`, where
`Predictor::executes_code()` is `true`) are refused with `Error::Config` unless
`--trust-unsafe-flavors` is explicitly passed. This trust gate is enforced both at startup and on
every hot-swap. See [SPEC §4.1 Trust model](../SPEC.md#41-trust-model) and
[`PredictorRegistry::get`](predictor.md#resolution-get).

### Request logging (metadata only)

Each `POST /predict` request emits a single structured `tracing::info!` event containing metadata only:
`method`, `path`, `status`, `latency_ms`, `req_bytes`, and `resp_bytes`. `GET /health` and unknown
routes are not logged.

- Request and response bodies are **never** logged.
- `4xx` responses log only metadata (error details from JSON parsing or `apply_json` are not logged
  because they may contain input values).
- `500` responses additionally log the internal predictor error at `warn` level (for external
  predictors, this may include the worker's stderr tail). Without a `transform` in the package, a
  predictor error may name a column sent by the client in that `warn` log (never the values).

## Retraining and hot-swap (`SIGHUP`)

Retraining with `mallops run` creates a new immutable `package` artifact (a new `ArtifactId`).
`runs/latest.json` is updated **only** when every node in the pipeline reaches `Succeeded` — if a
quality gate skips `std/package` or any node fails, `runs/latest.json` stays on the previous
accepted package.

On **Unix**, sending `SIGHUP` (`kill -HUP <pid>`) reloads the server without dropping traffic:

```bash
# 1. Retrain and promote a new package (updates runs/latest.json only if all nodes succeed)
mallops run

# 2. Hot-swap the running server to the newly promoted package
kill -HUP <pid>
```

### Continuity guarantees

- **Early signal installation:** the CLI installs the `SIGHUP` handler before initial
  `Server::load`, so a signal arriving during startup never terminates the process.
- **Metadata-only reload:** on `SIGHUP`, the CLI re-reads `runs/latest.json` from the store using
  the same `pipeline` name and `serve.package` reference configured at startup (`mallops.yaml` is
  not re-read) and calls `ServeHandle::swap`.
- **No-op on unchanged id:** if the resolved `ArtifactId` matches the active package, `swap`
  returns `Ok(false)` without reloading.
- **Build before swap:** the new state (`Package::load` into a fresh temporary directory, the
  `--trust-unsafe-flavors` gate, optional `FittedTransform`, and `Predictor::load`) is built
  completely before touching the active state.
- **Atomic swap & in-flight drain:** the active `Arc<State>` is swapped atomically without holding
  a lock across `.await`. Requests already in flight finish on the previous state; the old
  predictor and its temporary directory are dropped only when the last in-flight request completes.
- **Serialized swaps:** reloads are serialized through an async mutex so rapid `SIGHUP` signals do
  not load multiple packages concurrently.
- **Failure keeps the old package:** if `latest.json` is unreadable, its `pipeline` does not match,
  or loading the new package fails (including trust gate rejection), a warning is emitted (a
  `warning: ...` line on stderr from the CLI for `latest.json` problems, a `tracing` `warn` from
  `mallops-std` for load and trust failures) and the server continues serving the previous package without interruption.
- **Schema change warning:** if the new package has a different `io_schema`, the swap completes and
  logs a `tracing` `warn` (`mallops-std`).
- **Health update:** `GET /health` immediately reports the new `package` id and `flavor`.

### Operational notes for `SIGHUP`

- **Explicit `PACKAGE_ID`:** when started as `mallops serve <PACKAGE_ID>`, the `SIGHUP` handler is
  still installed so the signal does not kill the server, but reloads are ignored with a
  `warning: ...` line on stderr.
- **Unix only:** `SIGHUP` and `SIGTERM` handling exist only on Unix (`#[cfg(unix)]`). On Windows,
  only `Ctrl-C` is handled; reload requires restarting the process.
- **Stopping the server:** because `SIGHUP` reloads instead of terminating on Unix, closing the
  controlling terminal (hangup) triggers a reload rather than exit. Always stop `mallops serve`
  with `SIGTERM` or `Ctrl-C` (`SIGINT`).
- **No `/reload` endpoint or polling in v0.1:** remote reload endpoints and store polling are
  deferred to v0.3.

## Known limitations

- **Non-preemptive timeout for synchronous predictors:** `SmartcorePredictor` evaluates in-process
  within a single poll, so `tokio::time::timeout` cannot preempt it mid-computation. If a large
  batch exceeds `--timeout`, the `504` is sent only after the computation finishes, and CPU-heavy
  batches can occupy a Tokio worker thread. Bound batch size with `--max-body-bytes`.
- **Serialized external worker:** `ExternalPythonPredictor` serializes requests through an async
  mutex. Request timeouts work asynchronously on I/O, but timing out drops the request and kills
  the Python worker process, which restarts automatically on the next request.
- **Reverse proxy required in production:** `mallops serve` speaks plain HTTP/1.1 without TLS,
  connection caps, or header-read timeouts. Deploy behind a reverse proxy (e.g. nginx, Caddy,
  Envoy) for TLS termination, connection limits, and protection against slow clients. When binding
  to `0.0.0.0`, always configure `--auth-token-secret`.
- **No Docker/OCI in v0.1:** `mallops serve` runs as a native process. OCI image packaging
  (`FR-PKG-2`) and Docker/Kubernetes deploy blocks (`FR-DEP-1`) are planned for v0.2.

## Rust API (`mallops_std::serve`)

Enabled by the `serve-http` Cargo feature of `mallops-std`:

```rust
use mallops_std::serve::{ServeHandle, ServeOptions, Server};

let mut opts = ServeOptions::default();
opts.max_concurrency = 64;

let server = Server::load(store.as_ref(), &package_id, &predictors, opts).await?;
let handle: ServeHandle = server.handle();

let listener = tokio::net::TcpListener::bind("127.0.0.1:8080").await?;
server.run(listener, shutdown_signal).await?;
```

- **`ServeOptions`** (`#[non_exhaustive]`, `Default`): holds `host` (`"127.0.0.1"`), `port`
  (`8080`), `max_body_bytes` (10 MiB), `timeout` (30 s), `max_concurrency` (`64`),
  `trust_unsafe_flavors` (`false`), and `auth_token: Option<Secret>` (`None`; an empty token fails
  with `Error::Config`). `host` and `port` are configuration values for callers/CLI; binding the
  `TcpListener` is the caller's responsibility.
- **`Server::load(store, id, predictors, opts)`:** verifies and materializes the package into a
  server-owned temporary directory via `Package::load`, enforces `predictors.get(flavor,
  opts.trust_unsafe_flavors)`, and loads the optional `FittedTransform` and `LoadedPredictor`.
- **`Server::handle(&self) -> ServeHandle`:** returns a cheaply `Clone`able handle for hot-swaps.
- **`Server::run(self, listener, shutdown)`:** serves HTTP/1.1 on `listener` until `shutdown`
  resolves, drains in-flight requests, and drops the active state (predictor + temporary directory).
- **`ServeHandle::swap(&self, store, id, predictors) -> Result<bool, Error>`:** atomically replaces
  the active package. Returns `Ok(false)` when `id` is already active, `Ok(true)` on a successful
  swap, or `Err` (logging a `warn` and leaving the active package unchanged).
