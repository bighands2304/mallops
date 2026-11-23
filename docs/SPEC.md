# mallops — Technical Specification

> Status: **Draft** · Target: v0.1 · Last update: 2026-11-23

## 1. Vision

mallops is a modular MLOps framework written in Rust. It covers the whole machine-learning lifecycle — from data ingestion to serving, monitoring and retraining, down to edge devices with federated learning — through **composable blocks**.

It can be used in three ways, all backed by the same core:

| Mode | Who | How |
|---|---|---|
| **Library** | Rust developers | depend on `mallops-core`, implement `Block`, compose pipelines in code |
| **Plugins** | anyone, any language | ship a block as a WASM component or as an external process speaking the mallops gRPC protocol |
| **Standalone** | ML engineers, ops | describe a pipeline in `mallops.yaml`, run it with the `mallops` CLI, add/remove blocks without writing code |

### 1.1 Design principles

1. **Format and library agnostic.** mallops never imposes a model format, a data format or an ML library. An artifact is *opaque files + a manifest*. ONNX, safetensors, TorchScript, pickle, GGUF, XGBoost JSON, a custom binary — all first-class.
2. **Everything is a block.** Every lifecycle step, including data storage, deploy and retrain triggers, is a replaceable block with declared inputs and outputs (artifact store and lineage are core contracts acting on every edge; see [ADR-0006](adr/0006-store-and-lineage-contracts.md)).
3. **Small core, optional everything else.** The core has no ML dependency. Standard blocks live behind Cargo feature flags.
4. **Reproducible by default.** Content-addressed artifacts, recorded configs, seeds and environment.
5. **Local first, scale out later.** A full pipeline runs on a laptop with zero infrastructure; the same pipeline targets object storage, remote runners and edge fleets by swapping blocks.

### 1.2 Non-goals

- Not a container orchestrator (it can *call* Kubernetes, Nomad, Docker as deploy blocks).
- Not a deep-learning framework (it *wraps* burn, candle, PyTorch, scikit-learn, …).
- Not a data warehouse or feature store (it *integrates* with them via blocks).
- No GUI in v1 (a dashboard may come as a separate project).

## 2. Glossary

| Term | Definition |
|---|---|
| **Block** | Unit of work with typed inputs, outputs and config. Implements the `Block` trait (or the plugin protocol). |
| **Stage** | Lifecycle category a block belongs to (ingest, preprocess, train, …). Used for docs, validation hints and CLI listing — not enforced ordering. |
| **Artifact** | Immutable output of a block: a set of files plus a manifest, identified by its content hash. |
| **Pipeline** | DAG of block instances wired by artifact references, defined in YAML or code. |
| **Run** | One execution of a pipeline; has an id, a lineage record and logs. |
| **Flavor** | Tag in a model manifest telling which loader/predictor can use it (e.g. `onnx`, `torchscript`, `sklearn-pickle`, `gguf`, `custom:acme`). |
| **Artifact store** | Internal storage through which blocks exchange artifacts. Contract in the core, backends pluggable. Not to be confused with *data storage* blocks. |
| **Lineage sink** | Pluggable destination of the lineage events emitted by the runtime (JSON file, SQLite, OpenLineage, MLflow…). |
| **Plugin** | Block distributed outside the mallops binary (Rust crate, WASM component, or external process). |
| **Edge node** | Device running `mallops-edge`: local serving, local metrics, optional federated training. |

## 3. Functional requirements

Each stage below lists what the **standard library of blocks** (`mallops-std`) must offer and what the **core** must make possible. Anything not covered can be added as a plugin.

### 3.1 Data ingestion
- FR-ING-1: read from local files — in v0.1, `mallops-std` provides `std/ingest.csv` (feature `ingest-csv`) and `std/ingest.parquet` (feature `ingest-parquet`; meta-feature `ingest`, enabled by default), both `Stage::Ingest` version `0.1.0`; JSON/NDJSON and arbitrary-file ingest blocks are deferred follow-ups:
  - Both blocks reject unknown `with:` keys (`deny_unknown_fields`), run file I/O, conversion, and source hashing on `tokio::task::spawn_blocking` (checking `ctx.cancel()` between batches), resolve relative `path` values against the current working directory (`mallops run` changes into the directory of `mallops.yaml`, matching `store.path`), emit one `dataset` output (`kind: dataset`, `format: parquet`) backed by a single uncompressed `data.parquet` file with fixed footer `created_by: "mallops-std"` (deterministic bytes per `mallops-std` version) committed by the runtime (never calling `store.put` directly), and return `CachePolicy::Deterministic { extra_hash: Some(blake3(source_file)) }` (streamed BLAKE3 digest; missing source file errors) so editing the local file invalidates cached runs. Known v0.1 limitation: `extra_hash` is computed in `cache_policy` and the file is read again in `run` (TOCTOU); modifying a local source file concurrently during a run can associate the old cache key with the new content.
  - `std/ingest.csv` `with:` config: `path` (required), `delimiter` (ASCII `char` other than `\n`, `\r`, or `"`, default `','`; invalid values fail with `Error::Config`), `has_header` (`bool`, default `true`), `schema` (optional list of `{ name, type }`). Empty CSV files (0 columns) fail with `empty CSV: no columns`; header-only files (0 data rows) are valid. Without `schema`, column types are inferred across the entire file (`arrow-csv` may also infer `Null` for all-empty columns or `Timestamp` for ISO-8601 datetime strings — second-precision timestamps are promoted to milliseconds because Parquet has no seconds unit, so non-Arrow readers such as pandas see a timestamp rather than an integer — which lie outside the v0.1 `schema:` subset and therefore cannot be validated with `schema:` — a known v0.1 limitation). With `schema`, `has_header: true` requires the CSV header to match the declared column names in exact order (`schema mismatch` otherwise), while `has_header: false` checks that the first row has the exact declared column count (`schema mismatch: expected N columns, found M`) and assigns the declared names. Empty cells parse as `null`; non-empty unconvertible values fail the node with an error naming the source file.
  - `std/ingest.parquet` `with:` config: `path` (required), `schema` (optional list of `{ name, type }`). Reads external Parquet files compressed with Snappy, LZ4, Gzip, or Zstd (as well as uncompressed), checks exact column names and types in order when `schema` is set (`schema mismatch` otherwise), strips source schema and field key-value metadata (e.g. `pandas`), and rewrites an uncompressed `data.parquet` so identical data with different source metadata produces identical bytes.
- FR-ING-2: read from HTTP(S) and object stores (S3, GCS, Azure) — v0.2.
- FR-ING-3: read from databases (PostgreSQL, SQLite) and streams (Kafka, MQTT) — v0.3+. MQTT: `std/ingest.mqtt`, a bounded read (`max_messages`, `max_duration` or `idle_timeout`) into a `dataset`, uncacheable, at-least-once up to the block return ([ADR-0009](adr/0009-mqtt-blocks-and-serve-transport.md)).
- FR-ING-4: optional schema validation accepts the v0.1 column type subset `bool | int64 | float64 | string | date32` (Arrow `Boolean`, `Int64`, `Float64`, `Utf8`, `Date32`; nullability ignored), checked as header names (or column count when `has_header: false`) plus typed parsing for `std/ingest.csv` and exact column names and types in order for `std/ingest.parquet`; incremental ingestion is deferred to sources that support it.

### 3.2 Data storage
Two distinct concerns, deliberately kept apart ([ADR-0006](adr/0006-store-and-lineage-contracts.md)):

**a) Data storage blocks** — persisting *user data* to external systems is an ordinary, optional stage.
- FR-STO-1: blocks that write datasets/features to object stores, data lakes (Delta, Iceberg), databases and feature stores — v0.2+; `std/storage.mqtt` publishes artifacts to MQTT topics — v0.3 ([ADR-0009](adr/0009-mqtt-blocks-and-serve-transport.md)).

**b) Artifact store** — the internal mechanism through which blocks hand artifacts to each other. Not a block: it sits on every edge of the DAG.
- FR-STO-2: the core defines the immutable `ArtifactStore` contract (plus an auxiliary metadata API for cache entries, run lineage records, and future registry entries) and ships two minimal implementations: `local` (filesystem, default) and `memory` (ephemeral, for tests and throwaway runs) — v0.1 ([ADR-0007](adr/0007-core-contracts-revision.md)).
- FR-STO-3: remote backends (S3, GCS, Azure, …) are optional (`mallops-std` features or plugins) — v0.2.
- FR-STO-4: content addressing (canonical BLAKE3 over manifest and payload, excluding run provenance) — identical content is stored once; enables caching and resume.
- FR-STO-5: garbage collection of artifacts unreachable from retained runs — v0.2.

### 3.3 Preprocessing (data → features)
- FR-PRE-1: tabular transforms (select, filter, impute, scale, encode, split) on Arrow data — v0.1.
- FR-PRE-2: user-defined transforms (Rust closure, WASM, external process).
- FR-PRE-3: fitted transforms are themselves artifacts, so serving reuses the exact same preprocessing.

### 3.4 Training
- FR-TRN-1: native Rust training (linfa for classic ML) — v0.1.
- FR-TRN-2: deep learning via burn/candle — v0.3.
- FR-TRN-3: **external trainer**: run any command/container (Python, R, Julia…) that reads input artifacts and writes outputs into a mallops-managed workspace directory — v0.1. This is how PyTorch, TensorFlow, scikit-learn, XGBoost are supported without Rust bindings.
- FR-TRN-4: hyperparameters, seed and environment recorded in the run lineage.

### 3.5 Model engineering
- FR-ENG-1: hyperparameter search (grid, random; Bayesian as plugin) as a block that fans out training sub-runs — v0.2.
- FR-ENG-2: post-training transforms as blocks: quantization, pruning, format conversion (e.g. torch → onnx) — delegated to the relevant tool.

### 3.6 Model evaluation
- FR-EVL-1: standard metrics (accuracy, precision/recall/F1, ROC-AUC, RMSE, MAE, R²) — v0.1.
- FR-EVL-2: **quality gates**: a pipeline can fail or branch when a metric is below a threshold or worse than the current production model.
- FR-EVL-3: custom metrics via plugins.

### 3.7 Model packaging
- FR-PKG-1: package = model artifact + preprocessing `transform` artifacts + manifest (`flavor`, `io_schema`, requirements) — v0.1.
- FR-PKG-2: optional OCI image output for serving — v0.2.
- FR-PKG-3: artifact signing (Sigstore/cosign or ed25519) — v0.3.

### 3.8 Model versioning
- FR-VER-1: model registry backed by the artifact store: name → versions → stages (`staging`, `production`, `archived`) — v0.2.
- FR-VER-2: every version links to the run lineage that produced it (data hashes, code version, config).

### 3.9 Deploy
- FR-DEP-1: deploy targets as blocks: local process, Docker, Kubernetes manifest, SSH host, edge fleet — v0.2+.
- FR-DEP-2: strategies: replace, canary, shadow — v0.3.
- FR-DEP-3: rollback to a previous registry version in one command.

### 3.10 Model serving
- FR-SRV-1: HTTP/JSON inference server (`mallops serve [package-id]`) running outside the batch DAG, consuming a `package` artifact produced by `mallops run` — v0.1 ([ADR-0007](adr/0007-core-contracts-revision.md)).
- FR-SRV-2: gRPC endpoint, batching, health/readiness probes — v0.2.
- FR-SRV-4: MQTT transport (`serve.transport: mqtt`): read requests from a topic, predict, publish to a response topic; at-least-once, acknowledged after the prediction is published — v0.3 ([ADR-0009](adr/0009-mqtt-blocks-and-serve-transport.md)).
- FR-SRV-3: **backend-agnostic**: the server dispatches to a `Predictor` chosen by the package flavor (linfa, external Python process in v0.1; ort, tract, candle, burn, llama.cpp…). In v0.1 the minimal `Predictor` contract lives in `mallops-std` (shared by `evaluate` and `serve`) and is promoted to `mallops-core` before 1.0 ([ADR-0007](adr/0007-core-contracts-revision.md)).

### 3.11 Monitoring and logging
- FR-MON-1: structured logs (`tracing`) for every run and every request — v0.1.
- FR-MON-2: Prometheus metrics and OpenTelemetry traces — v0.3.
- FR-MON-3: data/prediction drift detection (PSI, KS test) as blocks — v0.3.

### 3.12 Retrain
- FR-RET-1: triggers as blocks: schedule (cron), drift alert, new data, manual — v0.3.
- FR-RET-2: a trigger starts a pipeline run; evaluation gates decide promotion.

### 3.13 Edge
- FR-EDG-1: `mallops-edge` agent: single static binary, cross-compiled for aarch64/armv7/x86_64, inference-only by default (`profile-edge` of `mallops-std`, which includes HTTP and MQTT serving) — v0.4 ([ADR-0008](adr/0008-edge-build-profiles-and-on-device-training.md)).
- FR-EDG-2: local serving with any `Predictor` backend, offline-capable.
- FR-EDG-3: distributed monitoring: nodes buffer metrics/drift stats locally and push them when connected.
- FR-EDG-4: federated learning: central aggregator (FedAvg, FedProx as plugin), nodes train locally, only model updates leave the device; optional secure aggregation and differential privacy as plugins.
- FR-EDG-5: fleet deploy: pull-based model updates with signature verification.
- FR-EDG-6: on-device online learning (`online` feature): a finite DAG (ingest new labelled samples, e.g. from MQTT → train from the current model via an optional `base` input → evaluate gate → package) produces a new immutable package; serving hot-swaps to it only if the gate passes, with rollback and bounded retention ([ADR-0008](adr/0008-edge-build-profiles-and-on-device-training.md)).

### 3.14 Lineage
- FR-LIN-1: the runtime emits lineage events (`RunStarted`, `BlockStarted`, `ArtifactProduced`, `BlockFinished`, `RunFinished`) with RFC 3339 UTC millisecond timestamps (`YYYY-MM-DDTHH:MM:SS.mmmZ`, computed from `SystemTime` without an external date crate) — v0.1. `RunStarted` carries `pipeline_hash` (domain-separated BLAKE3 `"mallops pipeline v1"` over canonical JSON of the topologically ordered nodes: `id`, `use`, `BlockSpec.version`, `config_hash`, `in`, `seed`), `git_commit` (`None` in the core library; populated by `mallops-cli`), and `host` (`os`, `arch`, `hostname` from `HOSTNAME`/`COMPUTERNAME` or `"unknown"`, `mallops_version`). Each started node emits `BlockStarted` (`block` string, `block_version`, `config_hash`, `seed`, `inputs`), one `ArtifactProduced` per output (including cache hits), and `BlockFinished` (`status`, `duration_ms`, `cached: true` on cache hits). Nodes that never start (`Skipped` or `Cancelled` before execution) emit only `BlockFinished` with `duration_ms: 0`. `RunFinished` reports `Cancelled` if the run was cancelled, else `Succeeded` if `RunSummary::is_success()`, else `Failed`.
- FR-LIN-2: events go to pluggable `LineageSink`s; the core ships `JsonSink` (default in `RuntimeConfig::new`), writing `runs/<run-id>/lineage.json` = `{"v":1,"run":"<run-id>","events":[...]}` (events internally tagged with `"event"` in `snake_case`) via `ArtifactStore::put_meta`, rewritten on every event so the file stays valid JSON after a crash — v0.1. `mallops_core::lineage::conformance::check_sink` provides a shared conformance suite reusable by future sinks. SQLite, OpenLineage and MLflow sinks are optional — v0.2+.
- FR-LIN-3: multiple sinks can be active at once via `RuntimeConfig::lineage` (YAML `lineage:` mapping lives in `mallops-cli`); an empty sink list (`lineage: off`) disables lineage entirely while caching still works (it relies on artifact hashes and `cache/<key>.json`, not on lineage). A sink returning `Err` logs a `tracing::warn!` and never fails the run.
- FR-LIN-4: secrets are never written to any sink, log, workdir, or store (verified by `lineage_anti_leak_*`).

## 4. Non-functional requirements

| Id | Requirement |
|---|---|
| NFR-1 Performance | Core overhead per block < 5 ms (median on `MemoryStore`, checked by the ignored test `runtime_nfr1_block_overhead`); independent branches run in parallel. |
| NFR-2 Reproducibility | Same config + same input hashes + same seed ⇒ same artifact hashes for deterministic blocks; unchanged nodes are not re-executed: their outputs are served from the content-hash cache (`NodeSummary.cached`), `cache: false` or `--no-cache` (`RuntimeConfig::no_cache`) bypass it. |
| NFR-3 Portability | Linux, macOS, Windows; x86_64 and aarch64. Edge targets armv7. |
| NFR-4 Security | Secrets only via redacted `${secret:NAME}` references (never written to `config.json`, cache keys, `config_hash`, logs, workdir, store, or lineage; checked by `lineage_anti_leak_*`); `mallops.yaml`, native blocks, and external processes are trusted code — sandboxing applies only to WASM plugins; unsafe code-executing model flavors require explicit opt-in at serve time; signed artifacts from v0.3. |
| NFR-5 Observability | Every run emits structured `tracing` logs (per-node `node` span, warnings on cache or lineage sink errors) and, unless `lineage: off` (`RuntimeConfig::lineage` empty), a lineage record (`runs/<run-id>/lineage.json` by default, rewritten incrementally per event). |
| NFR-6 Stability | Semantic versioning; public API of `mallops-core` frozen at 1.0 (`#[non_exhaustive]` on extensible public types). MSRV documented and tested in CI. |
| NFR-7 Footprint | `mallops-core` compiles without any ML, cloud-SDK, `object_store`, database, or HTTP/gRPC dependency; dependencies are restricted to the explicit allowlist (`tokio`, `serde`, `serde_json`, `blake3`, `tracing`, `thiserror`, `ulid`, `tempfile`) plus feature-gated `yaml` (`yaml_serde`, default) and `schema` (`schemars`, optional) ([ADR-0007](adr/0007-core-contracts-revision.md)). |

### 4.1 Trust model

mallops enforces explicit trust boundaries ([ADR-0007](adr/0007-core-contracts-revision.md)):

- **Pipeline definitions (`mallops.yaml`) are trusted code.** Running `mallops run` is equivalent to executing a build script: native blocks run in-process, external blocks run arbitrary configured commands (`command:`), and `mallops run` spawns `git rev-parse HEAD` (best-effort, in the directory of `mallops.yaml`, stderr discarded; failure yields `None`) to record `RunStarted.git_commit`. By contrast, `mallops validate` and `mallops graph` **never** execute user code, spawn external processes (not even `git`), or create the artifact store directory. `mallops store verify` likewise never runs blocks or spawns external processes.
- **Plugin isolation boundaries.** Only WASM component plugins (post-v0.1) execute inside a capability-restricted sandbox. Native Rust blocks have no isolation; external CLI and gRPC blocks run as host processes. In v0.1, external CLI blocks execute via `argv` (never through a shell), with an explicit allowlisted environment and an isolated workdir, but this is process hygiene rather than a security sandbox.
- **Model flavor deserialization.** Formats such as `pickle`, `joblib`, `sklearn-pickle`, and `torchscript` execute arbitrary code when loaded. `Predictor` implementations declare `executes_code(&self) -> bool`, and `mallops serve` refuses to load code-executing flavors unless `--trust-unsafe-flavors` is explicitly passed. Prefer non-executing formats (`onnx`, `safetensors`, `skops`) and, from v0.3, cryptographically signed packages.
- **Secrets.** `${secret:NAME}` references are resolved on demand through a `SecretProvider` (`MALLOPS_SECRET_<NAME>` environment variable in v0.1) into an opaque `Secret` newtype whose `Debug`/`Display` output is redacted (`"***"`) and which does not implement `Serialize`. Secrets never enter `Context.config`, `config.json`, cache keys, `config_hash`, logs, or lineage sinks.
- **Store integrity.** `ArtifactId` (strict 64 lowercase hex chars) and `RelPath` (relative POSIX UTF-8 path with no `..` or root prefixes) are strictly validated. `ArtifactStore` rejects symlinks and special files, stores payloads read-only on `LocalStore`, and materializes copies on `fetch` so blocks can never mutate stored artifacts in place.

## 5. Pipeline definition (`mallops.yaml`)

```yaml
version: 1
name: churn-classifier

store:                   # artifact store (optional, this is the default)
  kind: local            # local | memory | s3 | gcs | azure | plugin:<name>
  path: ./.mallops

lineage:                 # optional; default is [{ sink: json }]
  - sink: json
  - sink: openlineage
    with: { url: "${env:OPENLINEAGE_URL}" }
  # lineage: off         # disable completely

blocks:
  raw:
    use: std/ingest.csv
    with:
      path: data/customers.csv

  features:
    use: std/preprocess.tabular
    in: { data: raw.dataset }
    with:
      target: churned
      steps:
        - impute: { strategy: median }
        - scale: { method: standard }
        - split: { test: 0.2, seed: 42 }

  model:
    use: std/train.external            # any language, any library
    in: { train: features.train }
    with:
      command: ["python", "train.py"]
      flavor: sklearn-pickle
      params: { n_estimators: 200 }

  eval:
    use: std/evaluate.classification
    in: { model: model.model, test: features.test }
    with:
      gate: { metric: f1, min: 0.80 }

  package:
    use: std/package
    in: { model: model.model, preprocess: features.transform, metrics: eval.metrics }

serve:                   # optional serving config read by `mallops serve` (not a DAG node)
  package: package.package
  host: 127.0.0.1
  port: 8080
```

Rules:
- `blocks:` defines the finite batch DAG executed by `mallops run`. Long-running serving lives in the optional top-level `serve:` section (read by `mallops serve`, never started by `mallops run`).
- `use` references a block by `namespace/name`, where `name` is `stage[.variant]` (e.g. `std/ingest.csv`, `std/package`). Each segment starts with `[a-z0-9]` and contains only `[a-z0-9_-]`; anything else fails as a malformed block reference (`std/…` built-in in v0.1; `plugin/…`, `wasm:path.wasm`, and `grpc:host:port` are reserved for post-v0.1 plugin runtimes and fail with a clear error in v0.1).
- `in` wires named inputs to `<block>.<output>` (split at the first `.`); this defines the DAG. Cycles and self-loops are rejected.
- `with` is block-specific config, validated by `Block::validate_config` (using `serde` in v0.1; optional JSON Schema validation behind the `schema-validate` feature). Each block also supports `cache: false` to bypass caching for that node and an optional `seed:` (u64) for deterministic blocks; the seed is part of the cache key, so changing it re-executes the node. The run-level `--no-cache` (`RuntimeConfig::no_cache`) bypasses the cache for every node.
- Validation (`PipelineConfig::validate`, behind `mallops validate`) runs in one pass, never calls `Block::run`, and reports every problem at once as `Error::Validation(Vec<ValidationError>)`, each with a dotted `path`: `blocks.<id>.use` (unknown block, or a `Stage::Serve` block, which is not a DAG node), `blocks.<id>.in.<port>` (malformed reference, unknown upstream node or output port, undeclared input port, `kind`/`format`/`flavor` mismatch, required input not connected), `blocks.<id>.with[.<sub>]` (errors of `Block::validate_config` and `Block::resolve_ports`, prefixed with `blocks.<id>.with`), `blocks.<id>` (one error per cycle, on its first node by id, listing the nodes) and `serve.package` (not `<node>.<output>`, unknown node or output, or kind other than `package`). A node whose block is unknown, is a `Stage::Serve` block, or whose `with:` fails `validate_config` or `resolve_ports` is unresolved: checks that need its ports (declared input ports, required inputs, output ports, `kind`/`format`/`flavor` compatibility of the edges touching it, `serve.package` pointing at it) are skipped, so one root cause yields one error. When `validate_config` fails, `resolve_ports` is not called. Nodes of the validated pipeline are in topological order, ties broken by node id.
- Execution (`Runtime::run`, behind `mallops run`) runs nodes of the validated pipeline as soon as all their upstream nodes have finished, at most `--jobs` (`RuntimeConfig::max_jobs`, `0` counts as `1`) blocks at a time, and returns a `RunSummary` with one `NodeSummary` per node: `status` (`Succeeded`, `Failed`, `Skipped`, `Cancelled`), `duration`, committed output ids, `cached` (outputs served from the cache), and a skip/cancel `reason` or the `error`. Node outcomes never make `run` return `Err`; only runtime failures do (for example, the run workdir cannot be created). Rules:
  - A node whose block or output commit fails is `Failed`; all its descendants are `Cancelled` without running, with a reason naming the failed node. Independent branches finish.
  - `BlockOutcome::Skipped` marks the node `Skipped` (outputs it still returns are committed). A consumer is `Skipped` without running when any **required** input comes from a `Skipped` node or was not committed, even if its other inputs succeeded (fan-in). An optional input without a committed artifact is simply absent. If a node has both a failed/cancelled and a skipped parent, it is `Cancelled`.
  - Cancellation is cooperative: once the run's `CancelToken` fires, no new node starts (`Cancelled`, reason `run cancelled`); a running block that returns `Error::Cancelled` is `Cancelled` (not `Failed`). Nothing is aborted forcibly.
  - Output drafts are validated before any `store.put` (errors at `out.<port>`): every draft names a declared output port and satisfies its `kind`/`format`/`flavor`; a `Produced` outcome provides every required output; `File`/`Dir` payloads canonicalize inside the node workdir to a regular file / a directory, and the canonical path is what the store receives. The runtime sets `parents` to the node's input ids (sorted, deduplicated). Payload content (nested symlinks, special files inside a `Dir`) is checked by the store during `put`; if that fails the node is `Failed`, and outputs of the same node committed before stay in the store as unreferenced artifacts (immutable, harmless, removed by a later GC).
  - Cache (NFR-2): after the node gets its job slot, the cache is bypassed when `cache: false` is set or `no_cache` is on (`Block::cache_policy` is then not called); otherwise `Block::cache_policy` is consulted (an error fails the node; `extra_hash` must not depend on secrets) and `Uncacheable` bypasses it too. Otherwise a hit (all recorded output artifacts still in the store) marks the node `Succeeded` with `cached: true`, its outputs taken from the entry, and the block is not run; a miss runs it. Only `Succeeded` nodes are cached (`Skipped`, `Failed`, `Cancelled` always re-run). Failing to write the entry, or a store error while reading it (treated as a miss), is a warning, the node stays `Succeeded`. A rerun re-executes every node that did not succeed, plus the descendants whose inputs changed.
  - Each run gets a temporary workdir under `RuntimeConfig::workdir_root` (which may be relative) with one subdirectory per node; `ctx.workdir()` is always absolute and canonical. The run workdir is removed when the run ends.
- Values may reference environment variables `${env:NAME}` (resolved into `config` and included in `config_hash`; `NAME` is `[A-Za-z0-9_]+`; interpolated only in string values of each block's `with`, of lineage sink `with`, and in `store.path`, never in keys, `in`, `seed` or elsewhere; replaced text is not re-scanned; no escape syntax; a missing variable is an error naming the file and the block, sink or `store.path`) and secrets `${secret:NAME}` (same `NAME` charset, validated at load; kept literally in the config, never resolved at load time; resolved on demand via `MALLOPS_SECRET_<NAME>` in v0.1 into a redacted `Secret`, never written to `config.json`, cache keys, logs, or lineage).
- Unknown keys are rejected (`deny_unknown_fields`); `lineage` accepts only `off` or a list of sinks; `version` must be `1`; duplicate block ids (or `in` ports) are rejected. Syntax and schema errors are reported as `<file>:<line>:<column>: <message>`.
- Store and lineage backend mapping in the v0.1 CLI (`check_backends`, enforced by `validate`, `run`, and `store verify`):
  - `store.kind` accepts `local` (`LocalStore` at `<yaml-dir>/<store.path>`, where `<yaml-dir>` is the parent directory of the `mallops.yaml` path, defaulting to `.` when empty) and `memory` (`MemoryStore`). Any other `store.kind` fails with an error.
  - `lineage: off` disables lineage (`RuntimeConfig::lineage = vec![]`); otherwise every configured sink must have `sink: json` (`JsonSink` on the configured store). Any other sink name fails with an error.
- Built-in blocks in `mallops-cli`: `core/noop` (version `1.0.0`, `Stage::Custom`, optional input `in: blob`, output `out: blob` with `format: noop` and empty payload; test affordances `fail: "<message>"` and `skip: "<reason>"` in `with:`) plus all feature-enabled standard blocks registered via `mallops_std::register_std(&mut registry)` (in v0.1 so far: `std/ingest.csv`, `std/ingest.parquet`).
- Latest run pointer (`runs/latest.json`): after `mallops run`, if and only if every node is `Succeeded`, the CLI writes `runs/latest.json` via `ArtifactStore::put_meta` (`meta/runs/latest.json` on `LocalStore`) as `{"v":1,"pipeline":"<name>","run":"<run-id>","outputs":{"<node>":{"<port>":"<artifact-id>"}}}` (deterministic key order via `BTreeMap`), so `mallops serve` can resolve `serve.package` without an explicit id. If any node is `Skipped` (with no failed/cancelled nodes), `mallops run` leaves any previous `runs/latest.json` unchanged, prints `latest.json not updated: <S> skipped`, and exits `0`. A failed run leaves any previous `runs/latest.json` unchanged and exits `1` (`error: run failed`).
- A JSON Schema for `mallops.yaml` can be generated via the optional `schema` feature (`schemars`) for editor completion.

### 5.1 CLI (`mallops`)

Global flags: `--version` (`-V`), `-v` / `--verbose` (stderr `tracing` logs at `DEBUG` instead of `WARN`), and `-f` / `--file <PATH>` (default `mallops.yaml`).

Exit codes: `0` on success, `1` on configuration, validation, runtime, or store verification error, `2` on CLI argument/usage error (`clap`).

| Command | Behavior |
|---|---|
| `mallops init [DIR]` | Writes a minimal runnable `mallops.yaml` (two `core/noop` nodes, `hello` → `world`) into `DIR` (default `.`, creating missing parent directories); fails if `mallops.yaml` already exists. |
| `mallops validate` | Parses and validates `mallops.yaml` (including `serve:` if present) and checks supported store/lineage backends without executing blocks or spawning processes. Prints `ok: <name> (<N> blocks)` on stdout, or every validation error (`error: <path>: <message>`) on stderr. |
| `mallops run [--jobs N] [--no-cache]` | Validates and executes the batch DAG (`blocks:` only; ignores `serve:`). Changes the working directory to the directory of `mallops.yaml` so relative paths in block configs (such as `std/ingest.*` `path`) resolve against the pipeline directory, matching `store.path`. `--jobs N` overrides `RuntimeConfig::max_jobs` (default: available parallelism; `0` counts as `1`); `--no-cache` bypasses the cache. Populates `git_commit` via best-effort `git rev-parse HEAD` in the `mallops.yaml` directory. Prints one line per node in topological order (`<node> <status>[ (cached)] <ms>ms[: <reason/error>]`) followed by `run <run-id>: <E> executed, <C> cached, <F> failed, <S> skipped, <X> cancelled` (`executed` = non-cached `Succeeded`). Writes `runs/latest.json` only when every node is `Succeeded` (if any node is `Skipped`, prints `latest.json not updated: <S> skipped` and exits `0`); exits `1` on failure. |
| `mallops graph [--format mermaid\|dot]` | Prints the validated DAG in topological order as Mermaid (`flowchart TD`, default) or Graphviz DOT (`digraph`), with edges labeled by input port name, without executing blocks or spawning processes. |
| `mallops blocks list` | Prints registered blocks (`ID`, `VERSION`, `STAGE`, `INPUTS`, `OUTPUTS`; optional inputs suffixed with `?`, empty port list shown as `-`), sorted by block id. |
| `mallops store verify <ID>` | Parses `<ID>` as an `ArtifactId`, checks backends, opens the configured store (for `kind: local`, fails with `store not found: <path>` without creating it if absent), and calls `ArtifactStore::verify`. Prints `ok: <id>` on stdout on success; exits `1` on invalid id, missing store/artifact, or corruption/tampering. |

## 6. Artifact contract

```
<store>/artifacts/<blake3-hash>/
├── manifest.json
└── payload/…          # any files, opaque to the core
```

```json
{
  "schema_version": 1,
  "kind": "model",
  "format": "pickle",
  "flavor": "sklearn-pickle",
  "io_schema": { "inputs": [{"name": "age", "dtype": "f64"}], "outputs": [{"name": "churned", "dtype": "f64"}] },
  "metadata": { "framework": "scikit-learn", "framework_version": "1.6.0" },
  "parents": ["<hash>", "<hash>"]
}
```

Per-run provenance (`run`, `node`) is **not** part of `manifest.json`; it is recorded in `LineageEvent::ArtifactProduced` so identical deterministic runs produce identical `ArtifactId`s ([ADR-0007](adr/0007-core-contracts-revision.md)). Cryptographic signatures (`FR-PKG-3`, v0.3) are stored alongside artifacts rather than inside `manifest.json`.

**Canonical `ArtifactId` encoding (`v1`):**
1. Initialize a BLAKE3 keyed hasher with `blake3::derive_key("mallops artifact v1")`.
2. Serialize `Manifest` as canonical compact JSON (object keys sorted by UTF-8 byte order, numbers in `serde_json` format — not RFC 8785 — `io_schema: null` omitted as absent, and `parents` sorted lexicographically and deduplicated) and hash `u64::to_le_bytes(len) || manifest_json_bytes`.
3. Collect all payload files via validated `RelPath` (relative UTF-8 POSIX paths with `/` separators, no `..`, no symlinks) sorted bytewise; hash `u64::to_le_bytes(file_count)` followed by `len(path) || path || len(content) || blake3(content)` for each file (all lengths as `u64` little-endian).
4. Encode the 32-byte digest as 64 lowercase hexadecimal characters.

| `kind` | Typical `format` values | Notes |
|---|---|---|
| `dataset` | `parquet`, `arrow-ipc`, `csv`, `json`, `images`, `tfrecord`, any | Arrow/Parquet is the interchange default for std blocks only ([ADR-0002](adr/0002-arrow-default-interchange.md)): `std/ingest.*` emit `format: parquet` with a single uncompressed `payload/data.parquet` file (`created_by: "mallops-std"`, deterministic per `mallops-std` version, source schema metadata stripped) |
| `features` | as dataset | preprocessed feature splits |
| `transform` | `json`, `bin`, `pickle`, any | fitted preprocessor (scaler, imputer, encoder…) reused by `package` and `serve` |
| `model` | `onnx`, `safetensors`, `torchscript`, `pickle`, `joblib`, `gguf`, `xgboost-json`, `savedmodel`, any | `flavor` selects the predictor |
| `metrics` | `json` | key → number map + optional plots |
| `package` | `dir`, `oci` | model + transforms + manifest |
| `blob` | any | escape hatch |

**Model flavor code-execution safety:**

| Flavor / format | Executes code on load? | Serving default |
|---|---|---|
| `onnx`, `safetensors`, `linfa-*`, `xgboost-json`, `gguf`, `skops` | No | Allowed |
| `pickle`, `joblib`, `sklearn-pickle`, `torchscript` | **Yes** | Refused unless `--trust-unsafe-flavors` |

Inputs and outputs of blocks declare the accepted `kind` and optionally accepted `format`s / `flavor`s; `mallops validate` checks compatibility before running (using `Block::resolve_ports` when ports depend on `with:` config); an output is wired into an input only if its `kind` is equal and every `format`/`flavor` it may produce is accepted (an empty accepted list means any; an unconstrained output, i.e. an empty list, does not satisfy a constrained input). Error paths are listed in [§5](#5-pipeline-definition-mallopsyaml).

## 7. Roadmap

| Version | Scope |
|---|---|
| **v0.1** | `mallops-core` (Block trait, canonical artifacts, immutable `ArtifactStore` + local/memory + auxiliary metadata, explicit `BlockRegistry`, lineage events + JSON sink, DAG runtime, YAML), CLI (`init`, `validate`, `run`, `graph`, `blocks list`, `store verify`, `serve`), `mallops-std` (`Predictor` contract, ingest csv/parquet, preprocess tabular, train linfa + external, evaluate classification/regression, package, HTTP serving) |
| v0.2 | remote store backends, SQLite/OpenLineage/MLflow lineage sinks, data storage blocks, model registry & versioning, hyperparameter search, deploy local/docker, gRPC serving, OCI packaging |
| v0.3 | monitoring (Prometheus/OTel), drift detection, retrain triggers, DL training (burn/candle), signing, canary/shadow deploy, MQTT ingest/publish blocks and MQTT serving transport |
| v0.4 | `mallops-edge` (edge build profiles), on-device online learning, federated learning, fleet deploy, distributed monitoring |
| v1.0 | stable public API, plugin protocol v1 frozen |

## 8. Open questions

- Plugin discovery/distribution: crates.io-style index vs. OCI registry for WASM/gRPC plugins.
- Python SDK for writing external blocks (thin package implementing the protocol) — likely v0.2.
- Which lineage sink becomes the recommended default for querying (SQLite vs. OpenLineage backend) once registries need search.
