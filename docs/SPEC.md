# mallops — Technical Specification

> Status: **Draft** · Target: v0.1 · Last update: 2026-11-21

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
- FR-ING-1: read from local files (CSV, JSON/NDJSON, Parquet, arbitrary files) — v0.1.
- FR-ING-2: read from HTTP(S) and object stores (S3, GCS, Azure) — v0.2.
- FR-ING-3: read from databases (PostgreSQL, SQLite) and streams (Kafka, MQTT) — v0.3+.
- FR-ING-4: optional schema validation; ingest is incremental when the source supports it.

### 3.2 Data storage
Two distinct concerns, deliberately kept apart ([ADR-0006](adr/0006-store-and-lineage-contracts.md)):

**a) Data storage blocks** — persisting *user data* to external systems is an ordinary, optional stage.
- FR-STO-1: blocks that write datasets/features to object stores, data lakes (Delta, Iceberg), databases and feature stores — v0.2+.

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
- FR-SRV-3: **backend-agnostic**: the server dispatches to a `Predictor` chosen by the package flavor (linfa, external Python process in v0.1; ort, tract, candle, burn, llama.cpp…). In v0.1 the minimal `Predictor` contract lives in `mallops-std` (shared by `evaluate` and `serve`) and is promoted to `mallops-core` before 1.0 ([ADR-0007](adr/0007-core-contracts-revision.md)).

### 3.11 Monitoring and logging
- FR-MON-1: structured logs (`tracing`) for every run and every request — v0.1.
- FR-MON-2: Prometheus metrics and OpenTelemetry traces — v0.3.
- FR-MON-3: data/prediction drift detection (PSI, KS test) as blocks — v0.3.

### 3.12 Retrain
- FR-RET-1: triggers as blocks: schedule (cron), drift alert, new data, manual — v0.3.
- FR-RET-2: a trigger starts a pipeline run; evaluation gates decide promotion.

### 3.13 Edge
- FR-EDG-1: `mallops-edge` agent: single static binary, cross-compiled for aarch64/armv7/x86_64 — v0.4.
- FR-EDG-2: local serving with any `Predictor` backend, offline-capable.
- FR-EDG-3: distributed monitoring: nodes buffer metrics/drift stats locally and push them when connected.
- FR-EDG-4: federated learning: central aggregator (FedAvg, FedProx as plugin), nodes train locally, only model updates leave the device; optional secure aggregation and differential privacy as plugins.
- FR-EDG-5: fleet deploy: pull-based model updates with signature verification.

### 3.14 Lineage
- FR-LIN-1: the runtime emits lineage events (`RunStarted`, `BlockStarted`, `ArtifactProduced`, `BlockFinished`, `RunFinished`) carrying config hash, block versions, input/output artifact ids, seed, git commit, timings, host info — v0.1.
- FR-LIN-2: events go to pluggable **lineage sinks**; the core ships `json` (one `lineage.json` per run in the store, default) — v0.1. SQLite, OpenLineage and MLflow sinks are optional — v0.2+.
- FR-LIN-3: multiple sinks can be active at once; `lineage: off` disables lineage entirely (caching still works, since it relies on artifact hashes, not on lineage).
- FR-LIN-4: secrets are never written to any sink.

## 4. Non-functional requirements

| Id | Requirement |
|---|---|
| NFR-1 Performance | Core overhead per block < 5 ms; independent branches run in parallel. |
| NFR-2 Reproducibility | Same config + same input hashes + same seed ⇒ same artifact hashes for deterministic blocks; unchanged nodes are skipped (cache). |
| NFR-3 Portability | Linux, macOS, Windows; x86_64 and aarch64. Edge targets armv7. |
| NFR-4 Security | Secrets only via redacted `${secret:NAME}` references (never written to `config.json`, cache keys, logs, or lineage); `mallops.yaml`, native blocks, and external processes are trusted code — sandboxing applies only to WASM plugins; unsafe code-executing model flavors require explicit opt-in at serve time; signed artifacts from v0.3. |
| NFR-5 Observability | Every run emits structured logs and, unless `lineage: off`, a lineage record. |
| NFR-6 Stability | Semantic versioning; public API of `mallops-core` frozen at 1.0 (`#[non_exhaustive]` on extensible public types). MSRV documented and tested in CI. |
| NFR-7 Footprint | `mallops-core` compiles without any ML, cloud-SDK, `object_store`, database, or HTTP/gRPC dependency; dependencies are restricted to the explicit allowlist (`tokio`, `serde`, `serde_json`, `blake3`, `tracing`, `thiserror`, `ulid`, `tempfile`) plus feature-gated `yaml` (`yaml_serde`, default) and `schema` (`schemars`, optional) ([ADR-0007](adr/0007-core-contracts-revision.md)). |

### 4.1 Trust model

mallops enforces explicit trust boundaries ([ADR-0007](adr/0007-core-contracts-revision.md)):

- **Pipeline definitions (`mallops.yaml`) are trusted code.** Running `mallops run` is equivalent to executing a build script: native blocks run in-process and external blocks run arbitrary configured commands (`command:`). By contrast, `mallops validate` and `mallops graph` **never** execute user code or spawn external processes.
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
- `with` is block-specific config, validated by `Block::validate_config` (using `serde` in v0.1; optional JSON Schema validation behind the `schema-validate` feature). Each block also supports `cache: false` to bypass caching and an optional `seed:` (u64) for deterministic blocks.
- Validation (`PipelineConfig::validate`, behind `mallops validate`) runs in one pass, never calls `Block::run`, and reports every problem at once as `Error::Validation(Vec<ValidationError>)`, each with a dotted `path`: `blocks.<id>.use` (unknown block, or a `Stage::Serve` block, which is not a DAG node), `blocks.<id>.in.<port>` (malformed reference, unknown upstream node or output port, undeclared input port, `kind`/`format`/`flavor` mismatch, required input not connected), `blocks.<id>.with[.<sub>]` (errors of `Block::validate_config` and `Block::resolve_ports`, prefixed with `blocks.<id>.with`), `blocks.<id>` (one error per cycle, on its first node by id, listing the nodes) and `serve.package` (not `<node>.<output>`, unknown node or output, or kind other than `package`). A node whose block is unknown or whose `with:` fails validation does not produce follow-up port errors on its edges. Nodes of the validated pipeline are in topological order, ties broken by node id.
- Values may reference environment variables `${env:NAME}` (resolved into `config` and included in `config_hash`; `NAME` is `[A-Za-z0-9_]+`; interpolated only in string values of each block's `with`, of lineage sink `with`, and in `store.path`, never in keys, `in`, `seed` or elsewhere; replaced text is not re-scanned; no escape syntax; a missing variable is an error naming the file and the block, sink or `store.path`) and secrets `${secret:NAME}` (same `NAME` charset, validated at load; kept literally in the config, never resolved at load time; resolved on demand via `MALLOPS_SECRET_<NAME>` in v0.1 into a redacted `Secret`, never written to `config.json`, cache keys, logs, or lineage).
- Unknown keys are rejected (`deny_unknown_fields`); `lineage` accepts only `off` or a list of sinks; `version` must be `1`; duplicate block ids (or `in` ports) are rejected. Syntax and schema errors are reported as `<file>:<line>:<column>: <message>`.
- A JSON Schema for `mallops.yaml` can be generated via the optional `schema` feature (`schemars`) for editor completion.

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
| `dataset` | `parquet`, `arrow-ipc`, `csv`, `json`, `images`, `tfrecord`, any | Arrow/Parquet is the interchange default for std blocks only |
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

Inputs and outputs of blocks declare the accepted `kind` and optionally accepted `format`s / `flavor`s; `mallops validate` checks compatibility before running (using `Block::resolve_ports` when ports depend on `with:` config); an output is wired into an input only if its `kind` is equal and every `format`/`flavor` it may produce is accepted (an empty accepted list means any). Error paths are listed in [§5](#5-pipeline-definition-mallopsyaml).

## 7. Roadmap

| Version | Scope |
|---|---|
| **v0.1** | `mallops-core` (Block trait, canonical artifacts, immutable `ArtifactStore` + local/memory + auxiliary metadata, explicit `BlockRegistry`, lineage events + JSON sink, DAG runtime, YAML), CLI (`init`, `validate`, `run`, `graph`, `blocks list`, `serve`), `mallops-std` (`Predictor` contract, ingest csv/parquet, preprocess tabular, train linfa + external, evaluate classification/regression, package, HTTP serving) |
| v0.2 | remote store backends, SQLite/OpenLineage/MLflow lineage sinks, data storage blocks, model registry & versioning, hyperparameter search, deploy local/docker, gRPC serving, OCI packaging |
| v0.3 | monitoring (Prometheus/OTel), drift detection, retrain triggers, DL training (burn/candle), signing, canary/shadow deploy |
| v0.4 | `mallops-edge`, federated learning, fleet deploy, distributed monitoring |
| v1.0 | stable public API, plugin protocol v1 frozen |

## 8. Open questions

- Plugin discovery/distribution: crates.io-style index vs. OCI registry for WASM/gRPC plugins.
- Python SDK for writing external blocks (thin package implementing the protocol) — likely v0.2.
- Which lineage sink becomes the recommended default for querying (SQLite vs. OpenLineage backend) once registries need search.
