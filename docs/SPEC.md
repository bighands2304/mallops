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
2. **Everything is a block.** Every lifecycle step, including storage, deploy and retrain triggers, is a replaceable block with declared inputs and outputs.
3. **Small core, optional everything else.** The core has no ML dependency. Standard blocks live behind Cargo feature flags.
4. **Reproducible by default.** Content-addressed artifacts, locked configs, recorded seeds and environment.
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
- FR-STO-2: the core defines the `ArtifactStore` contract and ships two minimal implementations: `local` (filesystem, default) and `memory` (ephemeral, for tests and throwaway runs) — v0.1.
- FR-STO-3: remote backends (S3, GCS, Azure, …) are optional (`mallops-std` features or plugins) — v0.2.
- FR-STO-4: content addressing (BLAKE3) — identical content is stored once; enables caching and resume.
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
- FR-PKG-1: package = model artifact + preprocessing artifacts + manifest (flavor, signature, requirements) — v0.1.
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
- FR-SRV-1: HTTP/JSON inference server (`mallops serve`) — v0.1.
- FR-SRV-2: gRPC endpoint, batching, health/readiness probes — v0.2.
- FR-SRV-3: **backend-agnostic**: the server dispatches to a `Predictor` chosen by the package flavor (ort, tract, candle, burn, linfa, llama.cpp, external Python process…).

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
| NFR-4 Security | No secrets in config files (env/secret-provider references only); plugin sandboxing via WASM; signed artifacts from v0.3. |
| NFR-5 Observability | Every run emits structured logs and a lineage record. |
| NFR-6 Stability | Semantic versioning; public API of `mallops-core` frozen at 1.0. MSRV documented and tested in CI. |
| NFR-7 Footprint | `mallops-core` compiles without any ML, cloud-SDK or `object_store` dependency; only `tokio`, `serde`, `blake3`, `tracing`. |

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

  serve:
    use: std/serve.http
    in: { package: package.package }
    with: { port: 8080 }
```

Rules:
- `use` references a block by `namespace/name` (`std/…` built-in, `plugin/…` installed plugin, `wasm:path.wasm`, `grpc:host:port`).
- `in` wires named inputs to `<block>.<output>`; this defines the DAG. Cycles are rejected.
- `with` is block-specific config, validated against the block's JSON schema.
- Values may reference environment variables `${env:NAME}` and secrets `${secret:NAME}`; secrets are never written to lineage.
- A JSON Schema for the file is generated from Rust types (`schemars`) and published for editor completion.

## 6. Artifact contract

```
<store>/artifacts/<blake3-hash>/
├── manifest.json
└── payload/…          # any files, opaque to the core
```

```json
{
  "kind": "model",
  "format": "pickle",
  "flavor": "sklearn-pickle",
  "signature": { "inputs": [{"name": "age", "dtype": "f64"}], "outputs": [{"name": "churned", "dtype": "f64"}] },
  "metadata": { "framework": "scikit-learn", "framework_version": "1.6.0" },
  "produced_by": { "run": "01J…", "block": "model" },
  "parents": ["<hash>", "<hash>"]
}
```

| `kind` | Typical `format` values | Notes |
|---|---|---|
| `dataset` | `parquet`, `arrow-ipc`, `csv`, `json`, `images`, `tfrecord`, any | Arrow/Parquet is the interchange default for std blocks only |
| `features` | as dataset | plus optional fitted `transform` artifact |
| `model` | `onnx`, `safetensors`, `torchscript`, `pickle`, `joblib`, `gguf`, `xgboost-json`, `savedmodel`, any | `flavor` selects the predictor |
| `metrics` | `json` | key → number map + optional plots |
| `package` | `dir`, `oci` | model + transforms + manifest |
| `blob` | any | escape hatch |

Inputs and outputs of blocks declare the accepted `kind` and optionally accepted `format`s / `flavor`s; `mallops validate` checks compatibility before running.

## 7. Roadmap

| Version | Scope |
|---|---|
| **v0.1** | `mallops-core` (Block trait, artifacts, `ArtifactStore` + local/memory, lineage events + JSON sink, DAG runtime, YAML), CLI (`init`, `validate`, `run`, `graph`, `blocks list`, `serve`), std blocks: ingest csv/parquet, preprocess tabular, train linfa + external, evaluate classification/regression, package, serve http |
| v0.2 | remote store backends, SQLite/OpenLineage/MLflow lineage sinks, data storage blocks, model registry & versioning, hyperparameter search, deploy local/docker, gRPC serving, OCI packaging |
| v0.3 | monitoring (Prometheus/OTel), drift detection, retrain triggers, DL training (burn/candle), signing, canary/shadow deploy |
| v0.4 | `mallops-edge`, federated learning, fleet deploy, distributed monitoring |
| v1.0 | stable public API, plugin protocol v1 frozen |

## 8. Open questions

- Plugin discovery/distribution: crates.io-style index vs. OCI registry for WASM/gRPC plugins.
- Python SDK for writing external blocks (thin package implementing the protocol) — likely v0.2.
- Which lineage sink becomes the recommended default for querying (SQLite vs. OpenLineage backend) once registries need search.
