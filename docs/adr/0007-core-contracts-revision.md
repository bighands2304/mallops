# 0007 — Core contracts revision before v0.1 implementation

- Status: Accepted · 2026-11-21
- Partially supersedes: [0003](0003-plugins-wasm-and-grpc.md) (compile-time `inventory` registration; plugin isolation table), [0004](0004-agnostic-artifacts-and-flavors.md) (`Predictor` placement in v0.1; manifest fields)
- Refines: [0001](0001-modular-workspace.md), [0006](0006-store-and-lineage-contracts.md)

## Context

A pre-implementation design review of [SPEC.md](../SPEC.md), [ARCHITECTURE.md](../ARCHITECTURE.md), and the existing ADRs uncovered eight structural defects in `mallops-core` contracts that would cause cascading breaking changes as soon as v0.1 development starts:

1. Including `produced_by.run` inside the hashed `Manifest` breaks content addressing, deduplication, and caching; the hash framing itself was unspecified.
2. `NFR-7` restricted `mallops-core` to four crates while core features require additional infrastructure crates, `serde_yaml` is archived upstream, and `schemars::Schema` leaked into public signatures.
3. `ArtifactStore` lacked real immutability on `LocalStore`, validated ID/path types, and an API for auxiliary metadata (cache index, `lineage.json`, model registry).
4. Core trait and struct signatures (`Block`, `BlockSpec`, `Context`, `Manifest`, `LineageEvent`, `Error`) were missing versioning, skip outcomes, cancellation, secret isolation, runtime-managed output commits, `#[non_exhaustive]`, `ArtifactKind::Transform`, and an unambiguous I/O schema field name.
5. `serve` was modeled both as a batch DAG node and as a standalone CLI command.
6. `Predictor` was placed in `mallops-core` before being validated against real backends and out of dependency order with `evaluate`.
7. No explicit trust model existed for executable YAML pipelines, code-executing model flavors, or actual plugin isolation boundaries.
8. Compile-time `inventory` registration is fragile under LTO/cross-compilation, breaks test isolation, and adds unnecessary complexity for v0.1.

## Decisions

### D1 — Canonical `ArtifactId` encoding without `produced_by`

**Decision:**
- Remove `produced_by` (and all run-specific or timestamp fields) from the content-addressed `Manifest`. Per-run provenance (`run`, `node`, `artifact`) is recorded exclusively in `LineageEvent::ArtifactProduced`.
- Define a canonical, cross-platform encoding for `ArtifactId` using BLAKE3 with domain separation (`blake3::derive_key("mallops artifact v1")`):
  1. Serialize the canonical `Manifest` (`schema_version`, `kind`, `format`, `flavor`, `io_schema`, `parents`, `metadata`) as canonical JSON with sorted object keys (`BTreeMap` / RFC 8785) and `parents` sorted lexicographically by `ArtifactId`; feed `u64::to_le_bytes(manifest_bytes.len()) || manifest_bytes`.
  2. Feed `u64::to_le_bytes(file_count)` followed by each payload file sorted bytewise by its validated POSIX UTF-8 relative path (`RelPath`, `/` separators): `len(path) || path || len(content) || blake3(content)` (all lengths as `u64` little-endian).
- Represent `ArtifactId` as a 32-byte BLAKE3 digest formatted as 64 lowercase hexadecimal characters.

**Rationale:**
Because `produced_by.run` changes on every execution, hashing it made two identical deterministic runs produce different `ArtifactId`s, breaking `NFR-2`, `FR-STO-4` (content deduplication), and downstream caching. Domain separation, length-prefixed framing, canonical JSON key ordering, and POSIX path normalization prevent hash-boundary collisions and guarantee identical digests across Linux, macOS, and Windows.

### D2 — Explicit `mallops-core` dependency allowlist, YAML parser, and schema isolation

**Decision:**
- Amend `NFR-7` to distinguish **forbidden dependency categories** (no ML/tensor/dataframe libraries, no cloud SDKs, no `object_store`, no database drivers, no HTTP/gRPC servers) from an **explicit core allowlist**:
  - Default core dependencies: `tokio`, `serde`, `serde_json`, `blake3`, `tracing`, `thiserror`, `ulid`, `tempfile`.
  - Feature-gated dependencies:
    - `yaml` (enabled by default): `yaml_serde` (`0.10`).
    - `schema` (disabled by default): `schemars` for generating JSON Schema documents for editor completion.
    - `schema-validate` (disabled by default): optional JSON Schema validation engine when requested, rather than compiled into default builds.
- Remove `schemars::Schema` from `BlockSpec`'s public API; use `Option<serde_json::Value>` instead. In v0.1, `with:` configuration is validated by `Block::validate_config(&self, config: &serde_json::Value)` (typically via `serde::Deserialize`).
- Avoid adding `async-trait` to `mallops-core` (use `Pin<Box<dyn Future + Send + '_>>` or a dual native/dyn trait pattern).

**Rationale — YAML parser selection:**
- `serde_yaml` is archived and unmaintained upstream and depends on archived `unsafe-libyaml` (`RUSTSEC-2024-0449`), failing `cargo deny`.
- `serde_yml` is archived and affected by an unsoundness advisory (`RUSTSEC-2025-0068`).
- `serde-saphyr` (`MIT OR Apache-2.0`, pure Rust with `#![forbid(unsafe_code)]`, configurable resource budgets against alias bombs) is a strong future candidate, but requires MSRV Rust 1.89, which exceeds the workspace MSRV of Rust 1.85. Because the YAML parser is not exposed in `mallops-core`'s public API, migrating to `serde-saphyr` after an MSRV bump will be non-breaking.
- `serde_norway` is a single-maintainer fork, whereas `yaml_serde` is stewarded by The YAML Organization (`yaml/yaml-serde`).
- **`yaml_serde` (`0.10`)** is actively maintained by The YAML Organization, dual-licensed `MIT OR Apache-2.0` (permitted by `deny.toml`), supports MSRV 1.82 (compatible with 1.85), replaces `unsafe-libyaml` with `libyaml-rs`, has no active RustSec advisories, enforces recursion depth limits, and reports line/column locations on syntax errors. Parser tests must still explicitly cover YAML alias bombs and deeply nested inputs.

Keeping `schemars` out of public types protects `mallops-core` semver stability at 1.0 and keeps `BlockSpec` compatible with dynamic WASM/gRPC plugins.

### D3 — Immutable `ArtifactStore`, validated `ArtifactId`/`RelPath`, and auxiliary metadata API

**Decision:**
- Introduce validated newtypes in `mallops-core`:
  - `ArtifactId`: 32-byte digest with strict `FromStr` (exactly 64 lowercase hex chars) and no unchecked string constructor.
  - `RelPath`: validated relative path (non-empty UTF-8, normalized `/` separators, rejecting `..`, leading `/`, Windows drive letters/backslashes, NUL bytes, and empty segments).
- Harden `ArtifactStore` semantics:
  - `put(&self, manifest: Manifest, payload: PayloadSource<'_>) -> Result<ArtifactId>` accepts a structured `PayloadSource` (`File`, `Dir`, `Bytes`, or multi-file map), writes atomically (temp directory on the same filesystem + idempotent rename), rejects symlinks and special files, and verifies canonical paths stay inside the root. `LocalStore::put` marks stored files read-only (`0444`) and directories (`0555`) where supported by the OS.
  - `get(&self, id: &ArtifactId) -> Result<Option<Artifact>>` avoids `exists` + `get` TOCTOU races.
  - `fetch(&self, id: &ArtifactId, dest: &Path) -> Result<PathBuf>` has uniform semantics across all backends: it always materializes the payload into `dest` (read-only copy/materialization on `LocalStore`, download + integrity check on remote stores) and never returns a writable path inside the store itself.
  - `verify(&self, id: &ArtifactId) -> Result<()>` recomputes the canonical `ArtifactId` to detect corruption or tampering.
- Add an auxiliary key-value metadata API on the store (or companion `MetaStore` contract implemented by every backend): `put_meta`, `get_meta`, `list_meta`, and `delete_meta` keyed by `RelPath`. Caching (`cache/<key>.json`), `JsonSink` (`runs/<run-id>/lineage.json`), and the v0.2 model registry (`registry/<model>/<version>.json`) all persist through this API.

**Rationale:**
Making `fetch` a no-op on `LocalStore` allowed external blocks to mutate stored payloads in place, silently corrupting content addressing and cache integrity. Validated `ArtifactId`/`RelPath` types and symlink rejection close path-traversal and file-exfiltration vectors across `LocalStore` and external `outputs.json`. An explicit auxiliary metadata API ensures caching and `JsonSink` work identically across `LocalStore`, `MemoryStore`, and future remote stores without hardcoded filesystem paths.

### D4 — Core signatures (`BlockSpec`, `Block`, `Context`, `Error`, `#[non_exhaustive]`, `Transform`, `io_schema`)

**Decision:**
- **`BlockSpec`**: add `version: Cow<'static, str>`, use `Cow<'static, str>` for `id` and port/format/flavor strings, replace `config_schema` with `Option<serde_json::Value>`, return `&BlockSpec` from `Block::spec(&self)`, and add optional hooks `Block::validate_config(&self, config: &serde_json::Value)` and `Block::resolve_ports(&self, config: &serde_json::Value)` for config-driven blocks (such as `std/train.external`).
- **`BlockOutcome` & `cache_policy`**: `Block::run` returns `Result<BlockOutcome>` where `BlockOutcome` has `Produced(OutputDrafts)` and `Skipped { reason: String, outputs: OutputDrafts }`. Any downstream node with a required input from a skipped node is automatically skipped with `Status::Skipped` in lineage. Add `Block::cache_policy(&self, ctx: &Context) -> Result<CachePolicy>` (`Deterministic { extra_hash: Option<Digest> }` | `Uncacheable`) so source/external blocks invalidate when local files or scripts change.
- **Runtime-owned output commits**: blocks produce output drafts (`OutputDraft { manifest_draft, payload }`); the **runtime** commits them to `ArtifactStore`, injects `Manifest.parents` from the node's actual input `ArtifactId`s, validates outputs against resolved ports, and emits `LineageEvent::ArtifactProduced`. Blocks never call `store.put()` directly for outputs.
- **`Context`**:
  - Remove `pub span: tracing::Span` (the runtime wraps `block.run(...).instrument(node_span).await`).
  - Add `cancel: CancelToken` (cooperative cancellation built on `tokio::sync::watch::Receiver<bool>`).
  - Separate `${env:NAME}` (interpolated into `config` and hashed into `config_hash`) from `${secret:NAME}` (resolved on demand via `ctx.secret("NAME") -> Result<Secret>`, backed in v0.1 by a `SecretProvider` reading `MALLOPS_SECRET_<NAME>`). `Secret` is an opaque newtype with redacted `Debug`/`Display` (`"***"`), does not implement `Serialize`, and is excluded from `config.json`, cache keys, `config_hash`, logs, and lineage.
  - Provide `ctx.config_as::<T>()` for typed config deserialization.
- **`mallops_core::Error`**: define a structured `thiserror` error enum (`Config`, `Validation`, `Store`, `Runtime`, `Cancelled`, `Secret`, `InvalidArtifactId`, `InvalidPath`, `Block { id, source }`).
- **`#[non_exhaustive]`**: mark extensible public enums and structs (`BlockSpec`, `BlockOutcome`, `Context`, `Artifact`, `Manifest`, `ArtifactKind`, `Stage`, `LineageEvent`, `Status`, `Error`, `CachePolicy`, `PayloadSource`) as `#[non_exhaustive]` with dedicated constructors/builders.
- **`ArtifactKind::Transform`**: add `transform` to `ArtifactKind` for fitted preprocessors consumed by `package` and `serve`.
- **`Manifest.io_schema`**: rename `Manifest.signature` to `Manifest.io_schema`, reserving `signature` for cryptographic artifact signatures (`FR-PKG-3`, v0.3).
- **`LineageEvent`**: add `timestamp`, `block_version`, `seed: Option<u64>`, `Status::Skipped`, and use `Digest` consistently instead of an undefined `Hash` type.

**Rationale:**
These additions eliminate every known breaking gap between core issues (`#1`–`#9`) and downstream std/serving issues (`#11`–`#17`), prevent secret leakage and async tracing corruption, guarantee consistent `parents` and lineage events, and make the core API evolvable toward 1.0 under `cargo-semver-checks`.

### D5 — `serve` outside the batch DAG

**Decision:**
- Remove `serve` from the batch DAG (`blocks:` in `mallops.yaml`). `mallops run` executes the finite batch DAG and produces a `package` artifact (gated by `eval`).
- `mallops serve [package-id]` is a separate long-running command that consumes a `package` artifact. An optional top-level `serve:` section in `mallops.yaml` configures defaults (target package output, bind address, port) for `mallops serve` without participating in `mallops run`.

**Rationale:**
A long-running HTTP server cannot be a normal batch DAG node: either it blocks `mallops run` indefinitely (preventing `RunFinished` and `mallops run && mallops serve`) or requires ad-hoc skipping rules. Separating batch production (`run` → `package`) from online inference (`serve`) gives both commands clear, composable semantics.

### D6 — Minimal `Predictor` contract in `mallops-std` for v0.1

**Decision:**
- Partially supersede [ADR-0004](0004-agnostic-artifacts-and-flavors.md) for v0.1: keep `Predictor` (and `LoadedPredictor`) out of `mallops-core` and define a minimal contract in `mallops-std` shared by both `evaluate` and `serve` (implemented prior to `evaluate`).
- Include `flavors(&self) -> &[&str]`, `executes_code(&self) -> bool`, structured payloads (`Json` and `Bytes`), and batch inference (`predict_batch`).
- Promote `Predictor` into `mallops-core` only before `v1.0`, once validated across multiple backends (`linfa`, external Python worker, `ort`/gRPC).

**Rationale:**
Freezing `Predictor` in `mallops-core` before implementing a single backend or gRPC/binary inference risks an early breaking change (for example, locking into JSON-only payloads). Defining it first in `mallops-std` before `evaluate` fixes the dependency inversion between `evaluate` and `serve` while keeping `mallops-core` minimal.

### D7 — Explicit Trust model and corrected `NFR-4`

**Decision:**
- Add a **Trust model** section to [SPEC.md](../SPEC.md) defining explicit trust boundaries:
  1. `mallops.yaml` is trusted executable configuration: `mallops run` executes native blocks and external commands with host privileges, whereas `mallops validate` and `mallops graph` never execute user code or external processes.
  2. Only WASM component plugins (post-v0.1) provide sandboxing. Native Rust blocks and external CLI/gRPC processes are trusted code (external CLI blocks run via `argv` without a shell, with an allowlisted environment and isolated workdir, but are not a security sandbox).
  3. Model flavors that execute code on deserialization (`pickle`, `joblib`, `sklearn-pickle`, `torchscript`) are flagged (`executes_code = true`). `mallops serve` refuses to load them unless explicitly opted in via `--trust-unsafe-flavors`. Non-executing formats (`onnx`, `safetensors`, `skops`) and signed packages (v0.3) are recommended.
  4. Secrets never enter `Context.config`, `config.json`, cache keys, logs, or lineage; `ArtifactStore` enforces validated paths, symlink rejection, and read-only storage.
- Update `NFR-4` in [SPEC.md](../SPEC.md) to reflect these exact guarantees instead of claiming sandboxing for all plugins.

**Rationale:**
Documenting trust boundaries up front prevents false security assumptions (especially around `pickle` serving and external commands) and ensures v0.1 CLI and store implementations enforce safe defaults from day one.

### D8 — Explicit `BlockRegistry` in v0.1 (no `inventory`)

**Decision:**
- Partially supersede [ADR-0003](0003-plugins-wasm-and-grpc.md) for v0.1: `mallops-core` provides an explicit, instance-local `BlockRegistry` populated via `registry.register(...)`.
- `mallops-std` exposes `pub fn register_std(registry: &mut BlockRegistry)` registering all feature-enabled standard blocks; `mallops-cli` and library callers invoke it explicitly.
- Compile-time auto-registration (such as `inventory` behind an optional feature) is deferred until post-v0.1 if multi-crate ergonomics justify its trade-offs.

**Rationale:**
`inventory` relies on linker `ctor` sections that can be silently stripped under LTO or unused-symbol elimination, behave inconsistently across static/cross-compilation/WASM targets, introduce global state that hurts test isolation, and add an `unsafe`-containing dependency to `mallops-core`. Explicit registration is deterministic, zero-magic, and trivial for v0.1's standard library.

## Consequences

- `mallops-core` contracts are internally consistent across artifacts, store, blocks, config, caching, lineage, and serving before issue `#1` begins.
- `mallops-core` remains free of ML, cloud, database, and HTTP/gRPC dependencies while gaining a realistic, auditable infrastructure allowlist.
- `ArtifactId` digests are deterministic, cross-platform, and deduplicated across runs.
- `LocalStore` and external blocks have enforceable immutability and path-traversal protections.
- `BlockRegistry` is isolated and deterministic in unit and integration tests.
- Accepted ADRs 0001–0006 remain untouched per MADR policy; this ADR records all supersessions and refinements.
