# `std/package`

Model packaging block that bundles a trained model, an optional fitted preprocessing transform, and
optional evaluation metrics into a self-contained `package` artifact (`format: dir`). Behind the
optional `package` Cargo feature of `mallops-std` (`package = []`, not in `default`, no new crate
dependency; enabled in `mallops-cli`).
See [SPEC FR-PKG-1](../SPEC.md#37-model-packaging).

```yaml
blocks:
  package:
    use: std/package
    in:
      model: model.model
      preprocess: features.transform
      metrics: eval.metrics
```

`preprocess` and `metrics` are optional; a minimal model-only package wires only `model`:

```yaml
blocks:
  package:
    use: std/package
    in: { model: model.model }
```

## Configuration (`with:`)

`std/package` takes no configuration. `with:` must be omitted or an empty mapping (`{}`). Any
non-empty value — including unknown keys, non-object values, or `${secret:` references — is rejected
with `Error::Config` by both `validate_config` (caught by `mallops validate` under
`blocks.<id>.with`) and `run`.

## Ports

`version: "0.1.0"`, `Stage::Package`:

- **Block ID:** `std/package`
- **Inputs:**
  - `model` (required): `kind: model`, any `format`, any `flavor` (the model manifest must carry a
    `flavor` at runtime).
  - `preprocess` (optional): `kind: transform`, any `format`.
  - `metrics` (optional): `kind: metrics`, any `format`.
- **Output:**
  - `package`: `kind: package`, `format: dir`.

## Execution and directory layout

- **Quality gate check:** if `metrics` is wired and
  `metrics.manifest().metadata["gate"]["passed"] == false` (for example, from
  [`std/evaluate`](evaluate.md) with `on_fail: skip`), `std/package` does not bundle the model and
  returns `BlockOutcome::Skipped { reason: "metrics gate failed", outputs: OutputDrafts::new() }`.
  Because the node is `Skipped`, it is never cached and `mallops run` leaves `runs/latest.json`
  unchanged so `mallops serve` keeps serving the previous accepted package. When `metrics` is
  omitted, no gate check is performed.
- **Materialization:** each present input is materialized into `<workdir>/package/` via
  `store.fetch` (`package/model/`, optional `package/transform/`, optional `package/metrics/`).
  `store.fetch` rejects symlinks and validates relative paths via `RelPath`; the runtime re-validates
  the resulting `DraftPayload::Dir` on commit.
- **Directory layout (`payload/`):**

```text
payload/
├── package.json
├── model/…          # materialized model artifact payload
├── transform/…      # optional: materialized preprocess artifact payload
└── metrics/…        # optional: materialized metrics artifact payload
```

- **Cancellation:** `ctx.cancel()` is checked at run start and after each `store.fetch`.

## `package.json` format

`package.json` is compact deterministic JSON serialized from a struct with fixed field order, with
no timestamps or run IDs:

```json
{
  "schema_version": 1,
  "flavor": "smartcore-logistic",
  "io_schema": {
    "input": [{ "name": "age", "type": "f64" }],
    "output": [{ "name": "churned", "type": "bool" }]
  },
  "model": {
    "path": "model",
    "id": "<64-hex-artifact-id>",
    "manifest": { "...": "embedded model Manifest" }
  },
  "transform": {
    "path": "transform",
    "id": "<64-hex-artifact-id>",
    "manifest": { "...": "embedded transform Manifest" }
  },
  "metrics": {
    "path": "metrics",
    "id": "<64-hex-artifact-id>",
    "manifest": { "...": "embedded metrics Manifest" }
  }
}
```

- `schema_version`: always `1`.
- `flavor`: model flavor string, matching the package manifest's `flavor`.
- `io_schema`: omitted when absent; when present, matches the package manifest's `io_schema`.
- `model` (required), `transform` (optional), `metrics` (optional): each entry carries its fixed
  relative `path` (`"model"`, `"transform"`, `"metrics"`), its source `ArtifactId` (`id`), and its
  embedded source `Manifest` (`manifest`).
- Embedding the source manifests makes the package **self-contained**: downstream consumers
  (`mallops serve`, `std/predict.batch`) reconstruct the model's `Artifact` for
  [`Predictor::load`](predictor.md#predictor) from the package `ArtifactId` alone, without requiring
  the original source artifacts to remain in the store.

## Output manifest (`flavor` and `io_schema`)

Output manifest fields:

- `kind`: `package`
- `format`: `dir`
- `flavor`: propagated from `model.manifest().flavor`. If `model` has no `flavor`, the block fails
  with `Error::Runtime` (an unflavored model cannot be dispatched to a `Predictor` at serve time).
- `io_schema`:
  - **Without `preprocess`:** copied from `model.manifest().io_schema` (omitted if `model` has no
    `io_schema`, e.g. opaque models from [`std/external`](external.md)).
  - **With `preprocess`:** composed as `{ "input": <raw_input>, "output": model.io_schema.output }`,
    where `<raw_input>` is `transform.io_schema.input` with any column whose `name` appears in
    `model.io_schema.output` removed (so the target column is excluded from the serving input
    schema). If either `preprocess` or `model` lacks `io_schema`, the package's `io_schema` is
    omitted.
  - **Coherence check:** when both `transform.io_schema.output` and `model.io_schema.input` are
    present, every `{ "name", "type" }` column in `model.io_schema.input` must appear in
    `transform.io_schema.output` with matching `name` and `type`; otherwise the block fails with
    `Error::Runtime` at package time instead of failing later at serve time.
- `metadata`: empty (`{}`). No timestamps, run IDs, or block-written `parents` (the runtime injects
  `parents` on commit).

## Cache

`CachePolicy::Deterministic { extra_hash: None }`.

- Identical inputs, config, node `seed:`, and block version hit the cache and produce the same
  canonical `ArtifactId`, independent of the node workdir path.
- `Skipped` runs (such as a failed metrics gate) are never cached.

## Loading packages (`mallops_std::Package`)

When the `package` feature is enabled, `mallops-std` re-exports `mallops_std::Package` for
consumers such as `mallops serve` and `std/predict.batch`:

```rust
impl Package {
    pub async fn load(
        store: &dyn ArtifactStore,
        id: &ArtifactId,
        dest: &Path,
    ) -> Result<Package, Error>;

    pub fn model(&self) -> &Artifact;
    pub fn model_dir(&self) -> &Path;
    pub fn transform(&self) -> Option<(&Artifact, &Path)>;
    pub fn flavor(&self) -> &str;
    pub fn io_schema(&self) -> Option<&serde_json::Value>;
}
```

The future returned by `Package::load` is `Send`. Callers inside a `Block` can pass
`ctx.store().as_ref()`.

### What `Package::load` verifies

1. **Lookup and manifest check:** calls `store.get(id)`. A missing artifact or a manifest whose
   `kind` is not `package` returns `Error::Config`; a `format` other than `"dir"` or a missing
   `flavor` returns `Error::Runtime`.
2. **Integrity verification:** calls `store.verify(id)` **before** `store.fetch(id, dest)`. Because
   the canonical `ArtifactId` hashes `manifest.json`, `package.json`, and every bundled payload
   file, any tampering or corruption in the store fails with `Error::Store` before files are
   materialized or parsed.
3. **Materialization:** fetches the package directory into `dest` via `store.fetch(id, dest)`.
4. **Descriptor validation:** parses `dest/package.json` with `deny_unknown_fields` and checks:
   - `schema_version == 1`
   - `flavor` equals the package manifest's `flavor`
   - `io_schema` equals the package manifest's `io_schema`
   - `model.path == "model"`, `transform.path == "transform"` (if present), and
     `metrics.path == "metrics"` (if present)
   - embedded manifest kinds match (`Model`, `Transform`, `Metrics`)
   - `model/` (and `transform/` when present) exist inside `dest`.

### Trust policy

`Package::load` verifies store integrity and structural consistency, but **does not enforce model
code-execution trust policy**. That policy belongs to the consumer: callers must resolve the
predictor through [`PredictorRegistry::get(pkg.flavor(), trust_unsafe_flavors)`](predictor.md#resolution-get)
before calling `predictor.load(pkg.model(), pkg.model_dir())`.

## Errors

| Situation | Error |
|---|---|
| Non-empty `with:` (any key, non-object value, `${secret:` reference) | `Error::Config` (in `mallops validate`, aggregated into `Error::Validation` under `blocks.<id>.with`) |
| Missing `flavor` in `model` manifest | `Error::Runtime` |
| `model.io_schema.input` column missing from or incompatible with `transform.io_schema.output` | `Error::Runtime` |
| Cancellation via `ctx.cancel()` | `Error::Cancelled` |
| `Package::load`: artifact `id` not found in store, or `kind != package` | `Error::Config` |
| `Package::load`: store corruption or tampering detected by `store.verify(id)` | `Error::Store` |
| `Package::load`: `format != dir`, missing manifest `flavor`, invalid `package.json`, unsupported `schema_version`, mismatched `flavor`/`io_schema`/`path`/embedded `kind`, or missing bundled subdir | `Error::Runtime` |

A wired `metrics` input with `metadata["gate"]["passed"] == false` is not an error: `std/package`
returns `BlockOutcome::Skipped { reason: "metrics gate failed" }`.

## Known limitations

- **No environment requirements in v0.1:** recording runtime/Python dependency requirements in the
  package descriptor is deferred to a later release ([SPEC FR-PKG-1](../SPEC.md#37-model-packaging)).
- **No OCI packaging or signing in v0.1:** OCI image output (`FR-PKG-2`, v0.2) and cryptographic
  signing (`FR-PKG-3`, v0.3) are out of scope for v0.1.
