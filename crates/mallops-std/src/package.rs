//! `std/package`: bundle a `model`, an optional fitted `transform` and optional `metrics` into one
//! self-contained `package` artifact (`format: dir`), and [`Package::load`] to read it back.
//!
//! Layout of the payload: `package.json` plus `model/`, `transform/` and `metrics/`, each a
//! verbatim `store.fetch` of the source artifact. `package.json` embeds the source ids and
//! manifests so a consumer (`mallops serve`) can rebuild the model [`Artifact`] from the package
//! alone. No timestamps or run ids: the same inputs always give the same [`ArtifactId`].
//!
//! The package manifest carries the model's `flavor` and an `io_schema` (`input` = transform
//! input minus the target, `output` = model output; the model's own `io_schema` without a
//! transform). `io_schema` is absent when the model or the transform declares none (opaque
//! models such as `std/external`).
//!
//! Limits and the checks done by [`Package::load`]: see `docs/reference/package.md`.

use std::path::{Path, PathBuf};

use mallops_core::{
    Artifact, ArtifactId, ArtifactKind, ArtifactStore, Block, BlockOutcome, BlockSpec, BoxFuture,
    Context, DraftPayload, Error, Inputs, Manifest, OutputDraft, OutputDrafts, Port, Stage,
    StoreError,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const MODEL: &str = "model";
const PREPROCESS: &str = "preprocess";
const METRICS: &str = "metrics";
const PACKAGE: &str = "package";
/// Subdirectory names inside the package (`path` in `package.json`).
const TRANSFORM_DIR: &str = "transform";
/// Descriptor file at the root of the package.
const DESCRIPTOR: &str = "package.json";
const SCHEMA_VERSION: u32 = 1;

/// One bundled source artifact. Fields in alphabetical order: that is the serialized order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    id: ArtifactId,
    manifest: Manifest,
    path: String,
}

impl Entry {
    fn new(artifact: &Artifact, path: &str) -> Self {
        Self {
            id: artifact.id(),
            manifest: artifact.manifest().clone(),
            path: path.to_owned(),
        }
    }

    /// Checks `path`, `kind` and manifest `schema_version`, then returns the artifact.
    fn check(self, path: &str, kind: ArtifactKind) -> Result<Artifact, Error> {
        if self.path != path {
            return Err(bad(format!("`{path}` entry has path `{}`", self.path)));
        }
        if self.manifest.kind != kind {
            return Err(bad(format!("`{path}` entry is not a {kind} artifact")));
        }
        if self.manifest.schema_version != Manifest::SCHEMA_VERSION {
            return Err(bad(format!(
                "`{path}` entry has unsupported manifest schema_version {}",
                self.manifest.schema_version
            )));
        }
        Ok(Artifact::new(self.id, self.manifest))
    }
}

/// `package.json`. Fields in alphabetical order: that is the serialized order.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Descriptor {
    flavor: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    io_schema: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    metrics: Option<Entry>,
    model: Entry,
    schema_version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    transform: Option<Entry>,
}

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::Runtime(format!("invalid package: {msg}"))
}

/// A package materialized on disk, ready to be served.
///
/// [`Package::load`] applies no trust policy: the consumer must pass [`Package::flavor`] to
/// `PredictorRegistry::get` with its own `trust` (code-executing flavors need an explicit opt-in).
#[derive(Debug)]
pub struct Package {
    dir: PathBuf,
    model: Artifact,
    transform: Option<Artifact>,
    io_schema: Option<Value>,
}

impl Package {
    /// Loads the package `id` from `store` into the (new or empty) directory `dest`.
    ///
    /// The stored package is verified first (`store.verify`), then its manifest is checked, it is
    /// materialized, and the descriptor is checked against the manifest and the layout on disk.
    ///
    /// # Errors
    ///
    /// [`Error::Config`] if `id` is absent or is not a `package`/`dir` artifact; [`Error::Store`]
    /// if verification or the fetch fails; [`Error::Runtime`] if the descriptor is malformed,
    /// has an unsupported version or contradicts the manifest.
    pub async fn load(
        store: &dyn ArtifactStore,
        id: &ArtifactId,
        dest: &Path,
    ) -> Result<Self, Error> {
        match store.verify(id).await {
            Err(Error::Store(StoreError::NotFound(_))) => {
                return Err(Error::Config(format!("artifact {id} not found")));
            }
            r => r?,
        }
        let artifact = store
            .get(id)
            .await?
            .ok_or_else(|| Error::Config(format!("artifact {id} not found")))?;
        let manifest = artifact.manifest();
        if manifest.kind != ArtifactKind::Package || manifest.format != "dir" {
            return Err(Error::Config(format!(
                "artifact {id} is not a package (kind {}, format {})",
                manifest.kind, manifest.format
            )));
        }
        let dir = store.fetch(id, dest).await?;

        let bytes = std::fs::read(dir.join(DESCRIPTOR))
            .map_err(|e| bad(format!("cannot read {DESCRIPTOR}: {e}")))?;
        let d: Descriptor =
            serde_json::from_slice(&bytes).map_err(|e| bad(format!("{DESCRIPTOR}: {e}")))?;
        if d.schema_version != SCHEMA_VERSION {
            return Err(bad(format!(
                "unsupported schema_version {}",
                d.schema_version
            )));
        }
        if manifest.flavor.as_deref() != Some(d.flavor.as_str())
            || schema(manifest) != d.io_schema.as_ref()
        {
            return Err(bad("descriptor flavor/io_schema differ from the manifest"));
        }
        let model = d.model.check(MODEL, ArtifactKind::Model)?;
        if model.manifest().flavor.as_deref() != Some(d.flavor.as_str()) {
            return Err(bad("model flavor differs from the package flavor"));
        }
        let transform = d
            .transform
            .map(|e| e.check(TRANSFORM_DIR, ArtifactKind::Transform))
            .transpose()?;
        if let Some(m) = d.metrics {
            m.check(METRICS, ArtifactKind::Metrics)?;
        }
        if !dir.join(MODEL).is_dir() || transform.is_some() && !dir.join(TRANSFORM_DIR).is_dir() {
            return Err(bad("missing `model` or `transform` directory"));
        }
        Ok(Self {
            dir,
            model,
            transform,
            io_schema: d.io_schema,
        })
    }

    /// The bundled model artifact (id and manifest, as at packaging time).
    #[must_use]
    pub fn model(&self) -> &Artifact {
        &self.model
    }

    /// Directory with the model payload.
    #[must_use]
    pub fn model_dir(&self) -> PathBuf {
        self.dir.join(MODEL)
    }

    /// The bundled transform artifact and its payload directory, if any.
    #[must_use]
    pub fn transform(&self) -> Option<(&Artifact, PathBuf)> {
        self.transform
            .as_ref()
            .map(|t| (t, self.dir.join(TRANSFORM_DIR)))
    }

    /// The model flavor, which selects the predictor.
    #[must_use]
    pub fn flavor(&self) -> &str {
        self.model.manifest().flavor.as_deref().unwrap_or_default()
    }

    /// The package `io_schema` (`input`/`output` column lists), if the model declares one.
    #[must_use]
    pub fn io_schema(&self) -> Option<&Value> {
        self.io_schema.as_ref()
    }
}

/// The manifest's `io_schema`; `null` counts as absent.
fn schema(m: &Manifest) -> Option<&Value> {
    m.io_schema.as_ref().filter(|v| !v.is_null())
}

/// `input`/`output` column lists of an `io_schema`.
fn columns<'a>(schema: &'a Value, key: &str) -> Option<&'a Vec<Value>> {
    schema.get(key)?.as_array()
}

/// The `(name, type)` identity of a column.
fn key(c: &Value) -> (Option<&Value>, Option<&Value>) {
    (c.get("name"), c.get("type"))
}

/// Package `io_schema` from the model's and the optional transform's (see the module docs).
fn derive_io_schema(
    model: &Manifest,
    transform: Option<&Manifest>,
) -> Result<Option<Value>, Error> {
    let Some(m) = schema(model) else {
        return Ok(None);
    };
    let Some(transform) = transform else {
        return Ok(Some(m.clone()));
    };
    let Some(t) = schema(transform) else {
        return Ok(None);
    };
    let (t_in, t_out) = (columns(t, "input"), columns(t, "output"));
    let (m_in, m_out) = (columns(m, "input"), columns(m, "output"));
    if let (Some(t_out), Some(m_in)) = (t_out, m_in)
        && let Some(col) = m_in
            .iter()
            .find(|c| !t_out.iter().any(|o| key(o) == key(c)))
    {
        return Err(Error::Runtime(format!(
            "model input column {col} is not produced by the transform"
        )));
    }
    let (Some(t_in), Some(m_out)) = (t_in, m_out) else {
        return Ok(None);
    };
    let target = |c: &Value| {
        c.get("name")
            .is_some_and(|n| m_out.iter().any(|o| o.get("name") == Some(n)))
    };
    let input: Vec<&Value> = t_in.iter().filter(|c| !target(c)).collect();
    Ok(Some(serde_json::json!({ "input": input, "output": m_out })))
}

/// The `std/package` block.
pub(crate) struct PackageBlock {
    spec: BlockSpec,
}

impl PackageBlock {
    /// Creates `std/package`.
    pub(crate) fn new() -> Self {
        let spec = BlockSpec::new("std/package", "0.1.0", Stage::Package)
            .with_input(Port::new(MODEL, ArtifactKind::Model))
            .with_input(Port::new(PREPROCESS, ArtifactKind::Transform).optional())
            .with_input(Port::new(METRICS, ArtifactKind::Metrics).optional())
            .with_output(Port::new(PACKAGE, ArtifactKind::Package).with_format("dir"));
        Self { spec }
    }
}

/// `with:` must be absent, `null` or `{}`.
fn check_config(config: &Value) -> Result<(), Error> {
    if config.is_null() || config.as_object().is_some_and(|o| o.is_empty()) {
        Ok(())
    } else {
        Err(Error::Config("`std/package` takes no configuration".into()))
    }
}

impl Block for PackageBlock {
    fn spec(&self) -> &BlockSpec {
        &self.spec
    }

    fn validate_config(&self, config: &Value) -> Result<(), Error> {
        check_config(config)
    }

    fn run<'a>(
        &'a self,
        ctx: &'a Context,
        inputs: Inputs,
    ) -> BoxFuture<'a, Result<BlockOutcome, Error>> {
        Box::pin(async move {
            if ctx.cancel().is_cancelled() {
                return Err(Error::Cancelled);
            }
            check_config(ctx.config())?;
            let model = inputs.get(MODEL)?;
            let transform = inputs.try_get(PREPROCESS);
            let metrics = inputs.try_get(METRICS);

            // An optional `metrics` from a skipped evaluate still arrives: honour its gate here.
            let failed = metrics.is_some_and(|m| {
                m.manifest()
                    .metadata
                    .get("gate")
                    .and_then(|g| g.get("passed"))
                    == Some(&Value::Bool(false))
            });
            if failed {
                return Ok(BlockOutcome::Skipped {
                    reason: "metrics gate failed".into(),
                    outputs: OutputDrafts::new(),
                });
            }

            let flavor = model.manifest().flavor.clone().ok_or_else(|| {
                Error::Runtime("model artifact has no `flavor`: it cannot be served".into())
            })?;
            let io_schema = derive_io_schema(model.manifest(), transform.map(Artifact::manifest))?;

            let root = ctx.workdir().join(PACKAGE);
            std::fs::create_dir_all(&root)
                .map_err(|e| Error::Runtime(format!("{}: {e}", root.display())))?;
            let store = ctx.store();
            let mut parts = vec![(MODEL, model)];
            parts.extend(transform.map(|t| (TRANSFORM_DIR, t)));
            parts.extend(metrics.map(|m| (METRICS, m)));
            for (name, artifact) in parts {
                store.fetch(&artifact.id(), &root.join(name)).await?;
                if ctx.cancel().is_cancelled() {
                    return Err(Error::Cancelled);
                }
            }

            let descriptor = Descriptor {
                flavor: flavor.clone(),
                io_schema: io_schema.clone(),
                metrics: metrics.map(|m| Entry::new(m, METRICS)),
                model: Entry::new(model, MODEL),
                schema_version: SCHEMA_VERSION,
                transform: transform.map(|t| Entry::new(t, TRANSFORM_DIR)),
            };
            let bytes = serde_json::to_vec(&descriptor)
                .map_err(|e| Error::Runtime(format!("failed to serialize {DESCRIPTOR}: {e}")))?;
            let file = root.join(DESCRIPTOR);
            std::fs::write(&file, bytes)
                .map_err(|e| Error::Runtime(format!("{}: {e}", file.display())))?;

            let mut manifest = Manifest::new(ArtifactKind::Package, "dir").with_flavor(flavor);
            if let Some(s) = io_schema {
                manifest = manifest.with_io_schema(s);
            }
            Ok(BlockOutcome::Produced(OutputDrafts::new().with(
                PACKAGE,
                OutputDraft::new(manifest, DraftPayload::Dir(root)),
            )))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use mallops_core::{
        CancelSource, LocalStore, MemorySecretProvider, MemoryStore, PayloadSource, RunId,
    };
    use serde_json::json;

    use super::*;

    fn model_manifest() -> Manifest {
        Manifest::new(ArtifactKind::Model, "json")
            .with_flavor("mock")
            .with_io_schema(json!({
                "input": [{"name": "a", "type": "float64"}, {"name": "b", "type": "float64"}],
                "output": [{"name": "y", "type": "int64"}],
            }))
    }

    fn transform_manifest() -> Manifest {
        Manifest::new(ArtifactKind::Transform, "json").with_io_schema(json!({
            "input": [
                {"name": "a", "type": "float64"},
                {"name": "b", "type": "float64"},
                {"name": "y", "type": "int64"},
            ],
            "output": [{"name": "a", "type": "float64"}, {"name": "b", "type": "float64"}],
        }))
    }

    fn metrics_manifest(passed: Option<bool>) -> Manifest {
        let m = Manifest::new(ArtifactKind::Metrics, "json");
        match passed {
            Some(p) => m.with_metadata_entry("gate", json!({"metric": "f1", "passed": p})),
            None => m,
        }
    }

    async fn stage(store: &Arc<dyn ArtifactStore>, m: Manifest, data: &[u8]) -> Artifact {
        let id = store
            .put(m.clone(), PayloadSource::Bytes(data))
            .await
            .unwrap();
        Artifact::new(id, m)
    }

    struct Fixture {
        store: Arc<dyn ArtifactStore>,
        model: Artifact,
        transform: Artifact,
        metrics: Artifact,
    }

    async fn fixture(store: Arc<dyn ArtifactStore>) -> Fixture {
        Fixture {
            model: stage(&store, model_manifest(), b"model-bytes").await,
            transform: stage(&store, transform_manifest(), b"{}").await,
            metrics: stage(&store, metrics_manifest(Some(true)), b"{\"f1\":1.0}").await,
            store,
        }
    }

    fn full(f: &Fixture) -> Inputs {
        Inputs::new()
            .with(MODEL, f.model.clone())
            .with(PREPROCESS, f.transform.clone())
            .with(METRICS, f.metrics.clone())
    }

    fn ctx(store: &Arc<dyn ArtifactStore>, workdir: &Path, config: Value) -> Context {
        Context::new(
            RunId::new(),
            workdir.to_path_buf(),
            Arc::clone(store),
            config,
            CancelSource::new().token(),
            Arc::new(MemorySecretProvider::default()),
        )
    }

    /// Runs the block and commits the output like the runtime does; returns the artifact.
    async fn package(store: &Arc<dyn ArtifactStore>, inputs: Inputs) -> Result<Artifact, Error> {
        let wd = tempfile::tempdir().unwrap();
        let parents: Vec<ArtifactId> = inputs.iter().map(|(_, a)| a.id()).collect();
        let c = ctx(store, wd.path(), Value::Null);
        let BlockOutcome::Produced(drafts) = PackageBlock::new().run(&c, inputs).await? else {
            panic!("skipped");
        };
        let draft = drafts.get(PACKAGE).unwrap();
        let manifest = draft.manifest.clone().with_parents(parents);
        let id = store
            .put(manifest.clone(), draft.payload.as_source())
            .await?;
        Ok(Artifact::new(id, manifest))
    }

    fn stores() -> (tempfile::TempDir, Vec<Arc<dyn ArtifactStore>>) {
        let dir = tempfile::tempdir().unwrap();
        let local = LocalStore::new(dir.path().join("store")).unwrap();
        (dir, vec![Arc::new(local), Arc::new(MemoryStore::new())])
    }

    #[tokio::test]
    async fn package_bundles_model_transform_metrics() {
        let (_keep, stores) = stores();
        let f = fixture(Arc::clone(&stores[1])).await;
        let wd = tempfile::tempdir().unwrap();
        let c = ctx(&f.store, wd.path(), json!({}));
        let BlockOutcome::Produced(d) = PackageBlock::new().run(&c, full(&f)).await.unwrap() else {
            panic!("skipped");
        };
        let draft = d.get(PACKAGE).unwrap();
        let m = &draft.manifest;
        assert_eq!((m.kind, m.format.as_str()), (ArtifactKind::Package, "dir"));
        assert_eq!(m.flavor.as_deref(), Some("mock"));
        // target `y` is not an input; output is the model's
        assert_eq!(
            m.io_schema,
            Some(json!({
                "input": [{"name": "a", "type": "float64"}, {"name": "b", "type": "float64"}],
                "output": [{"name": "y", "type": "int64"}],
            }))
        );
        let root = wd.path().join(PACKAGE);
        assert_eq!(draft.payload, DraftPayload::Dir(root.clone()));
        for p in [
            "model/data",
            "transform/data",
            "metrics/data",
            "package.json",
        ] {
            assert!(root.join(p).is_file(), "{p}");
        }
        let d: Value =
            serde_json::from_slice(&std::fs::read(root.join("package.json")).unwrap()).unwrap();
        assert_eq!(d["schema_version"], 1);
        assert_eq!(d["model"]["path"], "model");
        assert_eq!(d["model"]["id"], f.model.id().to_string());
        assert_eq!(d["transform"]["path"], "transform");
        assert_eq!(d["metrics"]["path"], "metrics");
    }

    #[tokio::test]
    async fn package_roundtrip_load() {
        let (_keep, stores) = stores();
        for store in stores {
            let f = fixture(store).await;
            let pkg = package(&f.store, full(&f)).await.unwrap();
            let dest = tempfile::tempdir().unwrap();
            let loaded = Package::load(f.store.as_ref(), &pkg.id(), &dest.path().join("p"))
                .await
                .unwrap();
            assert_eq!(loaded.model(), &f.model);
            let (t, t_dir) = loaded.transform().unwrap();
            assert_eq!(t, &f.transform);
            assert!(t_dir.join("data").is_file());
            assert!(loaded.model_dir().join("data").is_file());
            assert_eq!(loaded.flavor(), "mock");
            assert_eq!(loaded.io_schema(), pkg.manifest().io_schema.as_ref());
        }
    }

    #[tokio::test]
    async fn package_optional_inputs_and_determinism() {
        let (_keep, stores) = stores();
        let mut ids = Vec::new();
        for store in stores {
            let f = fixture(store).await;
            let only = Inputs::new().with(MODEL, f.model.clone());
            let a = package(&f.store, only.clone()).await.unwrap();
            let b = package(&f.store, only).await.unwrap(); // different workdir
            assert_eq!(a.id(), b.id());
            // no transform: the model's own io_schema
            assert_eq!(a.manifest().io_schema, f.model.manifest().io_schema);
            let dest = tempfile::tempdir().unwrap();
            let l = Package::load(f.store.as_ref(), &a.id(), &dest.path().join("p"))
                .await
                .unwrap();
            assert!(l.transform().is_none());
            ids.push(a.id());
        }
        assert_eq!(ids[0], ids[1], "LocalStore and MemoryStore agree");
    }

    #[tokio::test]
    async fn package_skips_on_failed_gate() {
        let store: Arc<dyn ArtifactStore> = Arc::new(MemoryStore::new());
        let f = fixture(Arc::clone(&store)).await;
        let failed = stage(&store, metrics_manifest(Some(false)), b"{}").await;
        let inputs = Inputs::new().with(MODEL, f.model).with(METRICS, failed);
        let wd = tempfile::tempdir().unwrap();
        let out = PackageBlock::new()
            .run(&ctx(&store, wd.path(), Value::Null), inputs)
            .await
            .unwrap();
        assert!(
            matches!(out, BlockOutcome::Skipped { ref reason, .. } if reason == "metrics gate failed")
        );
    }

    #[tokio::test]
    async fn package_load_rejects_tampered() {
        let dir = tempfile::tempdir().unwrap();
        let store: Arc<dyn ArtifactStore> =
            Arc::new(LocalStore::new(dir.path().join("store")).unwrap());
        let f = fixture(Arc::clone(&store)).await;
        let pkg = package(&store, full(&f)).await.unwrap();
        let file = dir
            .path()
            .join("store/artifacts")
            .join(pkg.id().to_hex())
            .join("payload/package.json");
        let mut perms = std::fs::metadata(&file).unwrap().permissions();
        #[allow(clippy::permissions_set_readonly_false)] // portable; the store file is ours
        perms.set_readonly(false);
        std::fs::set_permissions(&file, perms).unwrap();
        let text = std::fs::read_to_string(&file)
            .unwrap()
            .replace("mock", "evil");
        std::fs::write(&file, text).unwrap();
        let err = Package::load(store.as_ref(), &pkg.id(), &dir.path().join("p"))
            .await
            .unwrap_err();
        assert!(
            matches!(err, Error::Store(StoreError::Corrupted { .. })),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn package_errors() {
        let store: Arc<dyn ArtifactStore> = Arc::new(MemoryStore::new());
        let f = fixture(Arc::clone(&store)).await;
        let wd = tempfile::tempdir().unwrap();
        let block = PackageBlock::new();

        // no flavor: not servable
        let opaque = stage(&store, Manifest::new(ArtifactKind::Model, "json"), b"x").await;
        let c = ctx(&store, wd.path(), Value::Null);
        let err = block
            .run(&c, Inputs::new().with(MODEL, opaque))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Runtime(_)), "{err:?}");

        // incoherent schemas
        let other = Manifest::new(ArtifactKind::Transform, "json").with_io_schema(json!({
            "input": [{"name": "z", "type": "float64"}],
            "output": [{"name": "z", "type": "float64"}],
        }));
        let other = stage(&store, other, b"{}").await;
        let inputs = Inputs::new()
            .with(MODEL, f.model.clone())
            .with(PREPROCESS, other);
        let wd2 = tempfile::tempdir().unwrap();
        let err = block
            .run(&ctx(&store, wd2.path(), Value::Null), inputs)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Runtime(_)), "{err:?}");

        // non-empty `with`, secrets included
        for cfg in [json!({"x": 1}), json!({"k": "${secret:K}"})] {
            assert!(matches!(block.validate_config(&cfg), Err(Error::Config(_))));
        }
        assert!(block.validate_config(&json!({})).is_ok());

        // load: absent id and wrong kind are Config errors
        let dest = tempfile::tempdir().unwrap();
        let err = Package::load(store.as_ref(), &f.model.id(), &dest.path().join("a"))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Config(_)), "{err:?}");
        let missing = ArtifactId::from_bytes([7; 32]);
        let err = Package::load(store.as_ref(), &missing, &dest.path().join("b"))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Config(_)), "{err:?}");
    }

    /// Stores a package whose sane `package.json` is mutated by `f`, then loads it.
    async fn tampered(f: fn(&mut Value), model_dir: bool) -> Result<Package, Error> {
        use std::collections::BTreeMap;

        use mallops_core::RelPath;

        let store: Arc<dyn ArtifactStore> = Arc::new(MemoryStore::new());
        let f_ = fixture(Arc::clone(&store)).await;
        let io = model_manifest().io_schema.unwrap();
        let mut d = json!({
            "flavor": "mock",
            "io_schema": io,
            "model": {"id": f_.model.id(), "manifest": f_.model.manifest(), "path": "model"},
            "schema_version": 1,
        });
        f(&mut d);
        let bytes = serde_json::to_vec(&d).unwrap();
        let mut files =
            BTreeMap::from([(RelPath::parse("package.json").unwrap(), bytes.as_slice())]);
        if model_dir {
            files.insert(RelPath::parse("model/data").unwrap(), b"x".as_slice());
        }
        let manifest = Manifest::new(ArtifactKind::Package, "dir")
            .with_flavor("mock")
            .with_io_schema(io);
        let id = store
            .put(manifest, PayloadSource::Files(files))
            .await
            .unwrap();
        let dest = tempfile::tempdir().unwrap();
        Package::load(store.as_ref(), &id, &dest.path().join("p")).await
    }

    #[tokio::test]
    async fn package_load_rejects_bad_descriptor() {
        type Case = (&'static str, fn(&mut Value), bool);
        let cases: [Case; 8] = [
            ("sane", |_| {}, true),
            ("version", |d| d["schema_version"] = json!(2), true),
            (
                "io_schema",
                |d| _ = d.as_object_mut().unwrap().remove("io_schema"),
                true,
            ),
            ("path", |d| d["model"]["path"] = json!("other"), true),
            (
                "kind",
                |d| d["model"]["manifest"]["kind"] = json!("transform"),
                true,
            ),
            ("unknown field", |d| d["extra"] = json!(1), true),
            (
                "manifest version",
                |d| d["model"]["manifest"]["schema_version"] = json!(2),
                true,
            ),
            ("missing model dir", |_| {}, false),
        ];
        for (name, f, model_dir) in cases {
            let r = tampered(f, model_dir).await;
            match (name, r) {
                ("sane", Ok(_)) | (_, Err(Error::Runtime(_))) => {}
                (_, r) => panic!("{name}: {r:?}"),
            }
        }
    }

    #[test]
    fn io_schema_coherence() {
        let model = |io: Value| Manifest::new(ArtifactKind::Model, "json").with_io_schema(io);
        let col = |n: &str| json!({"name": n, "type": "float64"});
        // extra fields on one side only are not an incoherence
        let t = Manifest::new(ArtifactKind::Transform, "json").with_io_schema(json!({
            "input": [col("a")],
            "output": [{"name": "a", "type": "float64", "nullable": true}],
        }));
        let m = model(json!({"input": [col("a")], "output": [col("y")]}));
        assert!(derive_io_schema(&m, Some(&t)).unwrap().is_some());
        // checked even when the model declares no `output`
        let m = model(json!({"input": [col("z")]}));
        let err = derive_io_schema(&m, Some(&t)).unwrap_err();
        assert!(matches!(err, Error::Runtime(_)), "{err:?}");
    }
}
