//! Block model types: stages, ports, specs, output drafts, outcomes, cache policy and inputs.
//!
//! These are the data types of the block contract (ADR-0007 D4). Blocks return
//! [`OutputDrafts`] and never call `ArtifactStore::put` for outputs: the runtime
//! commits them, injects `Manifest.parents` and validates them against the
//! resolved [`Port`]s. No ML type appears here (ADR-0004).

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::{
    Artifact, ArtifactKind, BoxFuture, Context, Digest, Error, Manifest, PayloadSource, RelPath,
};

/// Lifecycle stage a block belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum Stage {
    /// Data ingestion.
    Ingest,
    /// Writing user data to lakes/databases (unrelated to the artifact store).
    DataStorage,
    /// Preprocessing and feature engineering.
    Preprocess,
    /// Model training.
    Train,
    /// Model engineering (tuning, compression, conversion).
    ModelEngineering,
    /// Model evaluation.
    Evaluate,
    /// Packaging a deployable bundle.
    Package,
    /// Model/package versioning.
    Version,
    /// Deployment.
    Deploy,
    /// Online serving (not a DAG node, see ADR-0007 D5).
    Serve,
    /// Monitoring.
    Monitor,
    /// Retraining triggers.
    Retrain,
    /// Any user-defined stage.
    Custom,
}

/// A named block input or output with the artifact `kind`, and optionally accepted `formats` and `flavors`.
///
/// On an input port, empty `formats`/`flavors` mean "any" (SPEC §6) and several values
/// mean "any of these". On an output port they describe what the block produces.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Port {
    /// Port name, unique per direction within a block.
    pub name: Cow<'static, str>,
    /// Artifact kind carried by this port.
    pub kind: ArtifactKind,
    /// Accepted/produced payload formats; empty means unconstrained.
    pub formats: Vec<Cow<'static, str>>,
    /// Accepted/produced flavors; empty means unconstrained.
    pub flavors: Vec<Cow<'static, str>>,
    /// Whether the input must be provided (default `true`); a node whose required
    /// input comes from a skipped node is skipped too (ADR-0007 D4).
    pub required: bool,
}

/// `true` if every produced value is accepted; empty `accepted` accepts anything,
/// while an empty `produced` never satisfies a constraint.
fn covers(accepted: &[Cow<'static, str>], produced: &[&str]) -> bool {
    accepted.is_empty()
        || (!produced.is_empty() && produced.iter().all(|p| accepted.iter().any(|a| a == p)))
}

impl Port {
    /// Creates a required port accepting any format and flavor of `kind`.
    #[must_use]
    pub fn new(name: impl Into<Cow<'static, str>>, kind: ArtifactKind) -> Self {
        Self {
            name: name.into(),
            kind,
            formats: Vec::new(),
            flavors: Vec::new(),
            required: true,
        }
    }

    /// Adds an accepted/produced payload `format`.
    #[must_use]
    pub fn with_format(mut self, format: impl Into<Cow<'static, str>>) -> Self {
        self.formats.push(format.into());
        self
    }

    /// Adds an accepted/produced `flavor`.
    #[must_use]
    pub fn with_flavor(mut self, flavor: impl Into<Cow<'static, str>>) -> Self {
        self.flavors.push(flavor.into());
        self
    }

    /// Marks the port as optional (see [`Inputs::try_get`]).
    #[must_use]
    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }

    fn accepts_parts(&self, kind: ArtifactKind, formats: &[&str], flavors: &[&str]) -> bool {
        self.kind == kind && covers(&self.formats, formats) && covers(&self.flavors, flavors)
    }

    /// Returns `true` if an artifact with this `manifest` satisfies this port.
    #[must_use]
    pub fn accepts_manifest(&self, manifest: &Manifest) -> bool {
        self.accepts_parts(
            manifest.kind,
            &[&manifest.format],
            manifest.flavor.as_deref().as_slice(),
        )
    }

    /// Returns `true` if an output port `output` can be wired into this input port.
    ///
    /// Every format/flavor the output may produce must be accepted.
    #[must_use]
    pub fn accepts(&self, output: &Port) -> bool {
        let formats: Vec<&str> = output.formats.iter().map(|f| &**f).collect();
        let flavors: Vec<&str> = output.flavors.iter().map(|f| &**f).collect();
        self.accepts_parts(output.kind, &formats, &flavors)
    }
}

/// Static description of a block.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct BlockSpec {
    /// Block id, e.g. `std/ingest.csv`.
    pub id: Cow<'static, str>,
    /// Block version; part of the cache key.
    pub version: Cow<'static, str>,
    /// Lifecycle stage.
    pub stage: Stage,
    /// Declared input ports.
    pub inputs: Vec<Port>,
    /// Declared output ports.
    pub outputs: Vec<Port>,
    /// Optional JSON Schema of the `with:` configuration.
    pub config_schema: Option<serde_json::Value>,
}

impl BlockSpec {
    /// Creates a spec with no ports and no config schema.
    #[must_use]
    pub fn new(
        id: impl Into<Cow<'static, str>>,
        version: impl Into<Cow<'static, str>>,
        stage: Stage,
    ) -> Self {
        Self {
            id: id.into(),
            version: version.into(),
            stage,
            inputs: Vec::new(),
            outputs: Vec::new(),
            config_schema: None,
        }
    }

    /// Adds an input port.
    #[must_use]
    pub fn with_input(mut self, port: Port) -> Self {
        self.inputs.push(port);
        self
    }

    /// Adds an output port.
    #[must_use]
    pub fn with_output(mut self, port: Port) -> Self {
        self.outputs.push(port);
        self
    }

    /// Sets the JSON Schema of the block configuration.
    #[must_use]
    pub fn with_config_schema(mut self, schema: serde_json::Value) -> Self {
        self.config_schema = Some(schema);
        self
    }
}

/// Ports of a block after config-driven resolution (`Block::resolve_ports`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct ResolvedPorts {
    /// Resolved input ports.
    pub inputs: Vec<Port>,
    /// Resolved output ports.
    pub outputs: Vec<Port>,
}

impl ResolvedPorts {
    /// Creates resolved ports from explicit lists.
    #[must_use]
    pub fn new(inputs: Vec<Port>, outputs: Vec<Port>) -> Self {
        Self { inputs, outputs }
    }
}

impl From<&BlockSpec> for ResolvedPorts {
    fn from(spec: &BlockSpec) -> Self {
        Self {
            inputs: spec.inputs.clone(),
            outputs: spec.outputs.clone(),
        }
    }
}

/// Owned payload of an [`OutputDraft`]; mirrors [`PayloadSource`] without borrows.
///
/// The runtime (#7) must confine `File`/`Dir` paths inside `ctx.workdir()` before `put`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DraftPayload {
    /// A single regular file.
    File(PathBuf),
    /// A directory walked recursively.
    Dir(PathBuf),
    /// A single in-memory byte buffer.
    Bytes(Vec<u8>),
    /// Multiple in-memory files.
    Files(BTreeMap<RelPath, Vec<u8>>),
}

impl DraftPayload {
    /// Borrows this payload as the [`PayloadSource`] expected by `ArtifactStore::put`.
    #[must_use]
    pub fn as_source(&self) -> PayloadSource<'_> {
        match self {
            Self::File(p) => PayloadSource::File(p),
            Self::Dir(p) => PayloadSource::Dir(p),
            Self::Bytes(b) => PayloadSource::Bytes(b),
            Self::Files(f) => {
                PayloadSource::Files(f.iter().map(|(k, v)| (k.clone(), v.as_slice())).collect())
            }
        }
    }
}

/// An output a block wants stored: manifest draft plus payload.
///
/// The runtime commits it with `ArtifactStore::put`, filling `manifest.parents`
/// from the node's actual input ids.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct OutputDraft {
    /// Manifest draft; `parents` is overwritten by the runtime.
    pub manifest: Manifest,
    /// Payload to store.
    pub payload: DraftPayload,
}

impl OutputDraft {
    /// Creates a draft from a manifest and payload.
    #[must_use]
    pub fn new(manifest: Manifest, payload: DraftPayload) -> Self {
        Self { manifest, payload }
    }
}

/// Output drafts keyed by output port name.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OutputDrafts(BTreeMap<String, OutputDraft>);

impl OutputDrafts {
    /// Creates an empty set of drafts.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds (or replaces) the draft for output port `port`.
    #[must_use]
    pub fn with(mut self, port: impl Into<String>, draft: OutputDraft) -> Self {
        self.0.insert(port.into(), draft);
        self
    }

    /// Returns the draft for `port`, if any.
    #[must_use]
    pub fn get(&self, port: &str) -> Option<&OutputDraft> {
        self.0.get(port)
    }

    /// Iterates over `(port name, draft)` in port-name order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &OutputDraft)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v))
    }
}

impl IntoIterator for OutputDrafts {
    type Item = (String, OutputDraft);
    type IntoIter = std::collections::btree_map::IntoIter<String, OutputDraft>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// Result of running a block.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BlockOutcome {
    /// The block produced its outputs.
    Produced(OutputDrafts),
    /// The block chose to skip; downstream nodes needing its outputs are skipped too.
    Skipped {
        /// Why the block was skipped.
        reason: String,
        /// Outputs still provided despite skipping (may be empty).
        outputs: OutputDrafts,
    },
}

/// How the runtime may cache a block's result.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CachePolicy {
    /// Cacheable; `extra_hash` (e.g. of local files or scripts) is mixed into the cache key.
    Deterministic {
        /// Extra digest invalidating the cache when external state changes.
        extra_hash: Option<Digest>,
    },
    /// Never cached.
    Uncacheable,
}

impl Default for CachePolicy {
    fn default() -> Self {
        Self::Deterministic { extra_hash: None }
    }
}

/// Input artifacts of a block run, keyed by input port name.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Inputs(BTreeMap<String, Artifact>);

impl Inputs {
    /// Creates an empty input set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds (or replaces) the artifact bound to input port `port`.
    #[must_use]
    pub fn with(mut self, port: impl Into<String>, artifact: Artifact) -> Self {
        self.0.insert(port.into(), artifact);
        self
    }

    /// Returns the artifact bound to `port`, or [`Error::Validation`] if it is missing.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when no artifact is bound to `port`.
    pub fn get(&self, port: &str) -> Result<&Artifact, Error> {
        self.0
            .get(port)
            .ok_or_else(|| Error::validation("", format!("missing input port `{port}`")))
    }

    /// Returns the artifact bound to `port`, if any (for inputs declared [`Port::optional`]).
    #[must_use]
    pub fn try_get(&self, port: &str) -> Option<&Artifact> {
        self.0.get(port)
    }

    /// Iterates over `(port name, artifact)` in port-name order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Artifact)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v))
    }
}

/// A composable lifecycle step (ADR-0007 D2, D4).
///
/// Dyn-compatible: the runtime holds blocks as `Arc<dyn Block>`. A block returns
/// [`OutputDrafts`]; it never calls `ArtifactStore::put` for its outputs.
///
/// ```
/// use std::sync::Arc;
/// use mallops_core::*;
///
/// struct Hello(BlockSpec);
/// impl Block for Hello {
///     fn spec(&self) -> &BlockSpec { &self.0 }
///     fn run<'a>(&'a self, ctx: &'a Context, _inputs: Inputs)
///         -> BoxFuture<'a, Result<BlockOutcome, Error>> {
///         Box::pin(async move {
///             if ctx.cancel().is_cancelled() { return Err(Error::Cancelled) }
///             let m = Manifest::new(ArtifactKind::Blob, "text");
///             let d = OutputDraft::new(m, DraftPayload::Bytes(b"hi".to_vec()));
///             Ok(BlockOutcome::Produced(OutputDrafts::new().with("out", d)))
///         })
///     }
/// }
///
/// let spec = BlockSpec::new("ex/hello", "0.1.0", Stage::Custom)
///     .with_output(Port::new("out", ArtifactKind::Blob));
/// let ctx = Context::new(RunId::new(), std::env::temp_dir(), Arc::new(MemoryStore::new()),
///     serde_json::Value::Null, CancelSource::new().token(), Arc::new(MemorySecretProvider::default()));
/// let rt = tokio::runtime::Runtime::new().unwrap();
/// let out = rt.block_on(Hello(spec).run(&ctx, Inputs::new())).unwrap();
/// assert!(matches!(out, BlockOutcome::Produced(d) if d.get("out").is_some()));
/// ```
pub trait Block: Send + Sync {
    /// Static description of the block.
    fn spec(&self) -> &BlockSpec;

    /// Validates the `with:` configuration before the pipeline runs.
    ///
    /// # Errors
    ///
    /// An [`Error`] (typically [`Error::Validation`]) if `config` is invalid.
    fn validate_config(&self, _config: &serde_json::Value) -> Result<(), Error> {
        Ok(())
    }

    /// Resolves the ports for a given configuration (default: the spec's ports).
    ///
    /// # Errors
    ///
    /// An [`Error`] if the ports cannot be derived from `config`.
    fn resolve_ports(&self, _config: &serde_json::Value) -> Result<ResolvedPorts, Error> {
        Ok(ResolvedPorts::from(self.spec()))
    }

    /// Cache policy for this run (default: [`CachePolicy::default`]).
    ///
    /// `extra_hash` must not depend on secret values (NFR-4: secrets never enter cache keys).
    /// Not called when the node or the run bypasses the cache.
    ///
    /// # Errors
    ///
    /// An [`Error`] if the policy cannot be determined.
    fn cache_policy<'a>(&'a self, _ctx: &'a Context) -> BoxFuture<'a, Result<CachePolicy, Error>> {
        Box::pin(async { Ok(CachePolicy::default()) })
    }

    /// Runs the block and returns output drafts for the runtime to commit.
    ///
    /// Long-running blocks should check `ctx.cancel()` periodically and return
    /// [`Error::Cancelled`] once it is cancelled.
    ///
    /// # Errors
    ///
    /// [`Error::Cancelled`] on cancellation, otherwise any [`Error`] describing why the
    /// block failed (the runtime marks the node failed and cancels its descendants).
    fn run<'a>(
        &'a self,
        ctx: &'a Context,
        inputs: Inputs,
    ) -> BoxFuture<'a, Result<BlockOutcome, Error>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ArtifactId;

    fn manifest(flavor: Option<&str>) -> Manifest {
        let m = Manifest::new(ArtifactKind::Model, "onnx");
        match flavor {
            Some(f) => m.with_flavor(f),
            None => m,
        }
    }

    #[test]
    fn port_kind_must_match() {
        let p = Port::new("in", ArtifactKind::Model);
        assert!(p.accepts_manifest(&manifest(None)));
        assert!(!Port::new("in", ArtifactKind::Dataset).accepts_manifest(&manifest(None)));
        assert!(p.accepts(&Port::new("out", ArtifactKind::Model).with_format("x")));
        assert!(!p.accepts(&Port::new("out", ArtifactKind::Blob)));
    }

    #[test]
    fn port_format_constraint() {
        let p = Port::new("in", ArtifactKind::Model).with_format("onnx");
        assert!(p.accepts_manifest(&manifest(None)));
        assert!(
            !Port::new("in", ArtifactKind::Model)
                .with_format("pt")
                .accepts_manifest(&manifest(None))
        );
        // unconstrained output format cannot satisfy a constrained input
        assert!(!p.accepts(&Port::new("out", ArtifactKind::Model)));
        assert!(p.accepts(&Port::new("out", ArtifactKind::Model).with_format("onnx")));
    }

    #[test]
    fn port_flavor_constraint() {
        let p = Port::new("in", ArtifactKind::Model).with_flavor("tabular");
        assert!(p.accepts_manifest(&manifest(Some("tabular"))));
        assert!(!p.accepts_manifest(&manifest(Some("vision"))));
        assert!(!p.accepts_manifest(&manifest(None)));
        assert!(p.accepts(&Port::new("o", ArtifactKind::Model).with_flavor("tabular")));
        assert!(!p.accepts(&Port::new("o", ArtifactKind::Model)));
    }

    #[test]
    fn port_accepts_any_of_several_formats() {
        let p = Port::new("in", ArtifactKind::Model)
            .with_format("onnx")
            .with_format("safetensors");
        assert!(p.accepts_manifest(&manifest(None)));
        let st = Manifest::new(ArtifactKind::Model, "safetensors");
        assert!(p.accepts_manifest(&st));
        assert!(!p.accepts_manifest(&Manifest::new(ArtifactKind::Model, "pt")));
        assert!(p.accepts(&Port::new("o", ArtifactKind::Model).with_format("onnx")));
        let both = Port::new("o", ArtifactKind::Model)
            .with_format("onnx")
            .with_format("pt");
        assert!(!p.accepts(&both));
    }

    #[test]
    fn port_required_by_default_and_optional() {
        assert!(Port::new("in", ArtifactKind::Blob).required);
        assert!(!Port::new("in", ArtifactKind::Blob).optional().required);
    }

    #[test]
    fn skipped_outcome_carries_reason_and_outputs() {
        let o = BlockOutcome::Skipped {
            reason: "no new data".into(),
            outputs: OutputDrafts::new(),
        };
        let BlockOutcome::Skipped { reason, outputs } = o else {
            panic!("expected Skipped")
        };
        assert_eq!(reason, "no new data");
        assert_eq!(outputs, OutputDrafts::new());
    }

    #[test]
    fn produced_outcome_drafts_by_port() {
        let draft = OutputDraft::new(manifest(None), DraftPayload::Bytes(b"abc".to_vec()));
        let drafts = OutputDrafts::new().with("model", draft.clone());
        assert_eq!(drafts.get("model"), Some(&draft));
        assert_eq!(draft.payload.as_source(), PayloadSource::Bytes(b"abc"));
        assert!(matches!(
            BlockOutcome::Produced(drafts),
            BlockOutcome::Produced(_)
        ));
    }

    #[test]
    fn cache_policy_defaults_to_deterministic_without_extra_hash() {
        assert_eq!(
            CachePolicy::default(),
            CachePolicy::Deterministic { extra_hash: None }
        );
        assert_ne!(CachePolicy::default(), CachePolicy::Uncacheable);
    }

    #[test]
    fn inputs_missing_port_is_validation_error() {
        let art = Artifact::new(ArtifactId::from_bytes([1; 32]), manifest(None));
        let inputs = Inputs::new().with("model", art.clone());
        assert_eq!(inputs.get("model").unwrap(), &art);
        let err = inputs.get("data").unwrap_err();
        assert!(matches!(&err, Error::Validation(v) if v[0].message.contains("data")));
        assert!(inputs.try_get("data").is_none());
    }

    #[test]
    fn spec_builder_and_resolved_ports() {
        let spec = BlockSpec::new("std/train", "0.1.0", Stage::Train)
            .with_input(Port::new("features", ArtifactKind::Features))
            .with_output(Port::new("model", ArtifactKind::Model))
            .with_config_schema(serde_json::json!({"type": "object"}));
        let rp = ResolvedPorts::from(&spec);
        assert_eq!(rp.inputs, spec.inputs);
        assert_eq!(rp.outputs.len(), 1);
        assert!(spec.config_schema.is_some());
    }

    use crate::{CancelSource, MemorySecretProvider, MemoryStore, RunId};
    use std::sync::Arc;

    struct Toy {
        spec: BlockSpec,
        skip: bool,
    }

    impl Block for Toy {
        fn spec(&self) -> &BlockSpec {
            &self.spec
        }
        fn validate_config(&self, config: &serde_json::Value) -> Result<(), Error> {
            if config.get("bad").is_some() {
                return Err(Error::validation("", "bad config"));
            }
            Ok(())
        }
        fn resolve_ports(&self, config: &serde_json::Value) -> Result<ResolvedPorts, Error> {
            let name = config.get("out").and_then(|v| v.as_str()).unwrap_or("out");
            Ok(ResolvedPorts::new(
                vec![],
                vec![Port::new(name.to_string(), ArtifactKind::Blob)],
            ))
        }
        fn run<'a>(
            &'a self,
            _ctx: &'a Context,
            _inputs: Inputs,
        ) -> BoxFuture<'a, Result<BlockOutcome, Error>> {
            let skip = self.skip;
            Box::pin(async move {
                if skip {
                    return Ok(BlockOutcome::Skipped {
                        reason: "nothing to do".into(),
                        outputs: OutputDrafts::new(),
                    });
                }
                let d = OutputDraft::new(
                    Manifest::new(ArtifactKind::Blob, "text"),
                    DraftPayload::Bytes(b"x".to_vec()),
                );
                Ok(BlockOutcome::Produced(OutputDrafts::new().with("out", d)))
            })
        }
    }

    fn toy(skip: bool) -> Toy {
        Toy {
            spec: BlockSpec::new("ex/toy", "0.1.0", Stage::Custom)
                .with_output(Port::new("out", ArtifactKind::Blob)),
            skip,
        }
    }

    /// Block relying on every trait default.
    struct Plain(BlockSpec);
    impl Block for Plain {
        fn spec(&self) -> &BlockSpec {
            &self.0
        }
        fn run<'a>(
            &'a self,
            _: &'a Context,
            _: Inputs,
        ) -> BoxFuture<'a, Result<BlockOutcome, Error>> {
            Box::pin(async { Ok(BlockOutcome::Produced(OutputDrafts::new())) })
        }
    }

    fn ctx() -> Context {
        Context::new(
            RunId::new(),
            std::env::temp_dir(),
            Arc::new(MemoryStore::new()),
            serde_json::Value::Null,
            CancelSource::new().token(),
            Arc::new(MemorySecretProvider::default()),
        )
    }

    #[tokio::test]
    async fn block_is_dyn_compatible_and_runs() {
        let b: Arc<dyn Block> = Arc::new(toy(false));
        let boxed: Box<dyn Block> = Box::new(toy(true));
        assert_eq!(b.spec().id, "ex/toy");
        let BlockOutcome::Produced(d) = b.run(&ctx(), Inputs::new()).await.unwrap() else {
            panic!("expected Produced")
        };
        assert!(d.get("out").is_some());
        let out = boxed.run(&ctx(), Inputs::new()).await.unwrap();
        assert!(matches!(out, BlockOutcome::Skipped { .. }));
    }

    #[tokio::test]
    async fn trait_defaults() {
        let p = Plain(toy(false).spec);
        assert!(p.validate_config(&serde_json::json!({"any": 1})).is_ok());
        assert_eq!(
            p.resolve_ports(&serde_json::Value::Null).unwrap(),
            ResolvedPorts::from(p.spec())
        );
        assert_eq!(
            p.cache_policy(&ctx()).await.unwrap(),
            CachePolicy::default()
        );
    }

    #[test]
    fn overrides_validate_and_resolve() {
        let b = toy(false);
        assert!(b.validate_config(&serde_json::json!({})).is_ok());
        let err = b
            .validate_config(&serde_json::json!({"bad": 1}))
            .unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
        let rp = b
            .resolve_ports(&serde_json::json!({"out": "custom"}))
            .unwrap();
        assert_eq!(rp.outputs[0].name, "custom");
    }
}
