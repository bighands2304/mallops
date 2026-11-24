//! `std/train.external`: runs any external command as a training step through the versioned
//! file contract v1 (see [`contract`]), without Rust bindings (SPEC FR-TRN-3, ADR-0003).

pub(crate) mod contract;
pub(crate) mod process;

use std::collections::BTreeMap;
use std::path::PathBuf;

use mallops_core::{
    ArtifactKind, Block, BlockOutcome, BlockSpec, BoxFuture, CachePolicy, Context, Error, Inputs,
    Port, ResolvedPorts, Stage,
};
use serde::Deserialize;
// ponytail: placeholder until 13.1 logs the child output.
use tracing as _;

/// Default required input port.
pub(crate) const TRAIN: &str = "train";
/// Default optional input port.
pub(crate) const VAL: &str = "val";
/// Default output port.
pub(crate) const MODEL: &str = "model";
/// Default `log_cap_bytes` (per stream).
pub(crate) const DEFAULT_LOG_CAP: usize = 1024 * 1024;

/// `with:` of `std/train.external`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TrainExternalConfig {
    /// argv; elements starting with `./` or `../` are made absolute against mallops' cwd.
    pub(crate) command: Vec<String>,
    #[serde(default)]
    pub(crate) flavor: Option<String>,
    /// Free-form object written to `config.json`.
    #[serde(default)]
    pub(crate) params: Option<serde_json::Value>,
    /// Host environment variable names passed to the child.
    #[serde(default)]
    pub(crate) env: Vec<String>,
    /// Secret names resolved via `ctx.secret` and passed as env vars of the same name.
    #[serde(default)]
    pub(crate) secrets: Vec<String>,
    #[serde(default)]
    pub(crate) timeout_secs: Option<u64>,
    #[serde(default)]
    pub(crate) log_cap_bytes: Option<usize>,
    /// Files (relative to mallops' cwd) whose path+content make the block cacheable.
    #[serde(default)]
    pub(crate) watch_files: Vec<PathBuf>,
    /// Input ports (name → kind); replaces the defaults `train` + optional `val`.
    #[serde(default)]
    pub(crate) inputs: Option<BTreeMap<String, ArtifactKind>>,
    /// Output ports (name → kind); replaces the default `model`.
    #[serde(default)]
    pub(crate) outputs: Option<BTreeMap<String, ArtifactKind>>,
}

/// The `std/train.external` block.
pub(crate) struct TrainExternal(BlockSpec);

impl TrainExternal {
    pub(crate) fn new() -> Self {
        Self(
            BlockSpec::new("std/train.external", "0.1.0", Stage::Train)
                .with_input(Port::new(TRAIN, ArtifactKind::Features))
                .with_input(Port::new(VAL, ArtifactKind::Features).optional())
                .with_output(Port::new(MODEL, ArtifactKind::Model)),
        )
    }
}

impl Block for TrainExternal {
    fn spec(&self) -> &BlockSpec {
        &self.0
    }

    fn validate_config(&self, config: &serde_json::Value) -> Result<(), Error> {
        let _ = config;
        todo!("13.1")
    }

    fn resolve_ports(&self, config: &serde_json::Value) -> Result<ResolvedPorts, Error> {
        let _ = config;
        todo!("13.1")
    }

    fn cache_policy<'a>(&'a self, ctx: &'a Context) -> BoxFuture<'a, Result<CachePolicy, Error>> {
        Box::pin(async move {
            let _ = ctx;
            todo!("13.1")
        })
    }

    fn run<'a>(
        &'a self,
        ctx: &'a Context,
        inputs: Inputs,
    ) -> BoxFuture<'a, Result<BlockOutcome, Error>> {
        Box::pin(async move {
            let _ = (ctx, inputs);
            todo!("13.1")
        })
    }
}
