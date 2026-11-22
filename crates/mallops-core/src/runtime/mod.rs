//! Async DAG runtime: executes a [`ValidatedPipeline`] on `tokio` (SPEC §1, §4; ARCHITECTURE §4).
//!
//! The runtime schedules ready nodes concurrently (bounded by [`RuntimeConfig::max_jobs`]),
//! commits block output drafts to the store, propagates failure, skip and cancellation,
//! and reports per-node results in a [`RunSummary`].

mod commit;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use crate::{
    ArtifactId, ArtifactStore, CancelSource, CancelToken, EnvSecretProvider, Error, RunId,
    SecretProvider, ValidatedPipeline,
};

/// Runtime settings.
#[derive(Clone)]
#[non_exhaustive]
pub struct RuntimeConfig {
    /// Maximum number of blocks running at the same time (`--jobs`); `0` is treated as `1`.
    pub max_jobs: usize,
    /// Store receiving committed outputs and serving inputs.
    pub store: Arc<dyn ArtifactStore>,
    /// Provider behind `ctx.secret()`.
    pub secret_provider: Arc<dyn SecretProvider>,
    /// Directory under which each run gets an isolated, temporary workdir.
    pub workdir_root: PathBuf,
    /// External cancellation; blocks observe it through `ctx.cancel()`.
    pub cancel: CancelToken,
}

impl RuntimeConfig {
    /// Defaults: `max_jobs` = available parallelism, [`EnvSecretProvider`],
    /// `workdir_root` = [`std::env::temp_dir`], a cancel token that never fires.
    #[must_use]
    pub fn new(store: Arc<dyn ArtifactStore>) -> Self {
        Self {
            max_jobs: std::thread::available_parallelism().map_or(1, std::num::NonZero::get),
            store,
            secret_provider: Arc::new(EnvSecretProvider),
            workdir_root: std::env::temp_dir(),
            // A dropped source never fires (see `CancelToken::cancelled`).
            cancel: CancelSource::new().token(),
        }
    }
}

impl std::fmt::Debug for RuntimeConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeConfig")
            .field("max_jobs", &self.max_jobs)
            .field("store", &"<ArtifactStore>")
            .field("secret_provider", &"<SecretProvider>")
            .field("workdir_root", &self.workdir_root)
            .field("cancel", &self.cancel)
            .finish()
    }
}

/// Final state of a node in a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Status {
    /// The block ran and its outputs were committed.
    Succeeded,
    /// The block (or the commit of its outputs) failed.
    Failed,
    /// The block returned `BlockOutcome::Skipped`, or a required input came from a skipped node.
    Skipped,
    /// Not run (or stopped) because an upstream node failed or the run was cancelled.
    Cancelled,
}

/// Result of one node.
#[derive(Debug)]
#[non_exhaustive]
pub struct NodeSummary {
    /// Final status.
    pub status: Status,
    /// Wall-clock time spent on the node (zero if it never started).
    pub duration: Duration,
    /// Committed outputs by output port name.
    pub outputs: BTreeMap<String, ArtifactId>,
    /// Why the node was skipped or cancelled, when applicable.
    pub reason: Option<String>,
    /// The error of a [`Status::Failed`] node (or `Error::Cancelled` from a cancelled block).
    pub error: Option<Error>,
}

/// Result of a pipeline run.
#[derive(Debug)]
#[non_exhaustive]
pub struct RunSummary {
    /// Id of the run.
    pub run_id: RunId,
    /// Per-node results by node id.
    pub nodes: BTreeMap<String, NodeSummary>,
}

impl RunSummary {
    /// `true` when no node failed or was cancelled (skipped nodes are a normal outcome).
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.nodes
            .values()
            .all(|n| matches!(n.status, Status::Succeeded | Status::Skipped))
    }
}

/// Executes validated pipelines.
#[derive(Debug, Clone)]
pub struct Runtime {
    config: RuntimeConfig,
}

impl Runtime {
    /// Creates a runtime from its settings.
    #[must_use]
    pub fn new(config: RuntimeConfig) -> Self {
        Self { config }
    }

    /// Runs every node of `pipeline` and returns the per-node results.
    ///
    /// Node failures, skips and cancellation are reported in the [`RunSummary`].
    ///
    /// # Errors
    ///
    /// Only runtime-level failures (e.g. the run workdir cannot be created).
    pub async fn run(&self, pipeline: &ValidatedPipeline) -> Result<RunSummary, Error> {
        // 7.1: scheduler.
        let _ = (&self.config, pipeline, tracing::Span::none());
        Err(Error::Runtime("runtime not implemented yet".to_string()))
    }
}
