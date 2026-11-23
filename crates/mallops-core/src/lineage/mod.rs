//! Run lineage: events emitted by the runtime and the sinks that persist them (ADR-0006).
//!
//! Blocks never see lineage: the runtime is the only producer.

pub mod conformance;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use crate::{ArtifactId, ArtifactStore, BlockRef, BoxFuture, Digest, Error, RunId, Status};

/// Wall-clock instant of a lineage event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Timestamp(SystemTime);

impl Timestamp {
    /// Wraps a [`SystemTime`].
    #[must_use]
    pub fn new(time: SystemTime) -> Self {
        Self(time)
    }

    /// The wrapped [`SystemTime`].
    #[must_use]
    pub fn system_time(&self) -> SystemTime {
        self.0
    }
}

/// Host a run executed on.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct HostInfo {
    /// Operating system ([`std::env::consts::OS`]).
    pub os: String,
    /// CPU architecture ([`std::env::consts::ARCH`]).
    pub arch: String,
    /// Host name, `"unknown"` when not available.
    pub hostname: String,
    /// Version of `mallops-core`.
    pub mallops_version: String,
}

impl HostInfo {
    /// Describes the current host.
    #[must_use]
    pub fn current() -> Self {
        // ponytail: hostname from env only (no unsafe/libc); "unknown" when unset (common on macOS).
        let hostname = std::env::var("HOSTNAME")
            .or_else(|_| std::env::var("COMPUTERNAME"))
            .unwrap_or_else(|_| "unknown".into());
        Self {
            os: std::env::consts::OS.into(),
            arch: std::env::consts::ARCH.into(),
            hostname,
            mallops_version: env!("CARGO_PKG_VERSION").into(),
        }
    }
}

/// One lineage event of a run.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum LineageEvent {
    /// The run started.
    #[non_exhaustive]
    RunStarted {
        /// Run id.
        run: RunId,
        /// When.
        timestamp: Timestamp,
        /// Hash of the validated pipeline.
        pipeline_hash: Digest,
        /// Git commit of the pipeline sources, when known.
        git_commit: Option<String>,
        /// Host executing the run.
        host: HostInfo,
    },
    /// A node started (after acquiring its job slot).
    #[non_exhaustive]
    BlockStarted {
        /// Run id.
        run: RunId,
        /// When.
        timestamp: Timestamp,
        /// Node id.
        node: String,
        /// Block used by the node.
        block: BlockRef,
        /// `BlockSpec.version`.
        block_version: String,
        /// Hash of the node's `with:` config.
        config_hash: Digest,
        /// Node seed.
        seed: Option<u64>,
        /// Input artifact ids.
        inputs: Vec<ArtifactId>,
    },
    /// A node committed (or got from the cache) an output.
    #[non_exhaustive]
    ArtifactProduced {
        /// Run id.
        run: RunId,
        /// When.
        timestamp: Timestamp,
        /// Node id.
        node: String,
        /// Output artifact.
        artifact: ArtifactId,
    },
    /// A node reached its final status.
    #[non_exhaustive]
    BlockFinished {
        /// Run id.
        run: RunId,
        /// When.
        timestamp: Timestamp,
        /// Node id.
        node: String,
        /// Final status.
        status: Status,
        /// Time spent on the node (zero if it never started).
        duration: Duration,
        /// `true` when served from the cache.
        cached: bool,
    },
    /// The run ended.
    #[non_exhaustive]
    RunFinished {
        /// Run id.
        run: RunId,
        /// When.
        timestamp: Timestamp,
        /// Overall status.
        status: Status,
    },
}

/// Destination of lineage events. A failing sink never fails the run.
pub trait LineageSink: Send + Sync {
    /// Records one event.
    ///
    /// # Errors
    ///
    /// When the event cannot be persisted; the runtime logs it and continues.
    fn record<'a>(&'a self, event: &'a LineageEvent) -> BoxFuture<'a, Result<(), Error>>;
}

/// Core sink: writes `runs/<run-id>/lineage.json` through [`ArtifactStore::put_meta`].
pub struct JsonSink {
    store: Arc<dyn ArtifactStore>,
    runs: Mutex<HashMap<RunId, Vec<serde_json::Value>>>,
}

impl JsonSink {
    /// Creates a sink writing into `store`.
    #[must_use]
    pub fn new(store: Arc<dyn ArtifactStore>) -> Self {
        Self {
            store,
            runs: Mutex::new(HashMap::new()),
        }
    }
}

impl std::fmt::Debug for JsonSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsonSink").finish_non_exhaustive()
    }
}

impl LineageSink for JsonSink {
    fn record<'a>(&'a self, event: &'a LineageEvent) -> BoxFuture<'a, Result<(), Error>> {
        // 9.1: buffer per run, rewrite runs/<run-id>/lineage.json on every event.
        let _ = (event, &self.store, &self.runs);
        Box::pin(async { Ok(()) })
    }
}
