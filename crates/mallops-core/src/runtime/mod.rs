//! Async DAG runtime: executes a [`ValidatedPipeline`] on `tokio` (SPEC §1, §4; ARCHITECTURE §4).
//!
//! The runtime schedules ready nodes concurrently (bounded by [`RuntimeConfig::max_jobs`]),
//! commits block output drafts to the store, propagates failure, skip and cancellation,
//! and reports per-node results in a [`RunSummary`].

mod cache;
mod commit;

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tracing::Instrument;

use crate::lineage::{HostInfo, JsonSink, LineageSink};
use crate::{
    ArtifactId, ArtifactStore, BlockOutcome, CancelSource, CancelToken, Context, EnvSecretProvider,
    Error, Inputs, RunId, SecretProvider, ValidatedNode, ValidatedPipeline,
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
    /// Run-level cache bypass (`--no-cache`): every node executes; no cache entry is read or written.
    pub no_cache: bool,
    /// Lineage sinks; empty = `lineage: off`. Default: one [`JsonSink`] on `store`.
    pub lineage: Vec<Arc<dyn LineageSink>>,
    /// Clock for lineage timestamps (injectable for deterministic tests).
    pub clock: fn() -> SystemTime,
    /// Host recorded in `RunStarted`.
    pub host: HostInfo,
    /// Git commit recorded in `RunStarted` (set by the CLI; `None` in the library).
    pub git_commit: Option<String>,
}

impl RuntimeConfig {
    /// Defaults: `max_jobs` = available parallelism, [`EnvSecretProvider`],
    /// `workdir_root` = [`std::env::temp_dir`], a cancel token that never fires,
    /// lineage to a [`JsonSink`] on `store`, [`SystemTime::now`], [`HostInfo::current`].
    #[must_use]
    pub fn new(store: Arc<dyn ArtifactStore>) -> Self {
        Self {
            max_jobs: std::thread::available_parallelism().map_or(1, std::num::NonZero::get),
            lineage: vec![Arc::new(JsonSink::new(Arc::clone(&store)))],
            store,
            secret_provider: Arc::new(EnvSecretProvider),
            workdir_root: std::env::temp_dir(),
            // A dropped source never fires (see `CancelToken::cancelled`).
            cancel: CancelSource::new().token(),
            no_cache: false,
            clock: SystemTime::now,
            host: HostInfo::current(),
            git_commit: None,
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
            .field("no_cache", &self.no_cache)
            .field("lineage_sinks", &self.lineage.len())
            .field("host", &self.host)
            .field("git_commit", &self.git_commit)
            .finish()
    }
}

/// Final state of a node in a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
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
    /// `true` when the outputs were served from the cache and the block did not run.
    pub cached: bool,
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
        let run_id = RunId::new();
        let root = &self.config.workdir_root;
        // Removed when dropped at the end of the run.
        let run_dir = tempfile::Builder::new()
            .prefix(&format!("mallops-run-{run_id}-"))
            .tempdir_in(root)
            .map_err(|e| {
                Error::Runtime(format!(
                    "cannot create run workdir in {}: {e}",
                    root.display()
                ))
            })?;
        // Absolute and canonical even for a relative `workdir_root`, so `ctx.workdir()` is too.
        let run_path = std::fs::canonicalize(run_dir.path()).map_err(|e| {
            Error::Runtime(format!(
                "cannot canonicalize run workdir {}: {e}",
                run_dir.path().display()
            ))
        })?;
        let env = NodeEnv {
            run_id,
            store: Arc::clone(&self.config.store),
            secrets: Arc::clone(&self.config.secret_provider),
            cancel: self.config.cancel.clone(),
            jobs: Arc::new(Semaphore::new(self.config.max_jobs.max(1))),
            no_cache: self.config.no_cache,
        };

        let nodes = pipeline.nodes();
        let index: HashMap<&str, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.id.as_str(), i))
            .collect();
        let upstream: Vec<Vec<usize>> = nodes
            .iter()
            .map(|n| {
                n.config
                    .r#in
                    .values()
                    .filter_map(|src| index.get(upstream_node(src)).copied())
                    .collect()
            })
            .collect();
        let mut results: Vec<Option<NodeSummary>> =
            std::iter::repeat_with(|| None).take(nodes.len()).collect();
        let mut started = vec![false; nodes.len()];
        let mut tasks = JoinSet::new();
        let mut task_node = HashMap::new();

        loop {
            // Nodes are in topological order, so one pass also settles chains of
            // nodes that are cancelled or skipped without running.
            for (i, node) in nodes.iter().enumerate() {
                if started[i] || upstream[i].iter().any(|&u| results[u].is_none()) {
                    continue;
                }
                started[i] = true;
                match gate(node, &index, &results, &env.cancel) {
                    Err((status, reason)) => results[i] = Some(not_run(status, reason)),
                    Ok(inputs) => {
                        let workdir = run_path.join(format!("{i:04}"));
                        let task = run_node(node.clone(), inputs, workdir, env.clone());
                        let handle = tasks.spawn(task);
                        task_node.insert(handle.id(), i);
                    }
                }
            }
            let Some(joined) = tasks.join_next_with_id().await else {
                break;
            };
            let (task, summary) = match joined {
                Ok(done) => done,
                Err(e) => (
                    e.id(),
                    finished(Status::Failed, Duration::ZERO, BTreeMap::new(), None)
                        .with_error(Error::Runtime(format!("node task failed: {e}"))),
                ),
            };
            if let Some(i) = task_node.remove(&task) {
                results[i] = Some(summary);
            }
        }

        // Off the async thread: removing the run tree can take a while.
        let _ = tokio::task::spawn_blocking(move || drop(run_dir)).await;

        let nodes = nodes
            .iter()
            .zip(results)
            .map(|(n, r)| {
                let r = r.unwrap_or_else(|| not_run(Status::Cancelled, "not scheduled".into()));
                (n.id.clone(), r)
            })
            .collect();
        Ok(RunSummary { run_id, nodes })
    }
}

/// Per-run state shared by every node task.
#[derive(Clone)]
struct NodeEnv {
    run_id: RunId,
    store: Arc<dyn ArtifactStore>,
    secrets: Arc<dyn SecretProvider>,
    cancel: CancelToken,
    jobs: Arc<Semaphore>,
    no_cache: bool,
}

/// `<node>` of a validated `<node>.<port>` reference.
fn upstream_node(src: &str) -> &str {
    src.split_once('.').map_or(src, |(node, _)| node)
}

fn finished(
    status: Status,
    duration: Duration,
    outputs: BTreeMap<String, ArtifactId>,
    reason: Option<String>,
) -> NodeSummary {
    NodeSummary {
        status,
        duration,
        outputs,
        reason,
        error: None,
        cached: false,
    }
}

fn not_run(status: Status, reason: String) -> NodeSummary {
    finished(status, Duration::ZERO, BTreeMap::new(), Some(reason))
}

impl NodeSummary {
    fn with_error(mut self, error: Error) -> Self {
        self.error = Some(error);
        self
    }
}

/// Decides, once every upstream node has finished, whether `node` runs.
///
/// Returns the `(input port, artifact id)` pairs to run with, or the status and reason of
/// a node that does not run: `Cancelled` if an upstream failed or was cancelled (this wins over
/// skipping) or the run is cancelled; `Skipped` if a required input comes from a skipped
/// node or was not committed. Optional inputs without a committed artifact are left out.
fn gate(
    node: &ValidatedNode,
    index: &HashMap<&str, usize>,
    results: &[Option<NodeSummary>],
    cancel: &CancelToken,
) -> Result<Vec<(String, ArtifactId)>, (Status, String)> {
    let wired: Vec<(&str, &str, &str, &NodeSummary)> = node
        .config
        .r#in
        .iter()
        .filter_map(|(port, src)| {
            let (up, out) = src.split_once('.')?;
            let result = results[*index.get(up)?].as_ref()?;
            Some((port.as_str(), up, out, result))
        })
        .collect();

    if let Some((_, up, _, r)) = wired
        .iter()
        .find(|w| matches!(w.3.status, Status::Failed | Status::Cancelled))
    {
        let reason = match r.status {
            Status::Failed => format!("upstream node `{up}` failed"),
            // Keep the root cause (failed node or run cancellation) down the chain.
            _ => r
                .reason
                .clone()
                .unwrap_or_else(|| format!("upstream node `{up}` cancelled")),
        };
        return Err((Status::Cancelled, reason));
    }
    if cancel.is_cancelled() {
        return Err((Status::Cancelled, "run cancelled".into()));
    }

    let mut inputs = Vec::new();
    for (port, up, out, r) in wired {
        let required = node
            .ports
            .inputs
            .iter()
            .find(|p| p.name == port)
            .is_none_or(|p| p.required);
        match r.outputs.get(out) {
            _ if required && r.status == Status::Skipped => {
                let reason = format!("required input `{port}` comes from skipped node `{up}`");
                return Err((Status::Skipped, reason));
            }
            Some(id) => inputs.push((port.to_string(), *id)),
            None if required => {
                let reason = format!("required input `{port}`: `{up}.{out}` was not committed");
                return Err((Status::Skipped, reason));
            }
            None => {}
        }
    }
    Ok(inputs)
}

/// Runs one node: waits for a job slot, loads inputs, runs the block, commits its outputs.
///
/// The single per-node entry point where caching (#8) and lineage (#9) hook in.
async fn run_node(
    node: ValidatedNode,
    inputs: Vec<(String, ArtifactId)>,
    workdir: PathBuf,
    env: NodeEnv,
) -> NodeSummary {
    let Ok(_permit) = env.jobs.acquire().await else {
        return not_run(Status::Cancelled, "job pool closed".into());
    };
    if env.cancel.is_cancelled() {
        return not_run(Status::Cancelled, "run cancelled".into());
    }
    let span = tracing::info_span!("node", id = %node.id, block = %node.block.spec().id);
    let start = Instant::now();
    let result = execute(&node, inputs, workdir, &env).instrument(span).await;
    let duration = start.elapsed();
    match result {
        Ok((outputs, None, cached)) => {
            let mut summary = finished(Status::Succeeded, duration, outputs, None);
            summary.cached = cached;
            summary
        }
        Ok((outputs, Some(reason), _)) => {
            finished(Status::Skipped, duration, outputs, Some(reason))
        }
        Err(Error::Cancelled) => {
            let reason = if env.cancel.is_cancelled() {
                "run cancelled"
            } else {
                "block cancelled"
            };
            finished(
                Status::Cancelled,
                duration,
                BTreeMap::new(),
                Some(reason.into()),
            )
            .with_error(Error::Cancelled)
        }
        Err(e) => finished(Status::Failed, duration, BTreeMap::new(), None).with_error(e),
    }
}

/// Committed outputs, the reason for a skipped block, and whether they came from the cache.
type Executed = (BTreeMap<String, ArtifactId>, Option<String>, bool);

async fn execute(
    node: &ValidatedNode,
    inputs: Vec<(String, ArtifactId)>,
    workdir: PathBuf,
    env: &NodeEnv,
) -> Result<Executed, Error> {
    // ponytail: sync mkdir on the async thread, one cheap syscall per node.
    std::fs::create_dir(&workdir).map_err(|e| {
        Error::Runtime(format!(
            "cannot create node workdir {}: {e}",
            workdir.display()
        ))
    })?;
    let ctx = Context::new(
        env.run_id,
        workdir.clone(),
        Arc::clone(&env.store),
        node.config.with.clone(),
        env.cancel.clone(),
        Arc::clone(&env.secrets),
    );
    // `cache_policy` is only asked when the cache may be used.
    let key = if node.config.cache == Some(false) || env.no_cache {
        None
    } else {
        let policy = node.block.cache_policy(&ctx).await?;
        cache::cache_key(node, &inputs, &policy)
    };
    if let Some(key) = &key {
        // A store error while reading is a miss (symmetric with `record`); the entry is rewritten.
        match cache::lookup(&*env.store, key).await {
            Ok(Some(outputs)) => return Ok((outputs, None, true)),
            Ok(None) => {}
            Err(e) => tracing::warn!(error = %e, "cannot read cache entry, running the block"),
        }
    }
    let mut block_inputs = Inputs::new();
    let mut parents = Vec::with_capacity(inputs.len());
    for (port, id) in inputs {
        let artifact = env
            .store
            .get(&id)
            .await?
            .ok_or_else(|| Error::Runtime(format!("input artifact {id} not found in store")))?;
        parents.push(id);
        block_inputs = block_inputs.with(port, artifact);
    }
    let (drafts, produced, reason) = match node.block.run(&ctx, block_inputs).await? {
        BlockOutcome::Produced(drafts) => (drafts, true, None),
        BlockOutcome::Skipped { reason, outputs } => (outputs, false, Some(reason)),
    };
    let outputs = commit::commit_outputs(
        &*env.store,
        &node.ports,
        drafts,
        &parents,
        &workdir,
        produced,
    )
    .await?;
    if let (Some(key), None) = (&key, &reason) {
        if let Err(e) = cache::record(&*env.store, key, &outputs).await {
            tracing::warn!(error = %e, "cannot record cache entry");
        }
    }
    Ok((outputs, reason, false))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use tokio::sync::{Barrier, Notify};

    use super::*;
    use crate::{
        ArtifactKind, Block, BlockRegistry, BlockSpec, BoxFuture, DraftPayload, LocalStore,
        Manifest, MemoryStore, OutputDraft, OutputDrafts, PipelineBuilder, Port, Stage,
    };

    const LIMIT: Duration = Duration::from_secs(10);

    /// Shared observation points of the test blocks.
    #[derive(Default)]
    struct Probe {
        barrier: Option<Barrier>,
        calls: AtomicUsize,
        running: AtomicUsize,
        peak: AtomicUsize,
        started: Notify,
    }

    /// Test block whose behavior is picked by `with: {"mode": ...}`:
    /// `ok` (default), `fail`, `panic`, `skip`, `skip_out` (skipped, still provides `out`),
    /// `barrier`, `wait_cancel`, `file` / `dir` (payload written in `ctx.workdir()`).
    struct TestBlock {
        spec: BlockSpec,
        probe: Arc<Probe>,
    }

    impl Block for TestBlock {
        fn spec(&self) -> &BlockSpec {
            &self.spec
        }

        fn run<'a>(
            &'a self,
            ctx: &'a Context,
            inputs: Inputs,
        ) -> BoxFuture<'a, Result<BlockOutcome, Error>> {
            Box::pin(async move {
                let p = &self.probe;
                p.calls.fetch_add(1, Ordering::SeqCst);
                let now = p.running.fetch_add(1, Ordering::SeqCst) + 1;
                p.peak.fetch_max(now, Ordering::SeqCst);
                for _ in 0..10 {
                    tokio::task::yield_now().await;
                }
                let mode = ctx.config()["mode"].as_str().unwrap_or("ok");
                let result = match mode {
                    "fail" => Err(Error::Runtime("boom".into())),
                    "panic" => panic!("block panicked on purpose"),
                    "skip_out" => Ok(BlockOutcome::Skipped {
                        reason: "partial".into(),
                        outputs: OutputDrafts::new().with("out", blob(b"partial".to_vec())),
                    }),
                    "file" | "dir" => {
                        let canonical = std::fs::canonicalize(ctx.workdir()).unwrap();
                        if ctx.workdir() != canonical {
                            return Err(Error::Runtime("workdir is not canonical".into()));
                        }
                        let dir = ctx.workdir().join("dir");
                        std::fs::create_dir(&dir).unwrap();
                        std::fs::write(dir.join("a.bin"), b"a").unwrap();
                        std::fs::write(ctx.workdir().join("out.bin"), b"o").unwrap();
                        let payload = if mode == "file" {
                            DraftPayload::File(ctx.workdir().join("out.bin"))
                        } else {
                            DraftPayload::Dir(dir)
                        };
                        let draft =
                            OutputDraft::new(Manifest::new(ArtifactKind::Blob, mode), payload);
                        Ok(BlockOutcome::Produced(
                            OutputDrafts::new().with("out", draft),
                        ))
                    }
                    "skip" => Ok(BlockOutcome::Skipped {
                        reason: "gate closed".into(),
                        outputs: OutputDrafts::new(),
                    }),
                    "bad_kind" => Ok(BlockOutcome::Produced(bad_drafts("out"))),
                    "bad_port" => Ok(BlockOutcome::Produced(bad_drafts("bogus"))),
                    "wait_cancel" => {
                        p.started.notify_one();
                        ctx.cancel().cancelled().await;
                        Err(Error::Cancelled)
                    }
                    _ => {
                        if mode == "barrier" {
                            p.barrier.as_ref().expect("barrier").wait().await;
                        }
                        let n = inputs.iter().count();
                        let draft = OutputDraft::new(
                            Manifest::new(ArtifactKind::Blob, "text"),
                            DraftPayload::Bytes(format!("{mode}{n}").into_bytes()),
                        );
                        Ok(BlockOutcome::Produced(
                            OutputDrafts::new().with("out", draft),
                        ))
                    }
                };
                p.running.fetch_sub(1, Ordering::SeqCst);
                result
            })
        }
    }

    fn blob(bytes: Vec<u8>) -> OutputDraft {
        OutputDraft::new(
            Manifest::new(ArtifactKind::Blob, "text"),
            DraftPayload::Bytes(bytes),
        )
    }

    /// A `Dataset` draft on `port`: wrong kind for the `Blob` output `out`.
    fn bad_drafts(port: &str) -> OutputDrafts {
        let draft = OutputDraft::new(
            Manifest::new(ArtifactKind::Dataset, "csv"),
            DraftPayload::Bytes(b"bad".to_vec()),
        );
        OutputDrafts::new().with(port, draft)
    }

    /// `t/src` (no input), `t/one` (required `a`), `t/two` (required `a`, `b`),
    /// `t/opt` (optional `a`); all with output `out`.
    fn registry(probe: &Arc<Probe>) -> BlockRegistry {
        let blob = |n: &'static str| Port::new(n, ArtifactKind::Blob);
        let specs = [
            BlockSpec::new("t/src", "1", Stage::Custom),
            BlockSpec::new("t/one", "1", Stage::Custom).with_input(blob("a")),
            BlockSpec::new("t/two", "1", Stage::Custom)
                .with_input(blob("a"))
                .with_input(blob("b")),
            BlockSpec::new("t/opt", "1", Stage::Custom).with_input(blob("a").optional()),
        ];
        let mut reg = BlockRegistry::new();
        for spec in specs {
            let spec = spec.with_output(blob("out"));
            let probe = Arc::clone(probe);
            reg.register(Arc::new(TestBlock { spec, probe })).unwrap();
        }
        reg
    }

    /// Adds node `id` using block `t/<block>` with `mode` and `(port, source)` inputs.
    fn node(b: &mut PipelineBuilder, id: &str, block: &str, mode: &str, ins: &[(&str, &str)]) {
        let mut n = b.block(id, format!("t/{block}").parse().unwrap());
        n.with(serde_json::json!({ "mode": mode }));
        for (port, src) in ins {
            n.input(*port, *src);
        }
    }

    fn runtime(store: Arc<MemoryStore>, max_jobs: usize) -> Runtime {
        let mut cfg = RuntimeConfig::new(store);
        cfg.max_jobs = max_jobs;
        Runtime::new(cfg)
    }

    async fn run(rt: &Runtime, p: &ValidatedPipeline) -> RunSummary {
        tokio::time::timeout(LIMIT, rt.run(p))
            .await
            .expect("run timed out")
            .unwrap()
    }

    fn status(s: &RunSummary, id: &str) -> Status {
        s.nodes[id].status
    }

    fn reason<'a>(s: &'a RunSummary, id: &str) -> &'a str {
        s.nodes[id].reason.as_deref().unwrap_or_default()
    }

    #[tokio::test]
    async fn runtime_runs_independent_branches_concurrently() {
        let probe = Arc::new(Probe {
            barrier: Some(Barrier::new(2)),
            ..Probe::default()
        });
        let mut b = PipelineBuilder::new("t");
        node(&mut b, "x", "src", "barrier", &[]);
        node(&mut b, "y", "src", "barrier", &[]);
        let p = b.build(&registry(&probe)).unwrap();
        // Both blocks must be in flight together to pass the barrier.
        let s = run(&runtime(Arc::new(MemoryStore::new()), 2), &p).await;
        assert!(s.is_success());
        assert_eq!(status(&s, "x"), Status::Succeeded);
        assert_eq!(status(&s, "y"), Status::Succeeded);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn runtime_max_jobs_one_runs_one_block_at_a_time() {
        let probe = Arc::new(Probe::default());
        let mut b = PipelineBuilder::new("t");
        for id in ["a", "b", "c", "d"] {
            node(&mut b, id, "src", "ok", &[]);
        }
        let p = b.build(&registry(&probe)).unwrap();
        let s = run(&runtime(Arc::new(MemoryStore::new()), 1), &p).await;
        assert!(s.is_success());
        assert_eq!(probe.peak.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn runtime_failure_cancels_descendants_only() {
        let probe = Arc::new(Probe::default());
        let mut b = PipelineBuilder::new("t");
        node(&mut b, "bad", "src", "fail", &[]);
        node(&mut b, "child", "one", "ok", &[("a", "bad.out")]);
        node(&mut b, "grandchild", "one", "ok", &[("a", "child.out")]);
        node(&mut b, "good", "src", "ok", &[]);
        node(&mut b, "good2", "one", "ok", &[("a", "good.out")]);
        let p = b.build(&registry(&probe)).unwrap();
        let s = run(&runtime(Arc::new(MemoryStore::new()), 4), &p).await;

        assert!(!s.is_success());
        assert_eq!(status(&s, "bad"), Status::Failed);
        assert!(matches!(s.nodes["bad"].error, Some(Error::Runtime(_))));
        for id in ["child", "grandchild"] {
            assert_eq!(status(&s, id), Status::Cancelled);
            assert!(
                reason(&s, id).contains("`bad` failed"),
                "{}",
                reason(&s, id)
            );
            assert_eq!(s.nodes[id].duration, Duration::ZERO);
        }
        assert_eq!(status(&s, "good"), Status::Succeeded);
        assert_eq!(status(&s, "good2"), Status::Succeeded);
    }

    #[tokio::test]
    async fn runtime_skip_propagates_to_required_consumers() {
        let probe = Arc::new(Probe::default());
        let mut b = PipelineBuilder::new("t");
        node(&mut b, "gate", "src", "skip", &[]);
        node(&mut b, "ok", "src", "ok", &[]);
        node(&mut b, "consumer", "one", "ok", &[("a", "gate.out")]);
        node(
            &mut b,
            "fanin",
            "two",
            "ok",
            &[("a", "ok.out"), ("b", "gate.out")],
        );
        node(&mut b, "after", "one", "ok", &[("a", "fanin.out")]);
        node(&mut b, "optional", "opt", "ok", &[("a", "gate.out")]);
        node(&mut b, "bad", "src", "fail", &[]);
        node(
            &mut b,
            "mixed",
            "two",
            "ok",
            &[("a", "bad.out"), ("b", "gate.out")],
        );
        let p = b.build(&registry(&probe)).unwrap();
        let s = run(&runtime(Arc::new(MemoryStore::new()), 4), &p).await;

        assert_eq!(status(&s, "gate"), Status::Skipped);
        assert_eq!(reason(&s, "gate"), "gate closed");
        assert_eq!(status(&s, "ok"), Status::Succeeded);
        for id in ["consumer", "fanin"] {
            assert_eq!(status(&s, id), Status::Skipped);
            assert!(
                reason(&s, id).contains("skipped node `gate`"),
                "{}",
                reason(&s, id)
            );
        }
        assert_eq!(status(&s, "after"), Status::Skipped);
        assert!(reason(&s, "after").contains("`fanin`"));
        // Optional input without a committed artifact is simply absent.
        assert_eq!(status(&s, "optional"), Status::Succeeded);
        // Failed parent wins over skipped parent.
        assert_eq!(status(&s, "mixed"), Status::Cancelled);
    }

    #[tokio::test]
    async fn runtime_cancel_token_stops_in_flight_and_pending_nodes() {
        let probe = Arc::new(Probe::default());
        let mut b = PipelineBuilder::new("t");
        node(&mut b, "wait", "src", "wait_cancel", &[]);
        node(&mut b, "next", "one", "ok", &[("a", "wait.out")]);
        let p = b.build(&registry(&probe)).unwrap();
        let source = CancelSource::new();
        let mut cfg = RuntimeConfig::new(Arc::new(MemoryStore::new()));
        cfg.cancel = source.token();
        let rt = Runtime::new(cfg);

        let handle = tokio::spawn(async move { rt.run(&p).await });
        tokio::time::timeout(LIMIT, probe.started.notified())
            .await
            .expect("block never started");
        source.cancel();
        let s = tokio::time::timeout(LIMIT, handle)
            .await
            .expect("run timed out")
            .unwrap()
            .unwrap();

        assert_eq!(status(&s, "wait"), Status::Cancelled);
        assert!(matches!(s.nodes["wait"].error, Some(Error::Cancelled)));
        assert_eq!(status(&s, "next"), Status::Cancelled);
        assert_eq!(reason(&s, "next"), "run cancelled");
        assert!(!s.is_success());
    }

    #[tokio::test]
    async fn runtime_committed_outputs_have_input_parents() {
        let probe = Arc::new(Probe::default());
        let mut b = PipelineBuilder::new("t");
        node(&mut b, "a", "src", "ok", &[]);
        // A different mode gives different bytes, hence a distinct id from `a`.
        node(&mut b, "b", "src", "ok-too", &[]);
        node(
            &mut b,
            "join",
            "two",
            "ok",
            &[("a", "a.out"), ("b", "b.out")],
        );
        let p = b.build(&registry(&probe)).unwrap();
        let store = Arc::new(MemoryStore::new());
        let root = tempfile::tempdir().unwrap();
        let mut cfg = RuntimeConfig::new(Arc::clone(&store) as Arc<dyn ArtifactStore>);
        cfg.workdir_root = root.path().to_path_buf();
        let s = run(&Runtime::new(cfg), &p).await;
        assert!(s.is_success());

        let ids = |n: &str| s.nodes[n].outputs["out"];
        let joined = store.get(&ids("join")).await.unwrap().unwrap();
        let mut expected = vec![ids("a"), ids("b")];
        expected.sort();
        assert_ne!(ids("a"), ids("b"));
        assert_eq!(joined.manifest().parents, expected);
        let source = store.get(&ids("a")).await.unwrap().unwrap();
        assert!(source.manifest().parents.is_empty());
        // The per-run workdir is removed at the end of the run.
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
    }

    #[tokio::test]
    async fn runtime_draft_violating_output_port_fails_node() {
        let probe = Arc::new(Probe::default());
        let mut b = PipelineBuilder::new("t");
        node(&mut b, "kind", "src", "bad_kind", &[]);
        node(&mut b, "port", "src", "bad_port", &[]);
        node(&mut b, "after", "one", "ok", &[("a", "kind.out")]);
        node(&mut b, "ok", "src", "ok", &[]);
        let p = b.build(&registry(&probe)).unwrap();
        let store = Arc::new(MemoryStore::new());
        let s = run(&runtime(Arc::clone(&store), 4), &p).await;

        for (id, port) in [("kind", "out.out"), ("port", "out.bogus")] {
            assert_eq!(status(&s, id), Status::Failed);
            let Some(Error::Validation(errs)) = &s.nodes[id].error else {
                panic!(
                    "{id}: expected validation error, got {:?}",
                    s.nodes[id].error
                )
            };
            assert!(errs.iter().any(|e| e.path == port), "{id}: {errs:?}");
            assert!(s.nodes[id].outputs.is_empty());
        }
        assert_eq!(status(&s, "after"), Status::Cancelled);
        assert!(reason(&s, "after").contains("`kind` failed"));
        assert_eq!(status(&s, "ok"), Status::Succeeded);
        // Ids are content-addressed: the rejected draft would have had this id.
        let draft = bad_drafts("out").into_iter().next().unwrap().1;
        let rejected = MemoryStore::new()
            .put(draft.manifest, draft.payload.as_source())
            .await
            .unwrap();
        assert!(store.get(&rejected).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn runtime_panicking_block_fails_without_deadlock() {
        let probe = Arc::new(Probe::default());
        let mut b = PipelineBuilder::new("t");
        node(&mut b, "boom", "src", "panic", &[]);
        node(&mut b, "child", "one", "ok", &[("a", "boom.out")]);
        node(&mut b, "good", "src", "ok", &[]);
        let p = b.build(&registry(&probe)).unwrap();
        // With a single job slot, the panicking task must still release it.
        let s = run(&runtime(Arc::new(MemoryStore::new()), 1), &p).await;

        assert_eq!(status(&s, "boom"), Status::Failed);
        assert!(matches!(s.nodes["boom"].error, Some(Error::Runtime(_))));
        assert_eq!(s.nodes["boom"].duration, Duration::ZERO);
        assert_eq!(status(&s, "child"), Status::Cancelled);
        assert!(reason(&s, "child").contains("`boom` failed"));
        assert_eq!(status(&s, "good"), Status::Succeeded);
    }

    #[tokio::test]
    async fn runtime_max_jobs_zero_runs_like_one() {
        let probe = Arc::new(Probe::default());
        let mut b = PipelineBuilder::new("t");
        node(&mut b, "a", "src", "ok", &[]);
        // Different `with` from `a`: an identical node would be a cache hit.
        node(&mut b, "b", "src", "ok-too", &[]);
        node(&mut b, "c", "one", "ok", &[("a", "a.out")]);
        let p = b.build(&registry(&probe)).unwrap();
        let s = run(&runtime(Arc::new(MemoryStore::new()), 0), &p).await;
        assert!(s.is_success());
        assert_eq!(probe.calls.load(Ordering::SeqCst), 3);
        assert_eq!(probe.peak.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn runtime_cancelled_before_run_invokes_no_block() {
        let probe = Arc::new(Probe::default());
        let mut b = PipelineBuilder::new("t");
        node(&mut b, "a", "src", "ok", &[]);
        node(&mut b, "b", "one", "ok", &[("a", "a.out")]);
        let p = b.build(&registry(&probe)).unwrap();
        let source = CancelSource::new();
        source.cancel();
        let mut cfg = RuntimeConfig::new(Arc::new(MemoryStore::new()));
        cfg.cancel = source.token();
        let s = run(&Runtime::new(cfg), &p).await;

        for id in ["a", "b"] {
            assert_eq!(status(&s, id), Status::Cancelled);
            assert_eq!(reason(&s, id), "run cancelled");
        }
        assert_eq!(probe.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn runtime_skipped_output_reaches_optional_consumer_only() {
        let probe = Arc::new(Probe::default());
        let mut b = PipelineBuilder::new("t");
        node(&mut b, "gate", "src", "skip_out", &[]);
        node(&mut b, "optional", "opt", "ok", &[("a", "gate.out")]);
        node(&mut b, "required", "one", "ok", &[("a", "gate.out")]);
        let p = b.build(&registry(&probe)).unwrap();
        let store = Arc::new(MemoryStore::new());
        let s = run(&runtime(Arc::clone(&store), 2), &p).await;

        assert!(s.is_success());
        assert_eq!(status(&s, "gate"), Status::Skipped);
        let provided = s.nodes["gate"].outputs["out"];
        assert_eq!(status(&s, "optional"), Status::Succeeded);
        // The optional consumer ran with the skipped node's output as input.
        let out = store
            .get(&s.nodes["optional"].outputs["out"])
            .await
            .unwrap();
        assert_eq!(out.unwrap().manifest().parents, [provided]);
        assert_eq!(status(&s, "required"), Status::Skipped);
        assert!(reason(&s, "required").contains("skipped node `gate`"));
    }

    /// `ctx.workdir()` is canonical for a relative `workdir_root` and for one behind a
    /// symlink (the system temp dir on macOS), and `File`/`Dir` payloads in it commit.
    #[tokio::test]
    async fn runtime_relative_workdir_root_commits_file_and_dir_on_local_store() {
        let probe = Arc::new(Probe::default());
        let mut b = PipelineBuilder::new("t");
        node(&mut b, "file", "src", "file", &[]);
        node(&mut b, "dir", "src", "dir", &[]);
        let p = b.build(&registry(&probe)).unwrap();
        let store_root = tempfile::tempdir().unwrap();
        let store = Arc::new(LocalStore::new(store_root.path()).unwrap());

        // A relative root under the cwd (which is never changed), then the system temp dir.
        let local = tempfile::tempdir_in(".").unwrap();
        let cwd = std::env::current_dir().unwrap();
        let relative = local.path().strip_prefix(&cwd).unwrap().to_path_buf();
        assert!(relative.is_relative());
        let system = tempfile::tempdir().unwrap();
        for root in [relative, system.path().to_path_buf()] {
            let mut cfg = RuntimeConfig::new(Arc::clone(&store) as Arc<dyn ArtifactStore>);
            cfg.workdir_root = root.clone();
            let s = run(&Runtime::new(cfg), &p).await;
            for id in ["file", "dir"] {
                assert_eq!(status(&s, id), Status::Succeeded, "{id}: {:?}", s.nodes[id]);
                store.verify(&s.nodes[id].outputs["out"]).await.unwrap();
            }
            assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0, "{root:?}");
        }
    }

    /// No-op block: optional input `a`, emits an empty `out`.
    struct Noop(BlockSpec);

    impl Block for Noop {
        fn spec(&self) -> &BlockSpec {
            &self.0
        }

        fn run<'a>(
            &'a self,
            _: &'a Context,
            _: Inputs,
        ) -> BoxFuture<'a, Result<BlockOutcome, Error>> {
            Box::pin(async {
                let draft = OutputDraft::new(
                    Manifest::new(ArtifactKind::Blob, "empty"),
                    DraftPayload::Bytes(Vec::new()),
                );
                Ok(BlockOutcome::Produced(
                    OutputDrafts::new().with("out", draft),
                ))
            })
        }
    }

    /// NFR-1: median core overhead per block < 5 ms on `MemoryStore`.
    /// Run with `cargo test -p mallops-core runtime -- --ignored --nocapture`.
    #[tokio::test]
    #[ignore = "benchmark (NFR-1)"]
    async fn runtime_nfr1_block_overhead() {
        const N: usize = 200;
        const RUNS: usize = 9;
        let mut reg = BlockRegistry::new();
        let spec = BlockSpec::new("t/noop", "1", Stage::Custom)
            .with_input(Port::new("a", ArtifactKind::Blob).optional())
            .with_output(Port::new("out", ArtifactKind::Blob));
        reg.register(Arc::new(Noop(spec))).unwrap();
        // A chain: every block waits for the previous one, so overheads add up.
        let mut b = PipelineBuilder::new("nfr1");
        b.block("n0", "t/noop".parse().unwrap());
        for i in 1..N {
            b.block(format!("n{i}"), "t/noop".parse().unwrap())
                .input("a", format!("n{}.out", i - 1));
        }
        let p = b.build(&reg).unwrap();
        let rt = runtime(Arc::new(MemoryStore::new()), 4);

        assert!(run(&rt, &p).await.is_success()); // warm-up
        let mut per_block = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let start = Instant::now();
            assert!(run(&rt, &p).await.is_success());
            per_block.push(start.elapsed() / u32::try_from(N).unwrap());
        }
        per_block.sort();
        let median = per_block[RUNS / 2];
        eprintln!("NFR-1 median overhead per block: {median:?} ({N} blocks, {RUNS} runs)");
        assert!(median < Duration::from_millis(5), "median {median:?}");
    }

    // ---- cache (#8): scenarios run on `MemoryStore` and `LocalStore` ----

    use std::path::Path;
    use std::sync::Mutex;

    use crate::{CachePolicy, Digest, MemorySecretProvider, PayloadSource, RelPath};

    type Log = Arc<Mutex<Vec<String>>>;

    /// Deterministic block: its output bytes depend on version, policy, `with:` and inputs,
    /// so a change anywhere upstream changes every downstream input id. Logs `with.name`
    /// when it runs; `with.fail == true` makes it fail.
    struct CacheBlock {
        spec: BlockSpec,
        policy: CachePolicy,
        log: Log,
    }

    impl Block for CacheBlock {
        fn spec(&self) -> &BlockSpec {
            &self.spec
        }

        fn cache_policy<'a>(
            &'a self,
            _ctx: &'a Context,
        ) -> BoxFuture<'a, Result<CachePolicy, Error>> {
            Box::pin(async { Ok(self.policy.clone()) })
        }

        fn run<'a>(
            &'a self,
            ctx: &'a Context,
            inputs: Inputs,
        ) -> BoxFuture<'a, Result<BlockOutcome, Error>> {
            Box::pin(async move {
                let name = ctx.config()["name"].as_str().unwrap_or("?").to_string();
                self.log.lock().unwrap().push(name);
                if ctx.config()["fail"] == true {
                    return Err(Error::Runtime("boom".into()));
                }
                let mut text = format!("{:?}|{}|{}", self.policy, self.spec.version, ctx.config());
                for (port, a) in inputs.iter() {
                    text.push_str(&format!("|{port}={}", a.id()));
                }
                Ok(BlockOutcome::Produced(
                    OutputDrafts::new().with("out", blob(text.into_bytes())),
                ))
            })
        }
    }

    /// What a scenario may vary on the pipeline `a -> b -> c` plus the independent `x`.
    #[derive(Clone)]
    struct Setup {
        b_version: &'static str,
        b_policy: CachePolicy,
        b_with: serde_json::Value,
        b_cache: Option<bool>,
    }

    impl Default for Setup {
        fn default() -> Self {
            Self {
                b_version: "1",
                b_policy: CachePolicy::default(),
                b_with: serde_json::json!({ "name": "b" }),
                b_cache: None,
            }
        }
    }

    fn cache_pipeline(log: &Log, setup: &Setup) -> ValidatedPipeline {
        let blob_port = |n: &'static str| Port::new(n, ArtifactKind::Blob);
        let mut reg = BlockRegistry::new();
        let blocks = [
            ("c/src", "1", CachePolicy::default(), false),
            ("c/mid", setup.b_version, setup.b_policy.clone(), true),
            ("c/one", "1", CachePolicy::default(), true),
        ];
        for (id, version, policy, has_input) in blocks {
            let mut spec = BlockSpec::new(id, version, Stage::Custom).with_output(blob_port("out"));
            if has_input {
                spec = spec.with_input(blob_port("a"));
            }
            let block = CacheBlock {
                spec,
                policy,
                log: Arc::clone(log),
            };
            reg.register(Arc::new(block)).unwrap();
        }
        let mut b = PipelineBuilder::new("t");
        let named = |n: &str| serde_json::json!({ "name": n });
        b.block("a", "c/src".parse().unwrap()).with(named("a"));
        b.block("x", "c/src".parse().unwrap()).with(named("x"));
        {
            let mut n = b.block("b", "c/mid".parse().unwrap());
            n.with(setup.b_with.clone()).input("a", "a.out");
            if let Some(enabled) = setup.b_cache {
                n.cache(enabled);
            }
        }
        b.block("c", "c/one".parse().unwrap())
            .with(named("c"))
            .input("a", "b.out");
        b.build(&reg).unwrap()
    }

    /// Runs the pipeline for `setup`; returns the summary and the sorted names of executed blocks.
    async fn cache_run(
        store: &Arc<dyn ArtifactStore>,
        log: &Log,
        setup: &Setup,
        no_cache: bool,
        secrets: MemorySecretProvider,
    ) -> (RunSummary, Vec<String>) {
        let mut cfg = RuntimeConfig::new(Arc::clone(store));
        cfg.max_jobs = 2;
        cfg.no_cache = no_cache;
        cfg.secret_provider = Arc::new(secrets);
        let s = run(&Runtime::new(cfg), &cache_pipeline(log, setup)).await;
        let mut executed = std::mem::take(&mut *log.lock().unwrap());
        executed.sort();
        (s, executed)
    }

    async fn cache_run_plain(
        store: &Arc<dyn ArtifactStore>,
        log: &Log,
        setup: &Setup,
    ) -> (RunSummary, Vec<String>) {
        cache_run(store, log, setup, false, MemorySecretProvider::default()).await
    }

    fn cached_nodes(s: &RunSummary) -> Vec<&str> {
        s.nodes
            .iter()
            .filter(|(_, n)| n.cached)
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// Runs `$scenario` once per store kind.
    macro_rules! cache_tests {
        ($($name:ident => $scenario:ident),* $(,)?) => {$(
            mod $name {
                use super::*;

                #[tokio::test]
                async fn memory() {
                    $scenario(Arc::new(MemoryStore::new())).await;
                }

                #[tokio::test]
                async fn local() {
                    let dir = tempfile::tempdir().unwrap();
                    $scenario(Arc::new(LocalStore::new(dir.path()).unwrap())).await;
                }
            }
        )*};
    }

    cache_tests! {
        cache_rerun_unchanged_is_fully_cached => scenario_rerun,
        cache_changed_with_reruns_node_and_descendants => scenario_with,
        cache_changed_version_reruns_node_and_descendants => scenario_version,
        cache_changed_extra_hash_reruns_node_and_descendants => scenario_extra_hash,
        cache_rotated_secret_value_keeps_hit => scenario_secret,
        cache_uncacheable_always_executes => scenario_uncacheable,
        cache_node_flag_false_always_executes => scenario_node_flag,
        cache_no_cache_runtime_flag_executes_everything => scenario_no_cache,
        cache_resume_after_failure_reruns_failed_and_descendants => scenario_resume,
        cache_missing_output_artifact_is_a_miss => scenario_missing_output,
        cache_store_error_on_lookup_is_a_miss => scenario_lookup_error,
    }

    async fn scenario_rerun(store: Arc<dyn ArtifactStore>) {
        let log = Log::default();
        let setup = Setup::default();
        let (s, executed) = cache_run_plain(&store, &log, &setup).await;
        assert_eq!(executed, ["a", "b", "c", "x"]);
        assert!(cached_nodes(&s).is_empty());
        let (s2, executed) = cache_run_plain(&store, &log, &setup).await;
        assert!(executed.is_empty(), "{executed:?}");
        for (id, n) in &s2.nodes {
            assert_eq!(n.status, Status::Succeeded, "{id}");
            assert!(n.cached, "{id}");
            assert_eq!(n.outputs, s.nodes[id].outputs, "{id}");
        }
    }

    /// Baseline run, then `changed` must re-execute exactly `["b", "c"]`.
    async fn rerun_b_and_c(store: Arc<dyn ArtifactStore>, changed: Setup) {
        let log = Log::default();
        cache_run_plain(&store, &log, &Setup::default()).await;
        let (s, executed) = cache_run_plain(&store, &log, &changed).await;
        assert_eq!(executed, ["b", "c"]);
        assert_eq!(cached_nodes(&s), ["a", "x"]);
    }

    async fn scenario_with(store: Arc<dyn ArtifactStore>) {
        let changed = Setup {
            b_with: serde_json::json!({ "name": "b", "k": 2 }),
            ..Setup::default()
        };
        rerun_b_and_c(store, changed).await;
    }

    async fn scenario_version(store: Arc<dyn ArtifactStore>) {
        let changed = Setup {
            b_version: "2",
            ..Setup::default()
        };
        rerun_b_and_c(store, changed).await;
    }

    async fn scenario_extra_hash(store: Arc<dyn ArtifactStore>) {
        let changed = Setup {
            b_policy: CachePolicy::Deterministic {
                extra_hash: Some(Digest::hash(b"script v2")),
            },
            ..Setup::default()
        };
        rerun_b_and_c(store, changed).await;
    }

    async fn scenario_secret(store: Arc<dyn ArtifactStore>) {
        let log = Log::default();
        let setup = Setup {
            b_with: serde_json::json!({ "name": "b", "token": "${secret:TOKEN}" }),
            ..Setup::default()
        };
        let secrets = |v: &str| MemorySecretProvider::default().with_secret("TOKEN", v);
        let (s, executed) = cache_run(&store, &log, &setup, false, secrets("one")).await;
        assert!(s.is_success());
        assert_eq!(executed.len(), 4);
        let (s, executed) = cache_run(&store, &log, &setup, false, secrets("two")).await;
        assert!(executed.is_empty(), "{executed:?}");
        assert_eq!(cached_nodes(&s), ["a", "b", "c", "x"]);
    }

    /// With `bypass` on `b`, a rerun executes `b` (never served from cache); `c` gets the same
    /// input id back and stays a hit.
    async fn bypassed_b(store: Arc<dyn ArtifactStore>, setup: Setup) {
        let log = Log::default();
        cache_run_plain(&store, &log, &setup).await;
        let (s, executed) = cache_run_plain(&store, &log, &setup).await;
        assert_eq!(executed, ["b"]);
        assert!(!s.nodes["b"].cached);
        assert_eq!(s.nodes["b"].status, Status::Succeeded);
        assert_eq!(cached_nodes(&s), ["a", "c", "x"]);
    }

    async fn scenario_uncacheable(store: Arc<dyn ArtifactStore>) {
        let setup = Setup {
            b_policy: CachePolicy::Uncacheable,
            ..Setup::default()
        };
        bypassed_b(store, setup).await;
    }

    async fn scenario_node_flag(store: Arc<dyn ArtifactStore>) {
        let setup = Setup {
            b_cache: Some(false),
            ..Setup::default()
        };
        bypassed_b(store, setup).await;
    }

    async fn scenario_no_cache(store: Arc<dyn ArtifactStore>) {
        let log = Log::default();
        let setup = Setup::default();
        cache_run_plain(&store, &log, &setup).await;
        let secrets = MemorySecretProvider::default();
        let (s, executed) = cache_run(&store, &log, &setup, true, secrets).await;
        assert_eq!(executed, ["a", "b", "c", "x"]);
        assert!(cached_nodes(&s).is_empty());
        // The bypassing run still leaves the cache usable.
        let (_, executed) = cache_run_plain(&store, &log, &setup).await;
        assert!(executed.is_empty());
    }

    async fn scenario_resume(store: Arc<dyn ArtifactStore>) {
        let log = Log::default();
        let broken = Setup {
            b_with: serde_json::json!({ "name": "b", "fail": true }),
            ..Setup::default()
        };
        let (s, executed) = cache_run_plain(&store, &log, &broken).await;
        assert_eq!(executed, ["a", "b", "x"]);
        assert_eq!(status(&s, "b"), Status::Failed);
        assert_eq!(status(&s, "c"), Status::Cancelled);
        let (s, executed) = cache_run_plain(&store, &log, &Setup::default()).await;
        assert_eq!(executed, ["b", "c"]);
        assert!(s.is_success());
        assert_eq!(cached_nodes(&s), ["a", "x"]);
    }

    /// Delegates to `inner` but reports `hidden` as not existing.
    struct HidingStore {
        inner: Arc<dyn ArtifactStore>,
        hidden: Mutex<Option<ArtifactId>>,
        fail_get_meta: AtomicBool,
    }

    impl ArtifactStore for HidingStore {
        fn put<'a>(
            &'a self,
            manifest: Manifest,
            payload: PayloadSource<'a>,
        ) -> BoxFuture<'a, Result<ArtifactId, Error>> {
            self.inner.put(manifest, payload)
        }

        fn get<'a>(
            &'a self,
            id: &'a ArtifactId,
        ) -> BoxFuture<'a, Result<Option<crate::Artifact>, Error>> {
            self.inner.get(id)
        }

        fn fetch<'a>(
            &'a self,
            id: &'a ArtifactId,
            dest: &'a Path,
        ) -> BoxFuture<'a, Result<PathBuf, Error>> {
            self.inner.fetch(id, dest)
        }

        fn verify<'a>(&'a self, id: &'a ArtifactId) -> BoxFuture<'a, Result<(), Error>> {
            self.inner.verify(id)
        }

        fn exists<'a>(&'a self, id: &'a ArtifactId) -> BoxFuture<'a, Result<bool, Error>> {
            Box::pin(async move {
                if *self.hidden.lock().unwrap() == Some(*id) {
                    return Ok(false);
                }
                self.inner.exists(id).await
            })
        }

        fn put_meta<'a>(
            &'a self,
            key: &'a RelPath,
            data: &'a [u8],
        ) -> BoxFuture<'a, Result<(), Error>> {
            self.inner.put_meta(key, data)
        }

        fn get_meta<'a>(
            &'a self,
            key: &'a RelPath,
        ) -> BoxFuture<'a, Result<Option<Vec<u8>>, Error>> {
            if self.fail_get_meta.load(Ordering::SeqCst) {
                return Box::pin(async { Err(Error::Runtime("get_meta failed".into())) });
            }
            self.inner.get_meta(key)
        }

        fn list_meta<'a>(
            &'a self,
            prefix: &'a RelPath,
        ) -> BoxFuture<'a, Result<Vec<RelPath>, Error>> {
            self.inner.list_meta(prefix)
        }

        fn delete_meta<'a>(&'a self, key: &'a RelPath) -> BoxFuture<'a, Result<(), Error>> {
            self.inner.delete_meta(key)
        }
    }

    async fn scenario_missing_output(store: Arc<dyn ArtifactStore>) {
        let hiding = Arc::new(HidingStore {
            inner: store,
            hidden: Mutex::new(None),
            fail_get_meta: AtomicBool::new(false),
        });
        let store: Arc<dyn ArtifactStore> = hiding.clone();
        let log = Log::default();
        let setup = Setup::default();
        let (s, _) = cache_run_plain(&store, &log, &setup).await;
        *hiding.hidden.lock().unwrap() = Some(s.nodes["a"].outputs["out"]);
        let (s, executed) = cache_run_plain(&store, &log, &setup).await;
        assert_eq!(executed, ["a"]);
        assert!(!s.nodes["a"].cached);
        // Same bytes, same id: the descendants still hit.
        assert_eq!(cached_nodes(&s), ["b", "c", "x"]);
    }

    async fn scenario_lookup_error(store: Arc<dyn ArtifactStore>) {
        let hiding = Arc::new(HidingStore {
            inner: store,
            hidden: Mutex::new(None),
            fail_get_meta: AtomicBool::new(false),
        });
        let store: Arc<dyn ArtifactStore> = hiding.clone();
        let log = Log::default();
        let setup = Setup::default();
        cache_run_plain(&store, &log, &setup).await;
        hiding.fail_get_meta.store(true, Ordering::SeqCst);
        let (s, executed) = cache_run_plain(&store, &log, &setup).await;
        assert!(s.is_success());
        assert_eq!(executed, ["a", "b", "c", "x"]);
        assert!(cached_nodes(&s).is_empty());
        // The entries were rewritten: once reads work again everything hits.
        hiding.fail_get_meta.store(false, Ordering::SeqCst);
        let (s, executed) = cache_run_plain(&store, &log, &setup).await;
        assert!(executed.is_empty());
        assert_eq!(cached_nodes(&s), ["a", "b", "c", "x"]);
    }
}
