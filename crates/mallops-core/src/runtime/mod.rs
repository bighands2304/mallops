//! Async DAG runtime: executes a [`ValidatedPipeline`] on `tokio` (SPEC §1, §4; ARCHITECTURE §4).
//!
//! The runtime schedules ready nodes concurrently (bounded by [`RuntimeConfig::max_jobs`]),
//! commits block output drafts to the store, propagates failure, skip and cancellation,
//! and reports per-node results in a [`RunSummary`].

mod commit;

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tracing::Instrument;

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
        let env = NodeEnv {
            run_id,
            store: Arc::clone(&self.config.store),
            secrets: Arc::clone(&self.config.secret_provider),
            cancel: self.config.cancel.clone(),
            jobs: Arc::new(Semaphore::new(self.config.max_jobs.max(1))),
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
                        let workdir = run_dir.path().join(node_dir(i, &node.id));
                        let span = tracing::info_span!(
                            "node",
                            id = %node.id,
                            block = %node.block.spec().id
                        );
                        let task = run_node(node.clone(), inputs, workdir, env.clone());
                        let handle = tasks.spawn(task.instrument(span));
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
            if let Some(&i) = task_node.get(&task) {
                results[i] = Some(summary);
            }
        }

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
}

/// `<node>` of a validated `<node>.<port>` reference.
fn upstream_node(src: &str) -> &str {
    src.split_once('.').map_or(src, |(node, _)| node)
}

/// Workdir name of node `i`: index prefix (unique) plus the id with unsafe characters replaced.
fn node_dir(i: usize, id: &str) -> String {
    let safe: String = id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("{i:04}-{safe}")
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
    let start = Instant::now();
    let result = execute(&node, inputs, workdir, &env).await;
    let duration = start.elapsed();
    match result {
        Ok((outputs, None)) => finished(Status::Succeeded, duration, outputs, None),
        Ok((outputs, Some(reason))) => finished(Status::Skipped, duration, outputs, Some(reason)),
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

/// Committed outputs and, for a skipped block, its reason.
type Executed = (BTreeMap<String, ArtifactId>, Option<String>);

async fn execute(
    node: &ValidatedNode,
    inputs: Vec<(String, ArtifactId)>,
    workdir: PathBuf,
    env: &NodeEnv,
) -> Result<Executed, Error> {
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
    Ok((outputs, reason))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tokio::sync::{Barrier, Notify};

    use super::*;
    use crate::{
        ArtifactKind, Block, BlockRegistry, BlockSpec, BoxFuture, DraftPayload, Manifest,
        MemoryStore, OutputDraft, OutputDrafts, PipelineBuilder, Port, Stage,
    };

    const LIMIT: Duration = Duration::from_secs(10);

    /// Shared observation points of the test blocks.
    #[derive(Default)]
    struct Probe {
        barrier: Option<Barrier>,
        running: AtomicUsize,
        peak: AtomicUsize,
        started: Notify,
    }

    /// Test block whose behavior is picked by `with: {"mode": ...}`:
    /// `ok` (default), `fail`, `skip`, `barrier`, `wait_cancel`.
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
                let now = p.running.fetch_add(1, Ordering::SeqCst) + 1;
                p.peak.fetch_max(now, Ordering::SeqCst);
                for _ in 0..10 {
                    tokio::task::yield_now().await;
                }
                let mode = ctx.config()["mode"].as_str().unwrap_or("ok");
                let result = match mode {
                    "fail" => Err(Error::Runtime("boom".into())),
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
}
