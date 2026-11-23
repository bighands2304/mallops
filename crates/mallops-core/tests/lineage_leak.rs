//! Anti-leak integration test for lineage, tracing logs, workdir and `LocalStore` (SPEC §3.14, NFR-4).
//!
//! Runs in a dedicated test process with a single test so installing a global `tracing`
//! subscriber cannot race with parallel unit tests on tracing's global callsite interest cache.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use mallops_core::{
    ArtifactKind, ArtifactStore, Block, BlockOutcome, BlockRegistry, BlockSpec, BoxFuture, Context,
    Digest, DraftPayload, Error, Inputs, JsonSink, LineageEvent, LineageSink, LocalStore, Manifest,
    MemorySecretProvider, OutputDraft, OutputDrafts, PipelineBuilder, Port, RelPath, Runtime,
    RuntimeConfig, Stage,
};

const SECRET: &str = "s3cr3t-T0KEN-4f9a1c";
const LIMIT: Duration = Duration::from_secs(10);

fn fixed_clock() -> SystemTime {
    UNIX_EPOCH + Duration::from_millis(1_700_000_000_123)
}

/// Minimal `tracing` subscriber: one line per span or event with its level, name and fields.
struct Capture {
    out: Arc<Mutex<String>>,
    next_id: AtomicU64,
}

impl Capture {
    fn install_global() -> Arc<Mutex<String>> {
        let out = Arc::new(Mutex::new(String::new()));
        let capture = Capture {
            out: Arc::clone(&out),
            next_id: AtomicU64::new(1),
        };
        tracing::subscriber::set_global_default(capture)
            .expect("install global tracing subscriber");
        out
    }

    fn line(&self, meta: &tracing::Metadata<'_>, record: impl FnOnce(&mut Fields<'_>)) {
        let mut text = format!("{} {} ", meta.level(), meta.name());
        record(&mut Fields(&mut text));
        let mut out = self.out.lock().unwrap();
        out.push_str(&text);
        out.push('\n');
    }
}

struct Fields<'a>(&'a mut String);

impl tracing::field::Visit for Fields<'_> {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.0.push_str(&format!("{}={value} ", field.name()));
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0.push_str(&format!("{}={value:?} ", field.name()));
    }
}

impl tracing::Subscriber for Capture {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        self.line(span.metadata(), |f| span.record(f));
        tracing::span::Id::from_u64(self.next_id.fetch_add(1, Ordering::SeqCst))
    }

    fn record(&self, _: &tracing::span::Id, values: &tracing::span::Record<'_>) {
        let mut text = String::new();
        values.record(&mut Fields(&mut text));
        let mut out = self.out.lock().unwrap();
        out.push_str(&text);
        out.push('\n');
    }

    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}

    fn event(&self, event: &tracing::Event<'_>) {
        self.line(event.metadata(), |f| event.record(f));
    }

    fn enter(&self, _: &tracing::span::Id) {}

    fn exit(&self, _: &tracing::span::Id) {}
}

#[derive(Default)]
struct RecordingSink(Mutex<Vec<LineageEvent>>);

impl RecordingSink {
    fn take(&self) -> Vec<LineageEvent> {
        std::mem::take(&mut *self.0.lock().unwrap())
    }
}

impl LineageSink for RecordingSink {
    fn record<'a>(&'a self, event: &'a LineageEvent) -> BoxFuture<'a, Result<(), Error>> {
        self.0.lock().unwrap().push(event.clone());
        Box::pin(async { Ok(()) })
    }
}

fn blob(bytes: Vec<u8>) -> OutputDraft {
    OutputDraft::new(
        Manifest::new(ArtifactKind::Blob, "text"),
        DraftPayload::Bytes(bytes),
    )
}

/// Reads `TOKEN` through `ctx.secret()`, logs it (redacted), writes a file payload into
/// `ctx.workdir()`, inspects `ctx.workdir()` for leaks, and outputs that file.
struct SecretBlock {
    spec: BlockSpec,
    used: AtomicBool,
    workdir_scanned: AtomicUsize,
}

impl Block for SecretBlock {
    fn spec(&self) -> &BlockSpec {
        &self.spec
    }

    fn run<'a>(
        &'a self,
        ctx: &'a Context,
        _: Inputs,
    ) -> BoxFuture<'a, Result<BlockOutcome, Error>> {
        Box::pin(async move {
            let token = ctx.secret("TOKEN")?;
            tracing::info!(token = %token, debug = ?token, "authenticating");
            if token.expose() != SECRET {
                return Err(Error::Runtime("wrong token".into()));
            }
            self.used.store(true, Ordering::SeqCst);
            let proof = Digest::hash(token.expose().as_bytes());
            let out_file = ctx.workdir().join("auth.txt");
            std::fs::write(&out_file, format!("authorized {proof}")).unwrap();

            let mut scanned = 0;
            let hits = files_containing(ctx.workdir(), SECRET.as_bytes(), &mut scanned);
            assert!(hits.is_empty(), "secret leaked into workdir: {hits:?}");
            assert!(scanned > 0, "no file scanned in node workdir");
            self.workdir_scanned.store(scanned, Ordering::SeqCst);

            let draft = OutputDraft::new(
                Manifest::new(ArtifactKind::Blob, "text"),
                DraftPayload::File(out_file),
            );
            Ok(BlockOutcome::Produced(
                OutputDrafts::new().with("out", draft),
            ))
        })
    }
}

/// Consumes one blob input and produces a downstream blob output.
struct DownstreamBlock(BlockSpec);

impl Block for DownstreamBlock {
    fn spec(&self) -> &BlockSpec {
        &self.0
    }

    fn run<'a>(
        &'a self,
        _: &'a Context,
        inputs: Inputs,
    ) -> BoxFuture<'a, Result<BlockOutcome, Error>> {
        Box::pin(async move {
            let count = inputs.iter().count();
            Ok(BlockOutcome::Produced(OutputDrafts::new().with(
                "out",
                blob(format!("downstream:{count}").into_bytes()),
            )))
        })
    }
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

/// Files under `root` containing `needle`; counts the scanned files in `scanned`.
fn files_containing(root: &Path, needle: &[u8], scanned: &mut usize) -> Vec<PathBuf> {
    let mut hits = Vec::new();
    for entry in std::fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        let meta = std::fs::symlink_metadata(&path).unwrap();
        if meta.is_dir() {
            hits.extend(files_containing(&path, needle, scanned));
        } else if meta.is_file() {
            *scanned += 1;
            if contains(&std::fs::read(&path).unwrap(), needle) {
                hits.push(path);
            }
        }
    }
    hits
}

#[tokio::test]
async fn lineage_anti_leak_secret_absent_from_lineage_logs_and_files() {
    let logs = Capture::install_global();

    let mut reg = BlockRegistry::new();
    let auth_spec = BlockSpec::new("t/auth", "1", Stage::Custom)
        .with_output(Port::new("out", ArtifactKind::Blob));
    let auth = Arc::new(SecretBlock {
        spec: auth_spec,
        used: AtomicBool::new(false),
        workdir_scanned: AtomicUsize::new(0),
    });
    reg.register(auth.clone()).unwrap();

    let after_spec = BlockSpec::new("t/after", "1", Stage::Custom)
        .with_input(Port::new("a", ArtifactKind::Blob))
        .with_output(Port::new("out", ArtifactKind::Blob));
    reg.register(Arc::new(DownstreamBlock(after_spec))).unwrap();

    let mut builder = PipelineBuilder::new("anti-leak");
    builder
        .block("auth", "t/auth".parse().unwrap())
        .with(serde_json::json!({ "token": "${secret:TOKEN}" }));
    builder
        .block("after", "t/after".parse().unwrap())
        .input("a", "auth.out");
    let pipeline = builder.build(&reg).unwrap();

    let store_root = tempfile::tempdir().unwrap();
    let work_root = tempfile::tempdir().unwrap();
    let store: Arc<dyn ArtifactStore> = Arc::new(LocalStore::new(store_root.path()).unwrap());
    let sink = Arc::new(RecordingSink::default());
    let json_sink: Arc<dyn LineageSink> = Arc::new(JsonSink::new(Arc::clone(&store)));

    let mut cfg = RuntimeConfig::new(Arc::clone(&store));
    cfg.max_jobs = 2;
    cfg.lineage = vec![json_sink, sink.clone()];
    cfg.clock = fixed_clock;
    cfg.git_commit = Some("0123abc".into());
    cfg.workdir_root = work_root.path().to_path_buf();
    cfg.secret_provider = Arc::new(MemorySecretProvider::default().with_secret("TOKEN", SECRET));

    let summary = tokio::time::timeout(LIMIT, Runtime::new(cfg).run(&pipeline))
        .await
        .expect("run timed out")
        .unwrap();

    assert!(summary.is_success(), "{:?}", summary.nodes);
    assert!(
        auth.used.load(Ordering::SeqCst),
        "the block never used the secret"
    );
    assert!(
        auth.workdir_scanned.load(Ordering::SeqCst) > 0,
        "node workdir was not inspected during the run"
    );

    // Lineage: `runs/<run-id>/lineage.json` (written by `JsonSink`) and the recorded events.
    let lineage_path = RelPath::parse(&format!("runs/{}/lineage.json", summary.run_id)).unwrap();
    let lineage_bytes = store
        .get_meta(&lineage_path)
        .await
        .unwrap()
        .expect("lineage.json not written");
    assert!(contains(&lineage_bytes, b"auth"));
    assert!(!contains(&lineage_bytes, SECRET.as_bytes()));

    let events = format!("{:?}", sink.take());
    assert!(events.contains("auth"), "{events}");
    assert!(!events.contains(SECRET), "{events}");

    // Logs: captured the node span, the block's event, and redacted `***` fields, never the secret.
    let captured = logs.lock().unwrap().clone();
    assert!(captured.contains("id=auth"), "{captured}");
    assert!(captured.contains("authenticating"), "{captured}");
    assert!(captured.contains("token=***"), "{captured}");
    assert!(captured.contains("debug=***"), "{captured}");
    assert!(!captured.contains(SECRET), "{captured}");

    // Files: run workdir root was cleaned up, and the whole `LocalStore` root is secret-free.
    let mut work_scanned = 0;
    let hits = files_containing(work_root.path(), SECRET.as_bytes(), &mut work_scanned);
    assert!(hits.is_empty(), "{hits:?}");
    assert_eq!(work_scanned, 0, "run workdir not cleaned up");

    let mut store_scanned = 0;
    let hits = files_containing(store_root.path(), SECRET.as_bytes(), &mut store_scanned);
    assert!(hits.is_empty(), "{hits:?}");
    assert!(store_scanned > 0, "no file scanned in store");
}
