//! Acceptance tests of `std/external` through the public API (issues #13 and #52).
//!
//! The fake trainer is this test binary, re-executed on the ignored [`child_entry`] test: it
//! reads its mode from `params.mode` in `config.json` (its cwd). No shell, no Python, so the
//! tests run on Linux, macOS and Windows.

#![cfg(feature = "external")]

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use mallops_core::*;
use serde_json::{Value, json};

const LIMIT: Duration = Duration::from_secs(60);
const SECRET: &str = "s3cr3t-value-42";

// ---------------------------------------------------------------------------------------------
// Fake trainer (child side)
// ---------------------------------------------------------------------------------------------

/// argv re-executing this binary on [`child_entry`], followed by `extra`.
fn trainer(extra: &[&str]) -> Vec<String> {
    let exe = std::env::current_exe().unwrap();
    let mut argv = vec![exe.to_str().unwrap().to_owned()];
    argv.extend(
        [
            "--exact",
            "child_entry",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ]
        .iter()
        .chain(extra)
        .map(|s| (*s).to_owned()),
    );
    argv
}

/// Number of harness arguments (exe included) before the `extra` of [`trainer`].
const HARNESS_ARGS: usize = 6;

/// Entry point of the fake trainer; a no-op unless run by `std/external`.
#[test]
#[ignore]
fn child_entry() {
    let Ok(raw) = std::fs::read_to_string("config.json") else {
        return;
    };
    let config: Value = serde_json::from_str(&raw).unwrap();
    let params = &config["params"];
    let write_json = |path: &str, v: &Value| std::fs::write(path, v.to_string()).unwrap();
    let produced = |outputs: Value| {
        write_json(
            "outputs.json",
            &json!({ "protocol_version": 1, "status": "produced", "outputs": outputs }),
        )
    };
    let model = || {
        std::fs::write("model.bin", "weights").unwrap();
        json!({ "path": "model.bin", "kind": "model", "format": "bin" })
    };
    let echo = |v: Value| {
        write_json("echo.json", &v);
        json!({ "path": "echo.json", "kind": "metrics", "format": "json" })
    };
    match params["mode"].as_str().unwrap() {
        "contract" => {
            let inputs: Value =
                serde_json::from_str(&std::fs::read_to_string("inputs.json").unwrap()).unwrap();
            // Every materialized input file must refuse writes.
            let mut write_failed = Vec::new();
            for entry in inputs["inputs"].as_object().unwrap().values() {
                for file in std::fs::read_dir(entry["path"].as_str().unwrap()).unwrap() {
                    let path = file.unwrap().path();
                    let res = std::fs::OpenOptions::new().append(true).open(&path);
                    write_failed.push(res.is_err());
                }
            }
            let mut model = model();
            model["io_schema"] = json!({ "inputs": [{ "name": "x", "dtype": "f32" }] });
            produced(json!({
                "model": model,
                "echo": echo(json!({ "inputs": inputs, "config": config, "write_failed": write_failed })),
            }));
        }
        // Writes `params.outputs` verbatim as `outputs.json`.
        "raw" => write_json("outputs.json", &params["outputs"]),
        "echo" => {
            let args: Vec<String> = std::env::args().skip(HARNESS_ARGS).collect();
            let env: BTreeMap<String, String> = std::env::vars().collect();
            produced(json!({
                "model": model(),
                "echo": echo(json!({ "args": args, "env": env, "config": config })),
            }));
        }
        // Reads the `dataset` input, writes a `features` and a `metrics` output.
        "analyze" => {
            let inputs: Value =
                serde_json::from_str(&std::fs::read_to_string("inputs.json").unwrap()).unwrap();
            let entry = &inputs["inputs"]["dataset"];
            assert_eq!(entry["manifest"]["kind"], "dataset");
            let dir = entry["path"].as_str().unwrap();
            let f = std::fs::read_dir(dir).unwrap().next().unwrap().unwrap();
            let rows = std::fs::read_to_string(f.path()).unwrap();
            std::fs::write("feat.csv", rows).unwrap();
            std::fs::write("m.json", "{}").unwrap();
            produced(json!({
                "features": { "path": "feat.csv", "kind": "features", "format": "csv" },
                "metrics": { "path": "m.json", "kind": "metrics", "format": "json" },
            }));
        }
        "leak" => {
            eprint!("token={}", std::env::var("TRAIN_TOKEN").unwrap_or_default());
            std::process::exit(3);
        }
        "bad_path" => {
            model();
            #[cfg(unix)]
            std::os::unix::fs::symlink("model.bin", "link").unwrap();
            let path = params["path"].clone();
            produced(json!({ "model": { "path": path, "kind": "model", "format": "bin" } }));
        }
        "sleep" => {
            std::fs::write(params["ready"].as_str().unwrap(), "ready").unwrap();
            std::thread::sleep(Duration::from_secs(60));
        }
        "noisy" => {
            eprint!("HEAD{}TAIL", "x".repeat(10_000));
            std::process::exit(1);
        }
        "count" => {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(params["counter"].as_str().unwrap())
                .unwrap();
            writeln!(f, "run").unwrap();
            produced(json!({ "model": model() }));
        }
        other => panic!("unknown mode {other}"),
    }
}

// ---------------------------------------------------------------------------------------------
// Harness (test side)
// ---------------------------------------------------------------------------------------------

/// Produces an artifact of the kind declared by its `out` port, fed to the external block.
struct Source(BlockSpec);

impl Block for Source {
    fn spec(&self) -> &BlockSpec {
        &self.0
    }

    fn run<'a>(
        &'a self,
        _ctx: &'a Context,
        _inputs: Inputs,
    ) -> BoxFuture<'a, Result<BlockOutcome, Error>> {
        Box::pin(async {
            let kind = self.0.outputs[0].kind;
            let draft = OutputDraft::new(
                Manifest::new(kind, "csv"),
                DraftPayload::Bytes(b"x,y\n1,2\n".to_vec()),
            );
            Ok(BlockOutcome::Produced(
                OutputDrafts::new().with("out", draft),
            ))
        })
    }
}

/// `source` (artifact of `kind`) → `train` (`std/external`, input port `port`). The default
/// ports are `inputs: {train: features}` and `outputs: {model: model}`; `with` overrides them.
fn pipeline_from(
    source: &'static str,
    kind: ArtifactKind,
    port: &str,
    with: Value,
) -> ValidatedPipeline {
    let mut registry = BlockRegistry::new();
    mallops_std::register_std(&mut registry).unwrap();
    let spec = BlockSpec::new(source, "1", Stage::Preprocess).with_output(Port::new("out", kind));
    registry.register(Arc::new(Source(spec))).unwrap();
    let mut cfg = json!({ "inputs": { "train": "features" }, "outputs": { "model": "model" } });
    cfg.as_object_mut()
        .unwrap()
        .extend(with.as_object().unwrap().clone());
    let mut b = PipelineBuilder::new("external");
    b.block("features", source.parse().unwrap());
    b.block("train", "std/external".parse().unwrap())
        .input(port, "features.out")
        .with(cfg);
    b.build(&registry).unwrap()
}

fn pipeline(with: Value) -> ValidatedPipeline {
    pipeline_from("test/features", ArtifactKind::Features, "train", with)
}

/// `with:` running the fake trainer in `mode`, plus the `extra` top-level fields.
fn with_mode(mode: &str, extra: Value) -> Value {
    let mut with = json!({ "command": trainer(&[]), "params": { "mode": mode } });
    let obj = with.as_object_mut().unwrap();
    for (k, v) in extra.as_object().unwrap() {
        if k == "params" {
            let params = obj["params"].as_object_mut().unwrap();
            params.extend(v.as_object().unwrap().clone());
        } else {
            obj.insert(k.clone(), v.clone());
        }
    }
    with
}

/// Store and workdir root shared by the runs of one test.
struct Env {
    store: Arc<MemoryStore>,
    root: tempfile::TempDir,
}

impl Env {
    fn new() -> Self {
        Self {
            store: Arc::new(MemoryStore::new()),
            root: tempfile::tempdir().unwrap(),
        }
    }

    fn config(&self) -> RuntimeConfig {
        let mut cfg = RuntimeConfig::new(Arc::clone(&self.store) as Arc<dyn ArtifactStore>);
        cfg.workdir_root = self.root.path().to_path_buf();
        cfg.secret_provider =
            Arc::new(MemorySecretProvider::default().with_secret("TRAIN_TOKEN", SECRET));
        cfg
    }

    async fn run(&self, with: Value) -> RunSummary {
        run_with(self.config(), with).await
    }

    /// Payload of a committed single-file artifact.
    async fn payload(&self, id: &ArtifactId) -> String {
        let dir = tempfile::tempdir().unwrap();
        let path = self.store.fetch(id, &dir.path().join("p")).await.unwrap();
        if path.is_file() {
            return std::fs::read_to_string(path).unwrap();
        }
        std::fs::read_dir(path)
            .unwrap()
            .map(|e| std::fs::read_to_string(e.unwrap().path()).unwrap())
            .collect()
    }

    async fn echo(&self, summary: &RunSummary) -> Value {
        let node = train(summary);
        assert_eq!(node.status, Status::Succeeded, "{node:?}");
        serde_json::from_str(&self.payload(&node.outputs["echo"]).await).unwrap()
    }
}

async fn run_with(cfg: RuntimeConfig, with: Value) -> RunSummary {
    run_pipeline(cfg, pipeline(with)).await
}

async fn run_pipeline(cfg: RuntimeConfig, pipeline: ValidatedPipeline) -> RunSummary {
    tokio::time::timeout(LIMIT, Runtime::new(cfg).run(&pipeline))
        .await
        .expect("run timed out")
        .unwrap()
}

fn train(summary: &RunSummary) -> &NodeSummary {
    &summary.nodes["train"]
}

/// Message of the `Error::Runtime` of a failed `train` node.
fn runtime_error(summary: &RunSummary) -> String {
    let node = train(summary);
    assert_eq!(node.status, Status::Failed, "{node:?}");
    match &node.error {
        Some(Error::Runtime(m)) => m.clone(),
        other => panic!("expected Error::Runtime, got {other:?}"),
    }
}

/// `outputs:` adding the `echo` port written by the trainer.
fn echo_outputs() -> Value {
    json!({ "model": "model", "echo": "metrics" })
}

async fn wait_for_file(path: &Path) {
    tokio::time::timeout(LIMIT, async {
        while !path.exists() {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("the trainer never started");
}

// ---------------------------------------------------------------------------------------------
// external_contract_v1
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn external_contract_v1_files_inputs_and_manifest() {
    let env = Env::new();
    let summary = env
        .run(with_mode(
            "contract",
            json!({ "flavor": "onnx", "params": { "lr": 0.1 }, "outputs": echo_outputs() }),
        ))
        .await;
    let echo = env.echo(&summary).await;
    let features = summary.nodes["features"].outputs["out"];

    // config.json: version, flavor, params, run id.
    let config = &echo["config"];
    assert_eq!(config["protocol_version"], 1);
    assert_eq!(config["flavor"], "onnx");
    assert_eq!(config["params"], json!({ "mode": "contract", "lr": 0.1 }));
    assert_eq!(config["run_id"], summary.run_id.to_string());

    // inputs.json: version and the materialized `train` input.
    let inputs = &echo["inputs"];
    assert_eq!(inputs["protocol_version"], 1);
    let entry = &inputs["inputs"]["train"];
    assert_eq!(entry["path"], "inputs/train");
    assert_eq!(entry["id"], features.to_hex());
    assert_eq!(entry["manifest"]["kind"], "features");
    assert_eq!(inputs["inputs"].as_object().unwrap().len(), 1);

    // Inputs are read-only for the child.
    let write_failed = echo["write_failed"].as_array().unwrap();
    assert!(!write_failed.is_empty());
    assert!(write_failed.iter().all(|f| f == true), "{write_failed:?}");

    // Configured flavor and io_schema land in the committed manifest.
    let model = env
        .store
        .get(&train(&summary).outputs["model"])
        .await
        .unwrap()
        .expect("model committed");
    let m = model.manifest();
    assert_eq!(m.kind, ArtifactKind::Model);
    assert_eq!(m.format, "bin");
    assert_eq!(m.flavor.as_deref(), Some("onnx"));
    assert_eq!(
        m.io_schema,
        Some(json!({ "inputs": [{ "name": "x", "dtype": "f32" }] }))
    );
    assert_eq!(m.parents, [features]);

    // The run workdir is removed at the end of the run.
    let left: Vec<_> = std::fs::read_dir(env.root.path()).unwrap().collect();
    assert!(left.is_empty(), "run workdir not removed: {left:?}");
}

#[tokio::test]
async fn external_contract_v1_skipped_status() {
    let env = Env::new();
    let outputs =
        json!({ "protocol_version": 1, "status": "skipped", "skip_reason": "nothing new" });
    let summary = env
        .run(with_mode(
            "raw",
            json!({ "params": { "outputs": outputs } }),
        ))
        .await;
    let node = train(&summary);
    assert_eq!(node.status, Status::Skipped, "{node:?}");
    assert_eq!(node.reason.as_deref(), Some("nothing new"));
    assert!(node.outputs.is_empty());
}

#[tokio::test]
async fn external_contract_v1_rejects_wrong_outputs_version() {
    let env = Env::new();
    let outputs = json!({ "protocol_version": 2, "status": "produced" });
    let summary = env
        .run(with_mode(
            "raw",
            json!({ "params": { "outputs": outputs } }),
        ))
        .await;
    let msg = runtime_error(&summary);
    assert!(msg.contains("protocol_version must be 1, got 2"), "{msg}");
}

// ---------------------------------------------------------------------------------------------
// external_security
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn external_security_argv_is_literal() {
    let env = Env::new();
    let meta = ["$HOME;echo x", "|", "`id`"];
    let mut with = with_mode("echo", json!({ "outputs": echo_outputs() }));
    with["command"] = json!(trainer(&meta));
    let echo = env.echo(&env.run(with).await).await;
    assert_eq!(echo["args"], json!(meta));
}

#[tokio::test]
async fn external_security_env_and_secrets() {
    // Both are set by cargo in this process; only the listed one reaches the child.
    let pkg = std::env::var("CARGO_PKG_NAME").expect("CARGO_PKG_NAME set by cargo");
    assert!(std::env::var("CARGO_MANIFEST_DIR").is_ok());

    let env = Env::new();
    let with = with_mode(
        "echo",
        json!({ "env": ["CARGO_PKG_NAME"], "secrets": ["TRAIN_TOKEN"], "outputs": echo_outputs() }),
    );
    let echo = env.echo(&env.run(with).await).await;
    let child_env = &echo["env"];
    assert_eq!(child_env["CARGO_PKG_NAME"], pkg.as_str());
    assert!(child_env.get("CARGO_MANIFEST_DIR").is_none(), "{child_env}");
    assert_eq!(child_env["TRAIN_TOKEN"], SECRET);
    // config.json never carries the secret.
    assert!(!echo["config"].to_string().contains(SECRET));
    assert!(echo["config"].get("secrets").is_none());
}

#[tokio::test]
async fn external_security_secret_redacted_from_logs() {
    let env = Env::new();
    let summary = env
        .run(with_mode("leak", json!({ "secrets": ["TRAIN_TOKEN"] })))
        .await;
    let msg = runtime_error(&summary);
    assert!(msg.contains("token=***"), "{msg}");
    assert!(!msg.contains(SECRET), "{msg}");
}

#[tokio::test]
async fn external_security_rejects_escaping_output_paths() {
    let absolute = std::env::temp_dir().join("model.bin");
    let mut paths = vec![
        "..".to_owned(),
        "../model.bin".to_owned(),
        absolute.to_str().unwrap().to_owned(),
    ];
    if cfg!(unix) {
        paths.push("link".to_owned());
    }
    let env = Env::new();
    for path in paths {
        let summary = env
            .run(with_mode("bad_path", json!({ "params": { "path": path } })))
            .await;
        let msg = runtime_error(&summary);
        assert!(msg.contains("output `model`"), "{path}: {msg}");
        assert!(train(&summary).outputs.is_empty());
    }
}

// ---------------------------------------------------------------------------------------------
// external_timeout_and_cancel
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn external_block_timeout_and_cancel_timeout_fails_node() {
    let env = Env::new();
    let ready = env.root.path().join("ready");
    let started = Instant::now();
    let summary = env
        .run(with_mode(
            "sleep",
            json!({ "timeout_secs": 1, "params": { "ready": ready } }),
        ))
        .await;
    let msg = runtime_error(&summary);
    assert!(msg.contains("timed out"), "{msg}");
    assert!(
        started.elapsed() < Duration::from_secs(20),
        "{:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn external_block_timeout_and_cancel_cancel_marks_node_cancelled() {
    let env = Env::new();
    let dir = tempfile::tempdir().unwrap();
    let ready = dir.path().join("ready");
    let source = CancelSource::new();
    let mut cfg = env.config();
    cfg.cancel = source.token();
    let run = tokio::spawn(run_with(
        cfg,
        with_mode("sleep", json!({ "params": { "ready": ready } })),
    ));
    wait_for_file(&ready).await;
    let started = Instant::now();
    source.cancel();
    let summary = run.await.unwrap();
    let node = train(&summary);
    assert_eq!(node.status, Status::Cancelled, "{node:?}");
    assert!(
        started.elapsed() < Duration::from_secs(20),
        "{:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn external_block_timeout_and_cancel_stderr_capped_in_error() {
    let env = Env::new();
    let summary = env
        .run(with_mode("noisy", json!({ "log_cap_bytes": 64 })))
        .await;
    let msg = runtime_error(&summary);
    let (_, tail) = msg.split_once("stderr tail:\n").expect(&msg);
    assert!(tail.len() <= 64, "{} bytes: {tail}", tail.len());
    assert!(tail.contains("HEAD"), "{tail}");
    assert!(!tail.contains("TAIL"), "{tail}");
}

// ---------------------------------------------------------------------------------------------
// external_ports_and_cache
// ---------------------------------------------------------------------------------------------

#[test]
fn external_ports_and_cache_flavor_on_model_port() {
    let pipeline = pipeline(with_mode("count", json!({ "flavor": "onnx" })));
    let node = pipeline.nodes().iter().find(|n| n.id == "train").unwrap();
    let model = &node.ports.outputs[0];
    assert_eq!(model.name, "model");
    assert_eq!(model.flavors, ["onnx"]);
    assert!(node.ports.inputs.iter().all(|p| p.flavors.is_empty()));
}

fn runs(counter: &Path) -> usize {
    std::fs::read_to_string(counter).map_or(0, |s| s.lines().count())
}

#[tokio::test]
async fn external_ports_and_cache_reruns_without_watch_files() {
    let env = Env::new();
    let dir = tempfile::tempdir().unwrap();
    let counter = dir.path().join("count");
    let with = with_mode("count", json!({ "params": { "counter": counter } }));
    for expected in 1..=2 {
        let summary = env.run(with.clone()).await;
        let node = train(&summary);
        assert_eq!(node.status, Status::Succeeded, "{node:?}");
        assert!(!node.cached);
        assert_eq!(runs(&counter), expected);
    }
}

#[tokio::test]
async fn external_ports_and_cache_watch_files_enable_cache() {
    let env = Env::new();
    let dir = tempfile::tempdir().unwrap();
    let counter = dir.path().join("count");
    let script = dir.path().join("train.py");
    std::fs::write(&script, "v1").unwrap();
    let with = with_mode(
        "count",
        json!({ "params": { "counter": counter }, "watch_files": [script] }),
    );
    for (cached, expected) in [(false, 1), (true, 1)] {
        let summary = env.run(with.clone()).await;
        let node = train(&summary);
        assert_eq!(node.status, Status::Succeeded, "{node:?}");
        assert_eq!(node.cached, cached);
        assert_eq!(runs(&counter), expected);
    }
    // A changed watched file invalidates the cache.
    std::fs::write(&script, "v2").unwrap();
    let summary = env.run(with).await;
    assert!(!train(&summary).cached);
    assert_eq!(runs(&counter), 2);
}

// ---------------------------------------------------------------------------------------------
// external_block: generic ports (issue #52)
// ---------------------------------------------------------------------------------------------

/// A non-training step: `dataset` in, `features` + `metrics` out, committed by the runtime.
#[tokio::test]
async fn external_block_custom_ports_dataset_to_features_and_metrics() {
    let env = Env::new();
    let with = with_mode(
        "analyze",
        json!({
            "inputs": { "dataset": "dataset" },
            "outputs": { "features": "features", "metrics": "metrics" },
        }),
    );
    let pipeline = pipeline_from("test/dataset", ArtifactKind::Dataset, "dataset", with);
    let summary = run_pipeline(env.config(), pipeline).await;
    let node = train(&summary);
    assert_eq!(node.status, Status::Succeeded, "{node:?}");
    assert_eq!(node.outputs.len(), 2);
    assert_eq!(env.payload(&node.outputs["features"]).await, "x,y\n1,2\n");
    assert_eq!(env.payload(&node.outputs["metrics"]).await, "{}");
}

/// A training-shaped config: `train: features` → `model: model` + `flavor`.
#[tokio::test]
async fn external_block_training_shaped_config_sets_flavor() {
    let env = Env::new();
    let summary = env
        .run(with_mode(
            "contract",
            json!({
                "flavor": "onnx",
                "inputs": { "train": "features" },
                "outputs": { "model": "model", "echo": "metrics" },
            }),
        ))
        .await;
    let node = train(&summary);
    assert_eq!(node.status, Status::Succeeded, "{node:?}");
    let model = env
        .store
        .get(&node.outputs["model"])
        .await
        .unwrap()
        .expect("model committed");
    assert_eq!(model.manifest().flavor.as_deref(), Some("onnx"));
}

#[test]
fn external_block_validate_config_rejects_bad_ports() {
    let mut registry = BlockRegistry::new();
    mallops_std::register_std(&mut registry).unwrap();
    let block = registry.get(&"std/external".parse().unwrap()).unwrap();
    let ok = json!({
        "command": ["x"], "inputs": { "d": "dataset" }, "outputs": { "m": "metrics" },
    });
    block.validate_config(&ok).unwrap();
    let without = |key: &str| {
        let mut c = ok.clone();
        c.as_object_mut().unwrap().remove(key);
        c
    };
    let mut flavor_no_model = ok.clone();
    flavor_no_model["flavor"] = json!("onnx");
    let mut empty_outputs = ok.clone();
    empty_outputs["outputs"] = json!({});
    let mut empty_inputs = ok.clone();
    empty_inputs["inputs"] = json!({});
    block.validate_config(&empty_inputs).unwrap();
    for bad in [
        without("inputs"),
        without("outputs"),
        flavor_no_model,
        empty_outputs,
    ] {
        assert!(
            matches!(block.validate_config(&bad), Err(Error::Config(_))),
            "{bad}"
        );
    }
}

#[tokio::test]
async fn external_block_nonzero_exit_fails_node() {
    let env = Env::new();
    let summary = env.run(with_mode("noisy", json!({}))).await;
    let msg = runtime_error(&summary);
    assert!(msg.contains("failed"), "{msg}");
}
