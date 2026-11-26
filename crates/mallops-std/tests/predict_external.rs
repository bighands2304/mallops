//! Integration tests of the external Python predictor (`harness = false`, see `Cargo.toml`).
//!
//! With `--worker <mode>` this binary is the mock worker, speaking NDJSON protocol v1 (no Python
//! needed); otherwise it runs every `external_predictor_protocol_*` test matching the optional
//! positional filter and exits non-zero if one fails. The tests needing a real Python
//! interpreter (`worker.py`) skip when it is missing.

use std::future::Future;
use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::LazyLock;
use std::time::{Duration, Instant};

use mallops_core::{Artifact, ArtifactId, ArtifactKind, Error, Manifest};
use mallops_std::predictor::{
    ExternalPythonPredictor, LoadedPredictor, PredictInput, Predictor, conformance,
};
use serde_json::{Value, json};

const PYTHON: &str = if cfg!(windows) { "python" } else { "python3" };

macro_rules! tests {
    ($($name:ident),* $(,)?) => { &[$((stringify!($name), $name as fn())),*] };
}

const TESTS: &[(&str, fn())] = tests![
    external_predictor_protocol_executes_code,
    external_predictor_protocol_ready_handshake,
    external_predictor_protocol_load_rejects_bad_artifacts,
    external_predictor_protocol_conformance,
    external_predictor_protocol_predict_and_batch,
    external_predictor_protocol_worker_env,
    external_predictor_protocol_bytes_rejected,
    external_predictor_protocol_worker_error_keeps_worker,
    external_predictor_protocol_timeout_restarts,
    external_predictor_protocol_restart_after_crash,
    external_predictor_protocol_invalid_reply_restarts,
    external_predictor_protocol_dropped_request_restarts,
    external_predictor_protocol_child_killed_on_drop,
    external_predictor_protocol_worker_py_end_to_end,
];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(i) = args.iter().position(|a| a == "--worker") {
        return mock_worker(&args[i + 1..]);
    }
    let filter = args.iter().find(|a| !a.starts_with('-'));
    let (mut passed, mut failed) = (0, Vec::new());
    for (name, test) in TESTS {
        if filter.is_some_and(|f| !name.contains(f.as_str())) {
            continue;
        }
        let ok = std::panic::catch_unwind(*test).is_ok();
        println!("test {name} ... {}", if ok { "ok" } else { "FAILED" });
        if ok {
            passed += 1;
        } else {
            failed.push(*name);
        }
    }
    let filtered = TESTS.len() - passed - failed.len();
    println!(
        "\ntest result: {}. {passed} passed; {} failed; {filtered} filtered out",
        if failed.is_empty() { "ok" } else { "FAILED" },
        failed.len()
    );
    if !failed.is_empty() {
        println!("failures: {failed:?}");
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------------------------
// Mock worker (child side)
// ---------------------------------------------------------------------------------------------

/// `args`: the mode (`ok`, `silent`: never ready, `exit`: dies before ready,
/// `deaf`: ready then ignores stdin, `badready`: valid JSON but the wrong `ready`), then optionally
/// `--heartbeat <file>` (rewritten every 20 ms while alive).
///
/// In `ok` mode the input picks the behavior: `"sleep"`, `"crash"`, `"fail"` (ok:false),
/// `"pid"`, `"env"`, `"garbage"` (non-JSON line), `"wrong-id"`; anything else is echoed.
fn mock_worker(args: &[String]) {
    if let [_, flag, path, ..] = args
        && flag == "--heartbeat"
    {
        let path = path.clone();
        std::thread::spawn(move || {
            for n in 0u64..=u64::MAX {
                let _ = std::fs::write(&path, n.to_string());
                std::thread::sleep(Duration::from_millis(20));
            }
        });
    }
    match args[0].as_str() {
        "ok" => {}
        "silent" => return std::thread::sleep(Duration::from_secs(60)),
        // Ready, then deaf: ignores stdin (no exit on EOF) and keeps running.
        "deaf" => {
            println!(r#"{{"v":1,"type":"ready"}}"#);
            return std::thread::sleep(Duration::from_secs(60));
        }
        // A valid JSON line that is not the v1 `ready`.
        "badready" => {
            println!(r#"{{"v":2,"type":"ready"}}"#);
            return std::thread::sleep(Duration::from_secs(60));
        }
        "exit" => {
            eprintln!("no model here");
            std::process::exit(2);
        }
        other => panic!("unknown mock mode {other}"),
    }
    println!(r#"{{"v":1,"type":"ready"}}"#);
    for line in std::io::stdin().lock().lines() {
        let req: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let id = req["id"].as_u64().unwrap();
        let reply = if req["type"] == "predict" {
            match req["input"].as_str() {
                Some("garbage") => {
                    println!("this is not json");
                    continue;
                }
                Some("wrong-id") => json!({"id": id + 1000, "ok": true, "output": 0}),
                _ => match answer(&req["input"]) {
                    Ok(out) => json!({"id": id, "ok": true, "output": out}),
                    Err(e) => json!({"id": id, "ok": false, "error": e}),
                },
            }
        } else {
            let outs: Result<Vec<_>, _> = req["inputs"]
                .as_array()
                .unwrap()
                .iter()
                .map(answer)
                .collect();
            match outs {
                Ok(outs) => json!({"id": id, "ok": true, "outputs": outs}),
                Err(e) => json!({"id": id, "ok": false, "error": e}),
            }
        };
        let mut out = std::io::stdout().lock();
        writeln!(out, "{reply}").unwrap();
        out.flush().unwrap();
    }
}

fn answer(input: &Value) -> Result<Value, String> {
    match input.as_str() {
        Some("sleep") => {
            std::thread::sleep(Duration::from_secs(60));
            Ok(Value::Null)
        }
        Some("crash") => {
            eprintln!("boom: crashing on purpose");
            std::process::exit(3);
        }
        Some("fail") => Err("bad input".into()),
        Some("pid") => Ok(json!(std::process::id())),
        Some("env") => Ok(json!({
            "vars": std::env::vars().map(|(k, _)| k).collect::<Vec<_>>(),
            "model": std::env::var("MALLOPS_MODEL_FILE").ok(),
            "features": std::env::var("MALLOPS_FEATURES").ok(),
            "cwd": std::env::current_dir().unwrap(),
        })),
        _ => Ok(json!({ "echo": input })),
    }
}

// ---------------------------------------------------------------------------------------------
// Helpers (host side)
// ---------------------------------------------------------------------------------------------

fn block_on<F: Future>(f: F) -> F::Output {
    static RT: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    });
    RT.block_on(f)
}

/// The predictor running this binary as a mock worker in `mode`, with a 10 s timeout.
fn mock(mode: &str, extra: &[&str]) -> ExternalPythonPredictor {
    let exe = std::env::current_exe()
        .unwrap()
        .to_str()
        .unwrap()
        .to_owned();
    let mut argv = vec![exe, "--worker".into(), mode.into()];
    argv.extend(extra.iter().map(|s| (*s).to_owned()));
    ExternalPythonPredictor::new()
        .with_command(argv)
        .with_timeout(Duration::from_secs(10))
}

fn model(kind: ArtifactKind, flavor: &str) -> Artifact {
    let schema = json!({
        "input": [{"name": "a", "type": "f64"}, {"name": "b", "type": "f64"}],
        "output": [{"name": "y", "type": "i64"}],
    });
    Artifact::new(
        ArtifactId::from_bytes([7; 32]),
        Manifest::new(kind, "pickle")
            .with_flavor(flavor)
            .with_io_schema(schema),
    )
}

/// A payload dir holding a placeholder `model.pkl` (the mock never reads it).
fn payload() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("model.pkl"), "not a pickle").unwrap();
    dir
}

async fn load(p: &ExternalPythonPredictor, dir: &Path) -> Box<dyn LoadedPredictor> {
    p.load(&model(ArtifactKind::Model, "sklearn-pickle"), dir)
        .await
        .unwrap()
}

fn j(v: Value) -> PredictInput {
    PredictInput::Json(v)
}

async fn pid(loaded: &dyn LoadedPredictor) -> Value {
    match loaded.predict(j(json!("pid"))).await.unwrap() {
        PredictInput::Json(v) => v,
        other => panic!("{other:?}"),
    }
}

fn runtime_msg<T: std::fmt::Debug>(r: Result<T, Error>) -> String {
    match r {
        Err(Error::Runtime(m)) => m,
        other => panic!("expected Error::Runtime, got {other:?}"),
    }
}

/// Waits up to 10 s for `path` to hold content other than `old`.
fn changed(path: &Path, old: Option<&str>) -> String {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Ok(s) = std::fs::read_to_string(path)
            && !s.is_empty()
            && Some(s.as_str()) != old
        {
            return s;
        }
        assert!(Instant::now() < deadline, "heartbeat did not change");
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn python_available() -> bool {
    let ok = std::process::Command::new(PYTHON)
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success());
    if !ok {
        eprintln!("skipped: `{PYTHON}` not found in PATH");
    }
    ok
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

fn external_predictor_protocol_executes_code() {
    let p = ExternalPythonPredictor::new();
    assert!(p.executes_code());
    assert_eq!(p.flavors(), ["sklearn-pickle"]);
}

fn external_predictor_protocol_ready_handshake() {
    block_on(async {
        let dir = payload();
        let loaded = load(&mock("ok", &[]), dir.path()).await;
        assert_eq!(
            loaded.predict(j(json!(1))).await.unwrap(),
            j(json!({"echo": 1}))
        );

        let start = Instant::now();
        let p = mock("silent", &[]).with_timeout(Duration::from_millis(300));
        let r = p
            .load(&model(ArtifactKind::Model, "sklearn-pickle"), dir.path())
            .await;
        assert_eq!(
            runtime_msg(r.map(drop)),
            "worker handshake timed out after 300ms"
        );
        assert!(start.elapsed() < Duration::from_secs(5));

        let r = mock("badready", &[])
            .load(&model(ArtifactKind::Model, "sklearn-pickle"), dir.path())
            .await;
        let m = runtime_msg(r.map(drop));
        assert!(m.contains("ready"), "{m}");

        let r = mock("exit", &[])
            .load(&model(ArtifactKind::Model, "sklearn-pickle"), dir.path())
            .await;
        let m = runtime_msg(r.map(drop));
        assert!(
            m.contains("exited unexpectedly") && m.contains("no model here"),
            "{m}"
        );
    });
}

fn external_predictor_protocol_load_rejects_bad_artifacts() {
    block_on(async {
        let dir = payload();
        // Rejected before spawning anything: the command does not even exist.
        let p = ExternalPythonPredictor::new().with_command(vec!["/nonexistent/worker".into()]);
        for artifact in [
            model(ArtifactKind::Dataset, "sklearn-pickle"),
            model(ArtifactKind::Model, "onnx"),
        ] {
            let r = p.load(&artifact, dir.path()).await;
            assert!(matches!(r, Err(Error::Config(_))), "{:?}", r.map(drop));
        }
        let empty = tempfile::tempdir().unwrap();
        let r = p
            .load(&model(ArtifactKind::Model, "sklearn-pickle"), empty.path())
            .await;
        assert!(runtime_msg(r.map(drop)).contains("expected model.pkl"));
        let r = p
            .load(&model(ArtifactKind::Model, "sklearn-pickle"), dir.path())
            .await;
        assert!(runtime_msg(r.map(drop)).contains("failed to spawn"));
    });
}

fn external_predictor_protocol_conformance() {
    block_on(async {
        let dir = payload();
        let cases = [
            (j(json!(1)), j(json!({"echo": 1}))),
            (
                j(json!({"a": 1, "b": 2})),
                j(json!({"echo": {"a": 1, "b": 2}})),
            ),
        ];
        let artifact = model(ArtifactKind::Model, "sklearn-pickle");
        let malformed = PredictInput::Bytes {
            content_type: "image/png".into(),
            data: vec![1, 2, 3],
        };
        let p = mock("ok", &[]);
        let r = conformance::check(&p, &artifact, dir.path(), &cases, malformed);
        assert_eq!(r.await, Ok(()));
    });
}

fn external_predictor_protocol_predict_and_batch() {
    block_on(async {
        let dir = payload();
        let loaded = load(&mock("ok", &[]), dir.path()).await;
        assert_eq!(
            loaded.predict(j(json!("x"))).await.unwrap(),
            j(json!({"echo": "x"}))
        );
        let outs = loaded.predict_batch(vec![j(json!(1)), j(json!(2)), j(json!(3))]);
        assert_eq!(
            outs.await.unwrap(),
            vec![
                j(json!({"echo": 1})),
                j(json!({"echo": 2})),
                j(json!({"echo": 3}))
            ]
        );
        assert_eq!(loaded.predict_batch(Vec::new()).await.unwrap(), Vec::new());
        // One failing input fails the whole batch; the worker stays.
        let before = pid(&*loaded).await;
        let r = loaded
            .predict_batch(vec![j(json!(1)), j(json!("fail"))])
            .await;
        assert_eq!(runtime_msg(r), "worker error: bad input");
        assert_eq!(pid(&*loaded).await, before);
    });
}

fn external_predictor_protocol_worker_env() {
    block_on(async {
        let dir = payload();
        let loaded = load(&mock("ok", &[]), dir.path()).await;
        let PredictInput::Json(env) = loaded.predict(j(json!("env"))).await.unwrap() else {
            panic!("expected JSON")
        };
        // No host variable leaks except the allowlisted ones (the OS may add its own, such as
        // macOS `__CF_USER_TEXT_ENCODING`, so the child's list is not checked exhaustively).
        let allowed = ["PATH", "SYSTEMROOT"];
        let vars = env["vars"].as_array().unwrap();
        assert!(
            std::env::var("CARGO_MANIFEST_DIR").is_ok(),
            "run through cargo"
        );
        for (host, _) in std::env::vars() {
            if !allowed.iter().any(|a| a.eq_ignore_ascii_case(&host)) && !host.starts_with("__CF") {
                assert!(!vars.contains(&json!(host)), "{host} leaked into {env}");
            }
        }
        let pkl = dir.path().join("model.pkl");
        assert_eq!(env["model"], json!(pkl.to_str().unwrap()));
        assert_eq!(env["features"], json!(r#"["a","b"]"#));
        let cwd = Path::new(env["cwd"].as_str().unwrap());
        assert_eq!(
            cwd.canonicalize().unwrap(),
            dir.path().canonicalize().unwrap()
        );
    });
}

fn external_predictor_protocol_bytes_rejected() {
    block_on(async {
        let dir = payload();
        let loaded = load(&mock("ok", &[]), dir.path()).await;
        let before = pid(&*loaded).await;
        let bytes = || PredictInput::Bytes {
            content_type: "application/octet-stream".into(),
            data: vec![0],
        };
        assert!(runtime_msg(loaded.predict(bytes()).await).contains("takes JSON input"));
        let r = loaded.predict_batch(vec![j(json!(1)), bytes()]).await;
        assert!(runtime_msg(r).contains("takes JSON input"));
        assert_eq!(pid(&*loaded).await, before, "worker must not be touched");
    });
}

fn external_predictor_protocol_worker_error_keeps_worker() {
    block_on(async {
        let dir = payload();
        let loaded = load(&mock("ok", &[]), dir.path()).await;
        let before = pid(&*loaded).await;
        let r = loaded.predict(j(json!("fail"))).await;
        assert_eq!(runtime_msg(r), "worker error: bad input");
        assert_eq!(pid(&*loaded).await, before);
    });
}

fn external_predictor_protocol_timeout_restarts() {
    block_on(async {
        let dir = payload();
        let p = mock("ok", &[]).with_timeout(Duration::from_millis(500));
        let loaded = load(&p, dir.path()).await;
        let before = pid(&*loaded).await;
        let start = Instant::now();
        let r = loaded.predict(j(json!("sleep"))).await;
        assert_eq!(runtime_msg(r), "timed out after 500ms");
        assert!(start.elapsed() < Duration::from_secs(5));
        assert_ne!(
            pid(&*loaded).await,
            before,
            "a new worker must serve the next request"
        );
    });
}

fn external_predictor_protocol_restart_after_crash() {
    block_on(async {
        let dir = payload();
        let loaded = load(&mock("ok", &[]), dir.path()).await;
        let before = pid(&*loaded).await;
        let m = runtime_msg(loaded.predict(j(json!("crash"))).await);
        assert!(
            m.contains("exited unexpectedly") && m.contains("boom: crashing"),
            "{m}"
        );
        // Not retried; the next request gets a fresh worker.
        assert_eq!(
            loaded.predict(j(json!(5))).await.unwrap(),
            j(json!({"echo": 5}))
        );
        assert_ne!(pid(&*loaded).await, before);
    });
}

fn external_predictor_protocol_invalid_reply_restarts() {
    block_on(async {
        let dir = payload();
        let loaded = load(&mock("ok", &[]), dir.path()).await;
        for (input, msg) in [
            ("garbage", "invalid worker line"),
            ("wrong-id", "worker answered id"),
        ] {
            let before = pid(&*loaded).await;
            let m = runtime_msg(loaded.predict(j(json!(input))).await);
            assert!(m.contains(msg), "{input}: {m}");
            assert_ne!(pid(&*loaded).await, before, "{input}");
        }
    });
}

fn external_predictor_protocol_dropped_request_restarts() {
    block_on(async {
        let dir = payload();
        let beat = dir.path().join("heartbeat");
        let beat_arg = beat.to_str().unwrap();
        let loaded = load(&mock("ok", &["--heartbeat", beat_arg]), dir.path()).await;
        let before = pid(&*loaded).await;
        changed(&beat, None); // the first beat is out
        let dropped = tokio::time::timeout(
            Duration::from_millis(200),
            loaded.predict(j(json!("sleep"))),
        );
        assert!(
            dropped.await.is_err(),
            "the sleeping request must not finish"
        );
        // The worker that had the request in flight is dead (its heartbeat stops)...
        std::thread::sleep(Duration::from_millis(300));
        let settled = changed(&beat, Some("")); // any non-empty read (a rewrite may be mid-way)
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(changed(&beat, Some("")), settled, "worker still running");
        // ...and no stale reply can reach the next request.
        assert_eq!(
            loaded.predict(j(json!(9))).await.unwrap(),
            j(json!({"echo": 9}))
        );
        assert_ne!(pid(&*loaded).await, before);
    });
}

fn external_predictor_protocol_child_killed_on_drop() {
    block_on(async {
        let dir = payload();
        let beat = dir.path().join("heartbeat");
        let loaded = load(
            &mock("deaf", &["--heartbeat", beat.to_str().unwrap()]),
            dir.path(),
        )
        .await;
        let first = changed(&beat, None);
        changed(&beat, Some(&first)); // alive: the heartbeat advances
        drop(loaded);
        std::thread::sleep(Duration::from_millis(300));
        let settled = changed(&beat, Some(""));
        std::thread::sleep(Duration::from_millis(500));
        assert_eq!(changed(&beat, Some("")), settled, "worker still running");
    });
}

/// The real `worker.py` on a pickled builtins-only model (`predict` and `predict_proba` are
/// `list`, so a row comes back as the prediction and its second value as the probability), with
/// a payload `json.py` that must not shadow the stdlib module.
fn external_predictor_protocol_worker_py_end_to_end() {
    if !python_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let out = std::process::Command::new(PYTHON)
        .args([
            "-c",
            "import pickle, types; pickle.dump(types.SimpleNamespace(predict=list, \
             predict_proba=list, classes_=[0, 1]), open('model.pkl', 'wb'))",
        ])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::write(
        dir.path().join("json.py"),
        "raise ImportError('payload json.py shadowed the stdlib')\n",
    )
    .unwrap();
    block_on(async {
        let cases = [
            (
                j(json!({"b": 1, "a": 2})),
                j(json!({"prediction": [2, 1], "probability": 1})),
            ),
            (
                j(json!([3, 4])),
                j(json!({"prediction": [3, 4], "probability": 4})),
            ),
        ];
        let artifact = model(ArtifactKind::Model, "sklearn-pickle");
        let p = ExternalPythonPredictor::new();
        // A missing feature fails that request only.
        let malformed = j(json!({"a": 1}));
        let r = conformance::check(&p, &artifact, dir.path(), &cases, malformed);
        assert_eq!(r.await, Ok(()));
    });
}
