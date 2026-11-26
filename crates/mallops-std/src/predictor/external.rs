//! Predictor for the `sklearn-pickle` flavor, via an external Python worker.
//!
//! The worker is a child process speaking NDJSON protocol v1 on stdin/stdout (documented in the
//! reference worker `worker.py`, embedded in this crate). It gets no socket and a cleared
//! environment; one request runs at a time.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, PoisonError};
use std::time::Duration;

use mallops_core::{Artifact, ArtifactKind, BoxFuture, Error};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

use super::{LoadedPredictor, PredictInput, Predictor};

/// The reference worker, run with `python3 -c`.
const WORKER: &str = include_str!("worker.py");
/// Default per-request (and handshake) timeout.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
/// Bytes of worker stderr kept for error messages (the most recent ones).
const STDERR_TAIL: usize = 4 * 1024;
/// Longest accepted protocol line; a longer one is a protocol violation.
const MAX_LINE: u64 = 64 << 20;
/// How long to wait for the stderr reader after the worker dies, to report its last words.
const STDERR_DRAIN: Duration = Duration::from_secs(1);

/// Serves `sklearn-pickle` models through a Python worker process (executes code: needs
/// explicit trust).
///
/// `load` starts the worker in the payload directory with only `PATH` (+ `SYSTEMROOT` on
/// Windows), `MALLOPS_MODEL_FILE` and `MALLOPS_FEATURES` in its environment, and waits for its
/// `ready` line. A request that times out, gets an invalid reply, finds the worker dead, or is
/// dropped mid-flight kills the worker; the next request starts a new one (the failed request is
/// not retried). Dropping the loaded predictor kills the worker too: only the direct child is
/// killed (no process group, on Windows as on Unix), so a worker must not leave subprocesses.
#[derive(Debug, Clone)]
pub struct ExternalPythonPredictor {
    command: Vec<String>,
    timeout: Duration,
}

impl ExternalPythonPredictor {
    /// Runs the reference worker with `python3` (`python` on Windows) from `PATH`, 30 s timeout.
    #[must_use]
    pub fn new() -> Self {
        let python = if cfg!(windows) { "python" } else { "python3" };
        Self {
            command: vec![python.into(), "-c".into(), WORKER.into()],
            timeout: DEFAULT_TIMEOUT,
        }
    }

    /// Replaces the worker argv (program first, run directly, no shell).
    #[must_use]
    pub fn with_command(mut self, command: Vec<String>) -> Self {
        self.command = command;
        self
    }

    /// Sets the timeout of the `ready` handshake and of each request.
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

impl Default for ExternalPythonPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl Predictor for ExternalPythonPredictor {
    fn flavors(&self) -> &[&str] {
        &["sklearn-pickle"]
    }

    fn executes_code(&self) -> bool {
        true
    }

    fn load<'a>(
        &'a self,
        artifact: &'a Artifact,
        payload_dir: &'a Path,
    ) -> BoxFuture<'a, Result<Box<dyn LoadedPredictor>, Error>> {
        Box::pin(async move {
            let manifest = artifact.manifest();
            if manifest.kind != ArtifactKind::Model {
                return Err(Error::Config(format!(
                    "expected a model artifact, got `{}`",
                    manifest.kind
                )));
            }
            if !manifest
                .flavor
                .as_deref()
                .is_some_and(|f| self.flavors().contains(&f))
            {
                return Err(Error::Config(format!(
                    "unsupported flavor {:?}",
                    manifest.flavor
                )));
            }
            // No `canonicalize`: its `\\?\` prefix on Windows confuses Python.
            let dir = std::path::absolute(payload_dir)
                .map_err(|e| Error::Runtime(format!("{}: {e}", payload_dir.display())))?;
            let model = model_file(&dir)?;
            let features = feature_names(manifest.io_schema.as_ref())?;
            let spec = WorkerSpec {
                argv: self.command.clone(),
                env: worker_env(&model, &features, &|name| std::env::var(name).ok())?,
                cwd: dir,
                timeout: self.timeout,
            };
            let worker = Worker::spawn(&spec).await?;
            Ok(Box::new(Loaded {
                spec,
                worker: Mutex::new(Some(worker)),
                next_id: AtomicU64::new(0),
            }) as Box<dyn LoadedPredictor>)
        })
    }
}

/// `payload_dir/model.pkl` if it exists, else the only regular file in `payload_dir`.
fn model_file(dir: &Path) -> Result<PathBuf, Error> {
    let pkl = dir.join("model.pkl");
    if pkl.is_file() {
        return Ok(pkl);
    }
    let io = |e: std::io::Error| Error::Runtime(format!("{}: {e}", dir.display()));
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(io)? {
        let entry = entry.map_err(io)?;
        if entry.file_type().map_err(io)?.is_file() {
            files.push(entry.path());
        }
    }
    match <[PathBuf; 1]>::try_from(files) {
        Ok([file]) => Ok(file),
        Err(files) => Err(Error::Runtime(format!(
            "{}: expected model.pkl or exactly one file, found {} files",
            dir.display(),
            files.len()
        ))),
    }
}

/// Names in `io_schema.input` (`[{"name", "type"}, …]`); empty without one.
fn feature_names(io_schema: Option<&Value>) -> Result<Vec<String>, Error> {
    let Some(input) = io_schema.and_then(|s| s.get("input")) else {
        return Ok(Vec::new());
    };
    let bad = || Error::Runtime("io_schema.input must be a list of {\"name\", \"type\"}".into());
    input
        .as_array()
        .ok_or_else(bad)?
        .iter()
        .map(|f| f.get("name").and_then(Value::as_str).map(str::to_owned))
        .collect::<Option<_>>()
        .ok_or_else(bad)
}

/// The worker's whole environment: `PATH` (+ `SYSTEMROOT` on Windows) from `lookup`,
/// `MALLOPS_MODEL_FILE` and, when there are features, `MALLOPS_FEATURES` (JSON array).
fn worker_env(
    model: &Path,
    features: &[String],
    lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<Vec<(String, String)>, Error> {
    let fixed: &[&str] = if cfg!(windows) {
        &["PATH", "SYSTEMROOT"]
    } else {
        &["PATH"]
    };
    let mut env: Vec<(String, String)> = fixed
        .iter()
        .filter_map(|name| lookup(name).map(|v| ((*name).to_owned(), v)))
        .collect();
    let model = model
        .to_str()
        .ok_or_else(|| Error::Runtime(format!("non UTF-8 model path {}", model.display())))?;
    env.push(("MALLOPS_MODEL_FILE".into(), model.to_owned()));
    if !features.is_empty() {
        env.push(("MALLOPS_FEATURES".into(), json!(features).to_string()));
    }
    Ok(env)
}

/// How to (re)start the worker of one loaded model.
struct WorkerSpec {
    argv: Vec<String>,
    cwd: PathBuf,
    env: Vec<(String, String)>,
    timeout: Duration,
}

/// A running worker that sent `ready`. Dropping it kills the process.
struct Worker {
    _child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    stderr_tail: Arc<std::sync::Mutex<Vec<u8>>>,
    stderr_task: JoinHandle<()>,
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.stderr_task.abort();
    }
}

impl Worker {
    /// Spawns the worker and waits (bounded by `spec.timeout`) for its `ready` line.
    async fn spawn(spec: &WorkerSpec) -> Result<Self, Error> {
        let Some((program, args)) = spec.argv.split_first() else {
            return Err(Error::Runtime("worker command must not be empty".into()));
        };
        let mut child = Command::new(program)
            .args(args)
            .current_dir(&spec.cwd)
            .env_clear()
            .envs(spec.env.iter().map(|(k, v)| (k, v)))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| Error::Runtime(format!("failed to spawn `{program}`: {e}")))?;
        let (Some(stdin), Some(stdout), Some(stderr)) =
            (child.stdin.take(), child.stdout.take(), child.stderr.take())
        else {
            return Err(Error::Runtime("worker pipes unavailable".into()));
        };
        let stderr_tail = Arc::default();
        let stderr_task = tokio::spawn(keep_tail(stderr, Arc::clone(&stderr_tail)));
        let mut worker = Self {
            _child: child,
            stdin,
            stdout: BufReader::new(stdout),
            stderr_tail,
            stderr_task,
        };
        let ready = tokio::time::timeout(spec.timeout, worker.read_line())
            .await
            .map_err(|_| {
                Error::Runtime(format!(
                    "worker handshake timed out after {:?}",
                    spec.timeout
                ))
            })??;
        if ready != json!({"v": 1, "type": "ready"}) {
            return Err(Error::Runtime(format!(
                "worker sent {ready} instead of {{\"v\":1,\"type\":\"ready\"}}"
            )));
        }
        Ok(worker)
    }

    /// Writes `request` and reads the reply, which must carry `id`.
    async fn exchange(&mut self, request: &[u8], id: u64) -> Result<Value, Error> {
        if self.stdin.write_all(request).await.is_err() || self.stdin.flush().await.is_err() {
            return Err(self.died().await);
        }
        let reply = self.read_line().await?;
        match reply.get("id") {
            Some(got) if *got == json!(id) => Ok(reply),
            got => Err(Error::Runtime(format!(
                "worker answered id {}, expected {id}",
                got.unwrap_or(&Value::Null)
            ))),
        }
    }

    /// Reads one protocol line.
    async fn read_line(&mut self) -> Result<Value, Error> {
        let mut line = Vec::new();
        let read = (&mut self.stdout)
            .take(MAX_LINE)
            .read_until(b'\n', &mut line)
            .await;
        if read.is_err() || !line.ends_with(b"\n") {
            if line.len() as u64 == MAX_LINE {
                return Err(Error::Runtime(format!(
                    "worker line longer than {MAX_LINE} bytes"
                )));
            }
            return Err(self.died().await);
        }
        serde_json::from_slice(&line)
            .map_err(|e| Error::Runtime(format!("invalid worker line: {e}")))
    }

    /// The error for a worker that closed its pipes, with the tail of its stderr. Call once.
    async fn died(&mut self) -> Error {
        let _ = tokio::time::timeout(STDERR_DRAIN, &mut self.stderr_task).await;
        let tail = self
            .stderr_tail
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        Error::Runtime(format!(
            "worker exited unexpectedly; stderr: {}",
            String::from_utf8_lossy(&tail).trim_end()
        ))
    }
}

/// Reads `stderr` to EOF keeping only its last [`STDERR_TAIL`] bytes in `tail`.
async fn keep_tail(mut stderr: ChildStderr, tail: Arc<std::sync::Mutex<Vec<u8>>>) {
    let mut chunk = [0; 1024];
    while let Ok(n @ 1..) = stderr.read(&mut chunk).await {
        let mut tail = tail.lock().unwrap_or_else(PoisonError::into_inner);
        tail.extend_from_slice(&chunk[..n]);
        let excess = tail.len().saturating_sub(STDERR_TAIL);
        tail.drain(..excess);
    }
}

/// A model served by a worker; restarted on the next request after a failure.
struct Loaded {
    spec: WorkerSpec,
    // ponytail: one worker, requests serialized; a pool of workers if throughput matters.
    worker: Mutex<Option<Worker>>,
    next_id: AtomicU64,
}

impl Loaded {
    /// Sends `predict` (or `predict_batch` with an array `payload`) and returns the reply's
    /// `output` (`outputs`, checked to have one item per input).
    async fn call(&self, batch: bool, payload: Value) -> Result<Value, Error> {
        let (kind, field, reply_field) = if batch {
            ("predict_batch", "inputs", "outputs")
        } else {
            ("predict", "input", "output")
        };
        let expected = payload.as_array().map(Vec::len).filter(|_| batch);
        let mut slot = self.worker.lock().await;
        // Taken before writing: only a complete, valid reply puts it back. On any other outcome,
        // this future being dropped included, the worker is dropped, which kills it.
        let mut worker = match slot.take() {
            Some(worker) => worker,
            None => Worker::spawn(&self.spec).await?,
        };
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut request = json!({"id": id, "type": kind});
        request[field] = payload;
        let mut request = serde_json::to_vec(&request)
            .map_err(|e| Error::Runtime(format!("cannot encode request: {e}")))?;
        request.push(b'\n');
        let mut reply = tokio::time::timeout(self.spec.timeout, worker.exchange(&request, id))
            .await
            .map_err(|_| Error::Runtime(format!("timed out after {:?}", self.spec.timeout)))??;
        let invalid = || Error::Runtime(format!("invalid worker reply to `{kind}`"));
        let result = match reply.get("ok") {
            Some(Value::Bool(true)) => {
                let out = reply
                    .get_mut(reply_field)
                    .map(Value::take)
                    .ok_or_else(invalid)?;
                if expected.is_some() && out.as_array().map(Vec::len) != expected {
                    return Err(invalid());
                }
                Ok(out)
            }
            Some(Value::Bool(false)) => Err(Error::Runtime(format!(
                "worker error: {}",
                reply.get("error").and_then(Value::as_str).unwrap_or("")
            ))),
            _ => return Err(invalid()),
        };
        *slot = Some(worker);
        result
    }
}

/// The JSON body of `input`; the worker takes nothing else.
fn json_input(input: PredictInput) -> Result<Value, Error> {
    match input {
        PredictInput::Json(v) => Ok(v),
        PredictInput::Bytes { content_type, .. } => Err(Error::Runtime(format!(
            "sklearn-pickle takes JSON input, got `{content_type}` bytes"
        ))),
    }
}

impl LoadedPredictor for Loaded {
    fn predict(&self, input: PredictInput) -> BoxFuture<'_, Result<PredictInput, Error>> {
        Box::pin(async move {
            let out = self.call(false, json_input(input)?).await?;
            Ok(PredictInput::Json(out))
        })
    }

    fn predict_batch(
        &self,
        inputs: Vec<PredictInput>,
    ) -> BoxFuture<'_, Result<Vec<PredictInput>, Error>> {
        Box::pin(async move {
            if inputs.is_empty() {
                return Ok(Vec::new());
            }
            let inputs = inputs
                .into_iter()
                .map(json_input)
                .collect::<Result<Vec<_>, _>>()?;
            let Value::Array(outs) = self.call(true, Value::Array(inputs)).await? else {
                return Err(Error::Runtime(
                    "invalid worker reply to `predict_batch`".into(),
                ));
            };
            Ok(outs.into_iter().map(PredictInput::Json).collect())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_predictor_model_file_prefers_model_pkl_then_single_file() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        // Empty: nothing to load.
        assert!(matches!(model_file(d), Err(Error::Runtime(_))));
        std::fs::create_dir(d.join("sub")).unwrap(); // directories never count
        std::fs::write(d.join("m.joblib"), "x").unwrap();
        assert_eq!(model_file(d).unwrap(), d.join("m.joblib"));
        std::fs::write(d.join("other"), "x").unwrap();
        let Err(Error::Runtime(m)) = model_file(d) else {
            panic!("two files must be ambiguous")
        };
        assert!(m.contains("found 2 files"), "{m}");
        std::fs::write(d.join("model.pkl"), "x").unwrap();
        assert_eq!(model_file(d).unwrap(), d.join("model.pkl"));
    }

    #[test]
    fn external_predictor_feature_names_from_io_schema() {
        assert!(feature_names(None).unwrap().is_empty());
        assert!(
            feature_names(Some(&json!({"output": []})))
                .unwrap()
                .is_empty()
        );
        let schema = json!({"input": [{"name": "a", "type": "f64"}, {"name": "b", "type": "i64"}]});
        assert_eq!(feature_names(Some(&schema)).unwrap(), ["a", "b"]);
        for bad in [json!({"input": "a"}), json!({"input": [{"type": "f64"}]})] {
            assert!(matches!(feature_names(Some(&bad)), Err(Error::Runtime(_))));
        }
    }

    #[test]
    fn external_predictor_worker_env_is_allowlisted() {
        let lookup = |name: &str| (name != "SYSTEMROOT").then(|| format!("<{name}>"));
        let model = Path::new("/payload/model.pkl");
        let env = worker_env(model, &[], &lookup).unwrap();
        assert_eq!(
            env,
            [
                ("PATH".to_owned(), "<PATH>".to_owned()),
                (
                    "MALLOPS_MODEL_FILE".to_owned(),
                    "/payload/model.pkl".to_owned()
                ),
            ]
        );
        let env = worker_env(model, &["a".into(), "b \"q\"".into()], &|_| None).unwrap();
        assert_eq!(
            env,
            [
                (
                    "MALLOPS_MODEL_FILE".to_owned(),
                    "/payload/model.pkl".to_owned()
                ),
                (
                    "MALLOPS_FEATURES".to_owned(),
                    r#"["a","b \"q\""]"#.to_owned()
                ),
            ]
        );
    }

    #[test]
    fn external_predictor_default_command_runs_reference_worker() {
        let p = ExternalPythonPredictor::new();
        let python = if cfg!(windows) { "python" } else { "python3" };
        assert_eq!(p.command, [python, "-c", WORKER]);
        assert_eq!(p.timeout, DEFAULT_TIMEOUT);
        assert!(WORKER.contains(r#"{"v": 1, "type": "ready"}"#));
    }
}
