//! Child process execution for `std/train.external`: argv only, cleared environment, process
//! group, timeout/cancel, capped and redacted output. Knows nothing about `Context`.

use std::future::Future;
use std::path::PathBuf;
use std::pin::{Pin, pin};
use std::process::{ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::task::Poll;
use std::time::Duration;

use mallops_core::{CancelToken, Error, Secret};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;
use tokio::task::{AbortHandle, JoinHandle};

/// Maximum time to wait for stdout/stderr readers to drain after the child exits.
///
/// Guards against a surviving descendant holding an inherited pipe open.
const READER_DRAIN_TIMEOUT: Duration = Duration::from_secs(1);

/// Chunk size used while reading and draining child stdio pipes.
const READ_CHUNK_SIZE: usize = 8 * 1024;

/// Placeholder replacing redacted secret values in captured child output.
const REDACTED: &str = "***";

/// Everything needed to run the child; resolved by the block, plain data here.
#[derive(Clone)]
pub(crate) struct ChildSpec {
    /// Program and arguments, executed directly (never through a shell). Never empty.
    pub(crate) argv: Vec<String>,
    /// Working directory of the child (the node workdir).
    pub(crate) cwd: PathBuf,
    /// The child's complete environment (inherited env is cleared), secrets excluded.
    pub(crate) env: Vec<(String, String)>,
    /// Secrets added to the environment; their values are redacted from captured output.
    pub(crate) secret_env: Vec<(String, Secret)>,
    /// Kill the process group after this long.
    pub(crate) timeout: Option<Duration>,
    /// Maximum bytes kept per stream (stdout, stderr); the rest is drained and dropped.
    pub(crate) log_cap: usize,
}

/// Result of a child that ran to completion (any exit status).
#[derive(Debug)]
pub(crate) struct ChildOutput {
    pub(crate) status: ExitStatus,
    /// Captured stdout, secrets redacted, at most `log_cap` bytes.
    pub(crate) stdout: String,
    /// Captured stderr, secrets redacted, at most `log_cap` bytes.
    pub(crate) stderr: String,
    /// Whether stdout / stderr were truncated at `log_cap`.
    pub(crate) truncated: [bool; 2],
}

/// Runs the child and waits for it, killing its process group on timeout or cancellation.
///
/// A non-zero exit is NOT an error here: the caller maps `status` (D13-11).
///
/// # Errors
///
/// [`Error::Runtime`] if the child cannot be spawned or times out ("timed out after Ns"),
/// [`Error::Cancelled`] if `cancel` fires.
pub(crate) async fn run_child(spec: ChildSpec, cancel: &CancelToken) -> Result<ChildOutput, Error> {
    if cancel.is_cancelled() {
        return Err(Error::Cancelled);
    }

    let Some((program, args)) = spec.argv.split_first() else {
        return Err(Error::Runtime("child argv must not be empty".to_string()));
    };

    let mut cmd = Command::new(program);
    cmd.args(args)
        .current_dir(&spec.cwd)
        .env_clear()
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    for (key, value) in &spec.env {
        cmd.env(key, value);
    }
    for (key, secret) in &spec.secret_env {
        cmd.env(key, secret.expose());
    }

    #[cfg(unix)]
    cmd.process_group(0);

    let mut child = cmd
        .spawn()
        .map_err(|err| Error::Runtime(format!("failed to spawn `{program}`: {err}")))?;

    // Capture the process group id right after spawn while the child is still unreaped,
    // and guard against future-drop leaving descendants alive.
    #[cfg(unix)]
    let mut pgid_guard = ProcessGroupGuard::new(child.id());

    let max_secret_len = spec
        .secret_env
        .iter()
        .map(|(_, secret)| secret.expose().len())
        .max()
        .unwrap_or(0);
    let capture_cap = spec.log_cap.saturating_add(max_secret_len);

    let stdout_reader = spawn_stream_reader(child.stdout.take(), capture_cap);
    let stderr_reader = spawn_stream_reader(child.stderr.take(), capture_cap);

    enum WaitOutcome {
        Exited(std::io::Result<ExitStatus>),
        TimedOut(Duration),
        Cancelled,
    }

    let wait_outcome = {
        let mut cancel_fut = pin!(cancel.cancelled());
        let mut wait_fut = pin!(child.wait());
        let mut timeout_fut = pin!(async {
            match spec.timeout {
                Some(dur) => tokio::time::sleep(dur).await,
                None => std::future::pending::<()>().await,
            }
        });

        std::future::poll_fn(|cx| {
            if cancel_fut.as_mut().poll(cx).is_ready() {
                return Poll::Ready(WaitOutcome::Cancelled);
            }
            if let Poll::Ready(status) = wait_fut.as_mut().poll(cx) {
                return Poll::Ready(WaitOutcome::Exited(status));
            }
            if timeout_fut.as_mut().poll(cx).is_ready() {
                return Poll::Ready(WaitOutcome::TimedOut(spec.timeout.unwrap_or_default()));
            }
            Poll::Pending
        })
        .await
    };

    match wait_outcome {
        WaitOutcome::Cancelled => {
            #[cfg(unix)]
            if !pgid_guard.kill_now() {
                let _ = child.start_kill();
            }
            #[cfg(not(unix))]
            let _ = child.start_kill();

            let _ = child.wait().await;
            stdout_reader.abort_and_wait().await;
            stderr_reader.abort_and_wait().await;
            Err(Error::Cancelled)
        }
        WaitOutcome::TimedOut(dur) => {
            #[cfg(unix)]
            if !pgid_guard.kill_now() {
                let _ = child.start_kill();
            }
            #[cfg(not(unix))]
            let _ = child.start_kill();

            let _ = child.wait().await;
            stdout_reader.abort_and_wait().await;
            stderr_reader.abort_and_wait().await;
            Err(Error::Runtime(format_timeout(dur)))
        }
        WaitOutcome::Exited(status_res) => {
            // Reap any surviving grandchildren in the child's process group (ignoring ESRCH).
            #[cfg(unix)]
            let _ = pgid_guard.kill_now();

            let (stdout_raw, stderr_raw) = finish_readers(stdout_reader, stderr_reader).await;
            let status = status_res
                .map_err(|err| Error::Runtime(format!("failed waiting for `{program}`: {err}")))?;

            let mut secrets: Vec<&str> = spec
                .secret_env
                .iter()
                .map(|(_, secret)| secret.expose())
                .filter(|value| !value.is_empty())
                .collect();
            // Redact longer secrets first so a shorter secret that is a substring cannot
            // partially redact a longer secret and leave its remainder exposed.
            secrets.sort_unstable_by_key(|value| std::cmp::Reverse(value.len()));
            secrets.dedup();

            let (stdout, stdout_truncated) = redact_and_truncate(
                &stdout_raw.bytes,
                stdout_raw.overflowed,
                &secrets,
                spec.log_cap,
            );
            let (stderr, stderr_truncated) = redact_and_truncate(
                &stderr_raw.bytes,
                stderr_raw.overflowed,
                &secrets,
                spec.log_cap,
            );

            Ok(ChildOutput {
                status,
                stdout,
                stderr,
                truncated: [stdout_truncated, stderr_truncated],
            })
        }
    }
}

#[cfg(unix)]
struct ProcessGroupGuard {
    pgid: Option<nix::unistd::Pid>,
}

#[cfg(unix)]
impl ProcessGroupGuard {
    fn new(raw_pid: Option<u32>) -> Self {
        let pgid = raw_pid
            .and_then(|id| i32::try_from(id).ok())
            .filter(|&id| id > 0)
            .map(nix::unistd::Pid::from_raw);
        Self { pgid }
    }

    /// Sends `SIGKILL` to the process group and disarms the guard.
    ///
    /// Returns `true` if a valid process group id was signalled.
    fn kill_now(&mut self) -> bool {
        let Some(pgid) = self.pgid.take() else {
            return false;
        };
        kill_process_group(pgid);
        true
    }
}

#[cfg(unix)]
impl Drop for ProcessGroupGuard {
    fn drop(&mut self) {
        if let Some(pgid) = self.pgid.take() {
            kill_process_group(pgid);
        }
    }
}

#[cfg(unix)]
fn kill_process_group(pgid: nix::unistd::Pid) {
    match nix::sys::signal::killpg(pgid, nix::sys::signal::Signal::SIGKILL) {
        Ok(()) | Err(nix::errno::Errno::ESRCH) => {}
        Err(err) => {
            tracing::debug!(pgid = pgid.as_raw(), %err, "killpg returned unexpected error");
        }
    }
}

#[derive(Default)]
struct CapturedStream {
    bytes: Vec<u8>,
    overflowed: bool,
}

struct StreamReader {
    handle: Option<JoinHandle<()>>,
    abort: Option<AbortHandle>,
    state: Arc<Mutex<CapturedStream>>,
}

impl StreamReader {
    async fn abort_and_wait(mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
            let _ = handle.await;
        }
        self.abort = None;
    }

    fn take_captured(mut self) -> CapturedStream {
        self.abort = None;
        let mut guard = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::mem::take(&mut *guard)
    }
}

impl Drop for StreamReader {
    fn drop(&mut self) {
        if let Some(abort) = self.abort.take() {
            abort.abort();
        }
    }
}

fn spawn_stream_reader<R>(stream: Option<R>, capture_cap: usize) -> StreamReader
where
    R: AsyncRead + Unpin + Send + 'static,
{
    let state = Arc::new(Mutex::new(CapturedStream::default()));
    let Some(mut reader) = stream else {
        return StreamReader {
            handle: None,
            abort: None,
            state,
        };
    };

    let task_state = Arc::clone(&state);
    let handle = tokio::spawn(async move {
        let mut chunk = [0u8; READ_CHUNK_SIZE];
        loop {
            let n = match reader.read(&mut chunk).await {
                Ok(0) | Err(_) => break,
                Ok(n) => n,
            };

            let mut guard = task_state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if guard.bytes.len() < capture_cap {
                let remaining = capture_cap - guard.bytes.len();
                let to_take = remaining.min(n);
                guard.bytes.extend_from_slice(&chunk[..to_take]);
                if n > to_take {
                    guard.overflowed = true;
                }
            } else {
                guard.overflowed = true;
            }
        }
    });
    let abort = handle.abort_handle();

    StreamReader {
        handle: Some(handle),
        abort: Some(abort),
        state,
    }
}

async fn finish_readers(
    mut stdout: StreamReader,
    mut stderr: StreamReader,
) -> (CapturedStream, CapturedStream) {
    let mut stdout_handle = stdout.handle.take();
    let mut stderr_handle = stderr.handle.take();
    let mut stdout_done = stdout_handle.is_none();
    let mut stderr_done = stderr_handle.is_none();

    let _ = tokio::time::timeout(
        READER_DRAIN_TIMEOUT,
        std::future::poll_fn(|cx| {
            if let (false, Some(handle)) = (stdout_done, stdout_handle.as_mut()) {
                stdout_done = Pin::new(handle).poll(cx).is_ready();
            }
            if let (false, Some(handle)) = (stderr_done, stderr_handle.as_mut()) {
                stderr_done = Pin::new(handle).poll(cx).is_ready();
            }
            if stdout_done && stderr_done {
                Poll::Ready(())
            } else {
                Poll::Pending
            }
        }),
    )
    .await;

    if let (false, Some(handle)) = (stdout_done, stdout_handle) {
        handle.abort();
        let _ = handle.await;
    }
    if let (false, Some(handle)) = (stderr_done, stderr_handle) {
        handle.abort();
        let _ = handle.await;
    }

    (stdout.take_captured(), stderr.take_captured())
}

fn redact_and_truncate(
    raw: &[u8],
    overflowed: bool,
    secrets: &[&str],
    log_cap: usize,
) -> (String, bool) {
    let mut text = String::from_utf8_lossy(raw).into_owned();
    for secret in secrets {
        if !secret.is_empty() {
            text = text.replace(secret, REDACTED);
        }
    }

    let truncated = overflowed || text.len() > log_cap;
    if text.len() > log_cap {
        let end = floor_char_boundary(&text, log_cap);
        text.truncate(end);
    }

    (text, truncated)
}

fn floor_char_boundary(s: &str, max_bytes: usize) -> usize {
    if max_bytes >= s.len() {
        return s.len();
    }
    let mut idx = max_bytes;
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn format_timeout(dur: Duration) -> String {
    if dur.subsec_nanos() == 0 {
        format!("timed out after {}s", dur.as_secs())
    } else {
        format!("timed out after {:.3}s", dur.as_secs_f64())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::io::Write;
    use std::path::Path;
    use std::time::Duration;

    use mallops_core::CancelSource;

    use super::*;

    const CHILD_TEST_NAME: &str = "train::external::process::tests::child_entry";
    const MODE_MARKER: &str = "mallops_child_mode:";

    fn base_env() -> Vec<(String, String)> {
        let mut env = Vec::new();
        if let Ok(path) = std::env::var("PATH") {
            env.push(("PATH".to_string(), path));
        }
        #[cfg(windows)]
        if let Ok(sr) = std::env::var("SYSTEMROOT") {
            env.push(("SYSTEMROOT".to_string(), sr));
        }
        env
    }

    fn child_argv(mode: &str, extra_args: &[&str]) -> Vec<String> {
        let exe = std::env::current_exe()
            .expect("resolve current test binary")
            .to_string_lossy()
            .into_owned();
        let mut argv = vec![
            exe,
            "--exact".to_string(),
            CHILD_TEST_NAME.to_string(),
            "--ignored".to_string(),
            "--nocapture".to_string(),
            "--test-threads=1".to_string(),
            format!("{MODE_MARKER}{mode}"),
        ];
        argv.extend(extra_args.iter().map(|s| (*s).to_string()));
        argv
    }

    fn write_file_atomic(path: &Path, contents: &str) {
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, contents).expect("write temp marker file");
        std::fs::rename(&tmp, path).expect("rename marker file into place");
    }

    async fn wait_for_file(path: &Path, timeout: Duration) -> String {
        tokio::time::timeout(timeout, async {
            loop {
                match std::fs::read_to_string(path) {
                    Ok(contents) if !contents.is_empty() => return contents,
                    _ => {}
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("timed out waiting for marker file")
    }

    #[cfg(unix)]
    fn pid_is_alive(pid: u32) -> bool {
        std::process::Command::new("kill")
            .arg("-0")
            .arg(pid.to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .expect("run kill -0")
            .success()
    }

    #[cfg(unix)]
    async fn wait_until_pid_dead(pid: u32, timeout: Duration) {
        tokio::time::timeout(timeout, async {
            while pid_is_alive(pid) {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("grandchild process remained alive after process-group kill");
    }

    /// Re-executed test binary entry point acting as the external child process.
    #[test]
    #[ignore]
    fn child_entry() {
        let args: Vec<String> = std::env::args().collect();
        let Some(pos) = args.iter().position(|a| a.starts_with(MODE_MARKER)) else {
            return;
        };
        let mode = &args[pos][MODE_MARKER.len()..];
        let rest = &args[pos + 1..];

        match mode {
            "echo_args" => {
                let encoded = serde_json::to_string(rest).expect("encode args");
                let mut stderr = std::io::stderr().lock();
                stderr.write_all(encoded.as_bytes()).expect("write stderr");
                stderr.flush().expect("flush stderr");
            }
            "dump_env" => {
                let out_path = Path::new(&rest[0]);
                let vars: BTreeMap<String, String> = std::env::vars().collect();
                let encoded = serde_json::to_string(&vars).expect("encode env");
                write_file_atomic(out_path, &encoded);
                if let Ok(secret) = std::env::var("TEST_SECRET") {
                    let mut stdout = std::io::stdout().lock();
                    writeln!(stdout, "stdout-leak:{secret}:end").expect("write stdout");
                    stdout.flush().expect("flush stdout");

                    let mut stderr = std::io::stderr().lock();
                    write!(stderr, "stderr-leak:{secret}:end").expect("write stderr");
                    stderr.flush().expect("flush stderr");
                }
            }
            "emit_bytes" => {
                let stdout_len: usize = rest[0].parse().expect("stdout byte count");
                let stderr_len: usize = rest[1].parse().expect("stderr byte count");
                let chunk_out = vec![b'O'; 8192];
                let chunk_err = vec![b'E'; 8192];

                let mut stdout = std::io::stdout().lock();
                let mut rem = stdout_len;
                while rem > 0 {
                    let n = rem.min(chunk_out.len());
                    stdout.write_all(&chunk_out[..n]).expect("write stdout");
                    rem -= n;
                }
                stdout.flush().expect("flush stdout");

                let mut stderr = std::io::stderr().lock();
                let mut rem = stderr_len;
                while rem > 0 {
                    let n = rem.min(chunk_err.len());
                    stderr.write_all(&chunk_err[..n]).expect("write stderr");
                    rem -= n;
                }
                stderr.flush().expect("flush stderr");
            }
            "emit_secret_boundary" => {
                let prefix_len: usize = rest[0].parse().expect("prefix len");
                let suffix_len: usize = rest[1].parse().expect("suffix len");
                let secret = std::env::var("TEST_SECRET").unwrap_or_default();
                let mut stderr = std::io::stderr().lock();
                stderr
                    .write_all(&vec![b'A'; prefix_len])
                    .expect("write prefix");
                stderr.write_all(secret.as_bytes()).expect("write secret");
                stderr
                    .write_all(&vec![b'B'; suffix_len])
                    .expect("write suffix");
                stderr.flush().expect("flush stderr");
            }
            "exit_code" => {
                let code: i32 = rest[0].parse().expect("exit code");
                let mut stderr = std::io::stderr().lock();
                write!(stderr, "exiting with {code}").expect("write stderr");
                stderr.flush().expect("flush stderr");
                std::process::exit(code);
            }
            "sleep" => {
                let ready_path = Path::new(&rest[0]);
                let secs: u64 = rest[1].parse().expect("sleep secs");
                write_file_atomic(ready_path, "ready");
                std::thread::sleep(Duration::from_secs(secs));
            }
            "spawn_grandchild" => {
                let pid_path = Path::new(&rest[0]);
                let behavior = rest[1].as_str();
                let exe = std::env::current_exe().expect("resolve current exe");
                // Deliberately leave the grandchild running when `behavior == "exit_immediately"`
                // so the test can verify `run_child`'s post-exit `killpg` cleans it up.
                #[allow(clippy::zombie_processes)]
                let mut grandchild = std::process::Command::new(exe)
                    .args([
                        "--exact",
                        CHILD_TEST_NAME,
                        "--ignored",
                        "--nocapture",
                        "--test-threads=1",
                        &format!("{MODE_MARKER}grandchild"),
                        &rest[0],
                    ])
                    .stdin(Stdio::null())
                    .spawn()
                    .expect("spawn grandchild");

                // Wait until grandchild has published its PID before acting.
                let deadline = std::time::Instant::now() + Duration::from_secs(10);
                while std::fs::read_to_string(pid_path)
                    .map(|s| s.is_empty())
                    .unwrap_or(true)
                {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "grandchild failed to write pid file"
                    );
                    std::thread::sleep(Duration::from_millis(20));
                }

                if behavior == "sleep" {
                    let _ = grandchild.wait();
                }
            }
            "grandchild" => {
                let pid_path = Path::new(&rest[0]);
                write_file_atomic(pid_path, &std::process::id().to_string());
                std::thread::sleep(Duration::from_secs(60));
            }
            other => panic!("unknown child mode: {other}"),
        }
    }

    #[tokio::test]
    async fn train_external_security_argv_literal_without_shell() {
        let dir = tempfile::tempdir().expect("tempdir");
        let literal_args = [
            "$HOME;echo x",
            "$(whoami)",
            "`id`",
            "a && b | c > /tmp/should-not-exist",
            "quote'and\"double",
            "spaces and\ttabs",
        ];
        let spec = ChildSpec {
            argv: child_argv("echo_args", &literal_args),
            cwd: dir.path().to_path_buf(),
            env: base_env(),
            secret_env: vec![],
            timeout: Some(Duration::from_secs(10)),
            log_cap: 64 * 1024,
        };
        let cancel = CancelSource::new();
        let output = run_child(spec, &cancel.token())
            .await
            .expect("child should succeed");

        assert!(output.status.success());
        let received: Vec<String> =
            serde_json::from_str(&output.stderr).expect("parse echoed args json");
        assert_eq!(received, literal_args);
        assert!(!output.truncated[1]);
    }

    #[tokio::test]
    async fn train_external_security_env_cleared_and_secret_redacted() {
        // Cargo always exports CARGO_MANIFEST_DIR into test processes; use it without set_var.
        assert!(
            std::env::var("CARGO_MANIFEST_DIR").is_ok(),
            "expected CARGO_MANIFEST_DIR in test runner environment"
        );

        let dir = tempfile::tempdir().expect("tempdir");
        let env_dump_path = dir.path().join("child_env.json");
        let secret_val = "s3cr3t-value-never-leak-987654";

        let mut env = base_env();
        env.push(("ALLOWED_VAR".to_string(), "allowed-ok".to_string()));

        let spec = ChildSpec {
            argv: child_argv("dump_env", &[env_dump_path.to_str().expect("utf8 path")]),
            cwd: dir.path().to_path_buf(),
            env,
            secret_env: vec![
                ("TEST_SECRET".to_string(), Secret::new(secret_val)),
                ("EMPTY_SECRET".to_string(), Secret::new("")),
            ],
            timeout: Some(Duration::from_secs(10)),
            log_cap: 64 * 1024,
        };
        let cancel = CancelSource::new();
        let output = run_child(spec, &cancel.token())
            .await
            .expect("child should succeed");

        assert!(output.status.success());

        let child_env: BTreeMap<String, String> = serde_json::from_str(
            &std::fs::read_to_string(&env_dump_path).expect("read dumped env"),
        )
        .expect("parse dumped env");
        assert!(
            !child_env.contains_key("CARGO_MANIFEST_DIR"),
            "host env variable CARGO_MANIFEST_DIR leaked into child"
        );
        assert_eq!(
            child_env.get("ALLOWED_VAR").map(String::as_str),
            Some("allowed-ok")
        );
        assert_eq!(
            child_env.get("TEST_SECRET").map(String::as_str),
            Some(secret_val)
        );

        assert!(
            !output.stderr.contains(secret_val),
            "secret leaked in stderr: {}",
            output.stderr
        );
        assert_eq!(output.stderr, "stderr-leak:***:end");
        assert!(
            !output.stdout.contains(secret_val),
            "secret leaked in stdout: {}",
            output.stdout
        );
        assert!(output.stdout.contains("stdout-leak:***:end"));
    }

    #[tokio::test]
    async fn train_external_security_redacts_secret_straddling_log_cap() {
        let dir = tempfile::tempdir().expect("tempdir");
        let secret_val = "TOPSECRET_BOUNDARY_VALUE";
        // log_cap = 20; 16 'A's + 24-byte secret + 20 'B's -> secret crosses byte 20.
        let spec = ChildSpec {
            argv: child_argv("emit_secret_boundary", &["16", "20"]),
            cwd: dir.path().to_path_buf(),
            env: base_env(),
            secret_env: vec![("TEST_SECRET".to_string(), Secret::new(secret_val))],
            timeout: Some(Duration::from_secs(10)),
            log_cap: 20,
        };
        let cancel = CancelSource::new();
        let output = run_child(spec, &cancel.token())
            .await
            .expect("child should succeed");

        assert!(output.status.success());
        assert!(output.stderr.len() <= 20);
        assert!(output.truncated[1]);
        assert!(!output.stderr.contains("TOPSECRET"));
        assert_eq!(output.stderr, format!("{}***B", "A".repeat(16)));
    }

    #[tokio::test]
    async fn train_external_timeout_and_cancel_timeout_returns_runtime_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ready_path = dir.path().join("ready");
        let spec = ChildSpec {
            argv: child_argv("sleep", &[ready_path.to_str().expect("utf8 path"), "30"]),
            cwd: dir.path().to_path_buf(),
            env: base_env(),
            secret_env: vec![],
            timeout: Some(Duration::from_secs(1)),
            log_cap: 4096,
        };
        let cancel = CancelSource::new();

        let err = tokio::time::timeout(Duration::from_secs(10), run_child(spec, &cancel.token()))
            .await
            .expect("run_child did not return promptly after timeout")
            .expect_err("expected timeout error");

        match err {
            Error::Runtime(msg) => assert_eq!(msg, "timed out after 1s"),
            other => panic!("expected Error::Runtime timeout, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn train_external_timeout_and_cancel_cancel_returns_cancelled() {
        let dir = tempfile::tempdir().expect("tempdir");
        let ready_path = dir.path().join("ready");
        let spec = ChildSpec {
            argv: child_argv("sleep", &[ready_path.to_str().expect("utf8 path"), "30"]),
            cwd: dir.path().to_path_buf(),
            env: base_env(),
            secret_env: vec![],
            timeout: Some(Duration::from_secs(30)),
            log_cap: 4096,
        };
        let cancel = CancelSource::new();
        let token = cancel.token();

        let task = tokio::spawn(async move { run_child(spec, &token).await });

        let _ = wait_for_file(&ready_path, Duration::from_secs(10)).await;
        cancel.cancel();

        let err = tokio::time::timeout(Duration::from_secs(10), task)
            .await
            .expect("run_child did not return promptly after cancel")
            .expect("join run_child task")
            .expect_err("expected cancellation error");

        assert!(
            matches!(err, Error::Cancelled),
            "expected Error::Cancelled, got {err:?}"
        );
    }

    #[tokio::test]
    async fn train_external_timeout_and_cancel_precancelled_skips_spawn() {
        let dir = tempfile::tempdir().expect("tempdir");
        let spec = ChildSpec {
            argv: vec!["nonexistent-binary-that-must-not-spawn".to_string()],
            cwd: dir.path().to_path_buf(),
            env: base_env(),
            secret_env: vec![],
            timeout: None,
            log_cap: 1024,
        };
        let cancel = CancelSource::new();
        cancel.cancel();

        let err = run_child(spec, &cancel.token())
            .await
            .expect_err("pre-cancelled token must fail immediately");
        assert!(matches!(err, Error::Cancelled));
    }

    #[tokio::test]
    async fn train_external_timeout_and_cancel_caps_and_drains_streams() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Write 256 KiB per stream (well above typical OS pipe buffers) with a 256 B cap.
        let spec = ChildSpec {
            argv: child_argv("emit_bytes", &["262144", "262144"]),
            cwd: dir.path().to_path_buf(),
            env: base_env(),
            secret_env: vec![],
            timeout: Some(Duration::from_secs(15)),
            log_cap: 256,
        };
        let cancel = CancelSource::new();
        let output =
            tokio::time::timeout(Duration::from_secs(15), run_child(spec, &cancel.token()))
                .await
                .expect("draining large streams must not deadlock")
                .expect("child should succeed");

        assert!(output.status.success());
        assert_eq!(output.truncated, [true, true]);
        assert!(output.stdout.len() <= 256);
        assert_eq!(output.stderr.len(), 256);
        assert_eq!(output.stderr, "E".repeat(256));
    }

    #[tokio::test]
    async fn train_external_timeout_and_cancel_nonzero_exit_is_ok() {
        let dir = tempfile::tempdir().expect("tempdir");
        let spec = ChildSpec {
            argv: child_argv("exit_code", &["42"]),
            cwd: dir.path().to_path_buf(),
            env: base_env(),
            secret_env: vec![],
            timeout: Some(Duration::from_secs(10)),
            log_cap: 4096,
        };
        let cancel = CancelSource::new();
        let output = run_child(spec, &cancel.token())
            .await
            .expect("non-zero exit must be returned in ChildOutput");

        assert_eq!(output.status.code(), Some(42));
        assert_eq!(output.stderr, "exiting with 42");
        assert!(!output.truncated[1]);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn train_external_timeout_and_cancel_kills_process_group_on_cancel() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pid_path = dir.path().join("grandchild.pid");
        let spec = ChildSpec {
            argv: child_argv(
                "spawn_grandchild",
                &[pid_path.to_str().expect("utf8 path"), "sleep"],
            ),
            cwd: dir.path().to_path_buf(),
            env: base_env(),
            secret_env: vec![],
            timeout: Some(Duration::from_secs(30)),
            log_cap: 4096,
        };
        let cancel = CancelSource::new();
        let token = cancel.token();

        let task = tokio::spawn(async move { run_child(spec, &token).await });

        let pid: u32 = wait_for_file(&pid_path, Duration::from_secs(10))
            .await
            .trim()
            .parse()
            .expect("parse grandchild pid");
        assert!(
            pid_is_alive(pid),
            "grandchild {pid} should be alive before cancel"
        );

        cancel.cancel();

        let err = tokio::time::timeout(Duration::from_secs(10), task)
            .await
            .expect("run_child did not finish after cancel")
            .expect("join run_child task")
            .expect_err("expected Error::Cancelled");
        assert!(matches!(err, Error::Cancelled));

        wait_until_pid_dead(pid, Duration::from_secs(5)).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn train_external_timeout_and_cancel_kills_process_group_on_normal_exit() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pid_path = dir.path().join("grandchild.pid");
        let spec = ChildSpec {
            argv: child_argv(
                "spawn_grandchild",
                &[pid_path.to_str().expect("utf8 path"), "exit_immediately"],
            ),
            cwd: dir.path().to_path_buf(),
            env: base_env(),
            secret_env: vec![],
            timeout: Some(Duration::from_secs(30)),
            log_cap: 4096,
        };
        let cancel = CancelSource::new();

        let output =
            tokio::time::timeout(Duration::from_secs(10), run_child(spec, &cancel.token()))
                .await
                .expect("run_child should return promptly even with surviving grandchild")
                .expect("child exited cleanly");
        assert!(output.status.success());

        let pid: u32 = std::fs::read_to_string(&pid_path)
            .expect("read grandchild pid")
            .trim()
            .parse()
            .expect("parse grandchild pid");

        wait_until_pid_dead(pid, Duration::from_secs(5)).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn train_external_timeout_and_cancel_kills_process_group_on_timeout() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pid_path = dir.path().join("grandchild.pid");
        let spec = ChildSpec {
            argv: child_argv(
                "spawn_grandchild",
                &[pid_path.to_str().expect("utf8 path"), "sleep"],
            ),
            cwd: dir.path().to_path_buf(),
            env: base_env(),
            secret_env: vec![],
            timeout: Some(Duration::from_secs(2)),
            log_cap: 4096,
        };
        let cancel = CancelSource::new();
        let token = cancel.token();

        let task = tokio::spawn(async move { run_child(spec, &token).await });

        let pid: u32 = wait_for_file(&pid_path, Duration::from_secs(10))
            .await
            .trim()
            .parse()
            .expect("parse grandchild pid");
        assert!(
            pid_is_alive(pid),
            "grandchild {pid} should be alive before timeout"
        );

        let err = tokio::time::timeout(Duration::from_secs(10), task)
            .await
            .expect("run_child did not return after timeout")
            .expect("join run_child task")
            .expect_err("expected timeout error");
        match err {
            Error::Runtime(msg) => assert_eq!(msg, "timed out after 2s"),
            other => panic!("expected Error::Runtime timeout, got {other:?}"),
        }

        wait_until_pid_dead(pid, Duration::from_secs(5)).await;
    }

    #[tokio::test]
    async fn train_external_timeout_and_cancel_spawn_failure_returns_runtime_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let missing = dir.path().join("does-not-exist-binary");
        let spec = ChildSpec {
            argv: vec![missing.to_string_lossy().into_owned()],
            cwd: dir.path().to_path_buf(),
            env: base_env(),
            secret_env: vec![],
            timeout: None,
            log_cap: 1024,
        };
        let cancel = CancelSource::new();

        let err = run_child(spec, &cancel.token())
            .await
            .expect_err("missing binary must fail");
        assert!(
            matches!(err, Error::Runtime(_)),
            "expected Error::Runtime, got {err:?}"
        );
    }

    #[test]
    fn redact_and_truncate_handles_utf8_boundaries_and_overlapping_secrets() {
        let (out, trunc) = redact_and_truncate(b"prefix abcdef abc", false, &["abcdef", "abc"], 64);
        assert_eq!(out, "prefix *** ***");
        assert!(!trunc);

        // Multi-byte UTF-8 character 'é' (2 bytes) at boundary: must not panic or split char.
        let (out, trunc) = redact_and_truncate("aaaaé".as_bytes(), false, &[], 5);
        assert_eq!(out, "aaaa");
        assert!(trunc);

        // Overflowed stream remains flagged truncated even if redaction shrinks below log_cap.
        let (out, trunc) = redact_and_truncate(b"longsecretvalue", true, &["longsecretvalue"], 10);
        assert_eq!(out, "***");
        assert!(trunc);
    }
}
