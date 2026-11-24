//! Child process execution for `std/train.external`: argv only, cleared environment, process
//! group, timeout/cancel, capped and redacted output. Knows nothing about `Context`.

use std::future::Future;
use std::path::PathBuf;
use std::pin::pin;
use std::process::{ExitStatus, Stdio};
use std::task::Poll;
use std::time::Duration;

use mallops_core::{CancelToken, Error, Secret};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;
use tokio::task::JoinHandle;

/// Maximum time to wait for each stdout/stderr reader to drain after the child exits.
///
/// Guards against a surviving descendant holding an inherited pipe open.
const READER_DRAIN_TIMEOUT: Duration = Duration::from_secs(1);

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

    // Taken right after spawn, while the child is unreaped; also fires if this future is dropped.
    #[cfg(unix)]
    let mut group = Killpg(
        child
            .id()
            .and_then(|id| i32::try_from(id).ok())
            .filter(|&id| id > 0)
            .map(nix::unistd::Pid::from_raw),
    );

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
    let capture_cap = spec
        .log_cap
        .saturating_add(secrets.first().map_or(0, |s| s.len()));

    let readers = [
        spawn_reader(child.stdout.take(), capture_cap),
        spawn_reader(child.stderr.take(), capture_cap),
    ];

    // ponytail: hand-rolled select, `tokio::select!` needs the `macros` feature.
    let waited = {
        let mut cancelled = pin!(cancel.cancelled());
        let mut exited = pin!(child.wait());
        let mut deadline = pin!(async {
            match spec.timeout {
                Some(dur) => {
                    tokio::time::sleep(dur).await;
                    dur.as_secs()
                }
                None => std::future::pending().await,
            }
        });
        std::future::poll_fn(|cx| {
            if cancelled.as_mut().poll(cx).is_ready() {
                return Poll::Ready(Err(Error::Cancelled));
            }
            if let Poll::Ready(status) = exited.as_mut().poll(cx) {
                return Poll::Ready(Ok(status));
            }
            deadline
                .as_mut()
                .poll(cx)
                .map(|secs| Err(Error::Runtime(format!("timed out after {secs}s"))))
        })
        .await
    };

    // Every outcome: after a normal exit this reaps surviving grandchildren.
    #[cfg(unix)]
    group.kill();
    let status = match waited {
        Ok(status) => status
            .map_err(|err| Error::Runtime(format!("failed waiting for `{program}`: {err}")))?,
        Err(err) => {
            // Always, even after killpg: a child that left its process group escapes it.
            let _ = child.start_kill();
            let _ = child.wait().await;
            for reader in readers {
                reader.abort();
            }
            return Err(err);
        }
    };

    let [stdout, stderr] = readers;
    let (stdout, stdout_truncated) = drain(stdout, &secrets, spec.log_cap).await;
    let (stderr, stderr_truncated) = drain(stderr, &secrets, spec.log_cap).await;
    Ok(ChildOutput {
        status,
        stdout,
        stderr,
        truncated: [stdout_truncated, stderr_truncated],
    })
}

/// SIGKILLs the child's process group once, on [`Killpg::kill`] or on drop.
#[cfg(unix)]
struct Killpg(Option<nix::unistd::Pid>);

#[cfg(unix)]
impl Killpg {
    fn kill(&mut self) {
        if let Some(pgid) = self.0.take() {
            // ESRCH (group already gone) is expected.
            let _ = nix::sys::signal::killpg(pgid, nix::sys::signal::Signal::SIGKILL);
        }
    }
}

#[cfg(unix)]
impl Drop for Killpg {
    fn drop(&mut self) {
        self.kill();
    }
}

/// Reads `stream` to EOF keeping the first `cap` bytes; the flag is set if anything was dropped
/// (or reading failed).
fn spawn_reader<R>(stream: Option<R>, cap: usize) -> JoinHandle<(Vec<u8>, bool)>
where
    R: AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut bytes = Vec::new();
        let Some(mut stream) = stream else {
            return (bytes, false);
        };
        let kept = (&mut stream).take(cap as u64).read_to_end(&mut bytes).await;
        let rest = tokio::io::copy(&mut stream, &mut tokio::io::sink()).await;
        (bytes, kept.is_err() || !matches!(rest, Ok(0)))
    })
}

/// Waits (bounded) for a reader; a descendant still holding the pipe loses that stream's output.
async fn drain(
    mut reader: JoinHandle<(Vec<u8>, bool)>,
    secrets: &[&str],
    log_cap: usize,
) -> (String, bool) {
    let (bytes, overflowed) = match tokio::time::timeout(READER_DRAIN_TIMEOUT, &mut reader).await {
        Ok(Ok(captured)) => captured,
        _ => {
            reader.abort();
            (Vec::new(), true)
        }
    };
    redact_and_truncate(&bytes, overflowed, secrets, log_cap)
}

/// Redacts `secrets` (non-empty, longest first) from `raw`, then truncates to `log_cap` bytes.
fn redact_and_truncate(
    raw: &[u8],
    overflowed: bool,
    secrets: &[&str],
    log_cap: usize,
) -> (String, bool) {
    let mut text = String::from_utf8_lossy(raw).into_owned();
    for secret in secrets {
        text = text.replace(secret, REDACTED);
    }
    if overflowed {
        // The capture cut can split a secret (and a char, which lossy turns into U+FFFD): a
        // trailing proper prefix of a secret would survive `replace`, so redact it too.
        if text.ends_with(char::REPLACEMENT_CHARACTER) {
            text.pop();
        }
        let partial = secrets
            .iter()
            .flat_map(|s| {
                (1..s.len()).filter(|&n| s.is_char_boundary(n) && text.ends_with(&s[..n]))
            })
            .max();
        if let Some(n) = partial {
            text.truncate(text.len() - n);
            text.push_str(REDACTED);
        }
    }

    let truncated = overflowed || text.len() > log_cap;
    // ponytail: `str::floor_char_boundary` is stable only above MSRV 1.85.
    let end = (0..=log_cap.min(text.len()))
        .rev()
        .find(|&i| text.is_char_boundary(i))
        .unwrap_or(0);
    text.truncate(end);
    (text, truncated)
}

#[cfg(test)]
mod tests {
    use std::io::Read;
    use std::path::Path;
    use std::time::Duration;

    use mallops_core::CancelSource;

    use super::*;

    const CHILD_TEST_NAME: &str = "train::external::process::tests::child_entry";
    const MODE_MARKER: &str = "mallops_child_mode:";
    const SECRET_VAR: &str = "TEST_SECRET";

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

    /// `argv` run in `dir` with only `PATH` (+ `SYSTEMROOT`) and no secrets.
    fn spec(dir: &Path, argv: Vec<String>, timeout: Option<Duration>, log_cap: usize) -> ChildSpec {
        ChildSpec {
            argv,
            cwd: dir.to_path_buf(),
            env: base_env(),
            secret_env: vec![],
            timeout,
            log_cap,
        }
    }

    /// Runs the `emit_secret` child on `template` with `secret` and the given `log_cap`.
    async fn emit_secret(secret: &str, template: &str, log_cap: usize) -> ChildOutput {
        let dir = tempfile::tempdir().expect("tempdir");
        let spec = ChildSpec {
            secret_env: vec![(SECRET_VAR.to_string(), Secret::new(secret))],
            ..spec(
                dir.path(),
                child_argv("emit_secret", &[template]),
                Some(Duration::from_secs(10)),
                log_cap,
            )
        };
        let output = run_child(spec, &CancelSource::new().token())
            .await
            .expect("child should succeed");
        assert!(output.status.success());
        output
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
    async fn wait_until_pid_dead(pid: u32, timeout: Duration) {
        let pid = nix::unistd::Pid::from_raw(i32::try_from(pid).expect("pid fits i32"));
        tokio::time::timeout(timeout, async {
            while nix::sys::signal::kill(pid, None).is_ok() {
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
            "emit_bytes" => {
                let n = |i: usize| rest[i].parse::<u64>().expect("byte count");
                let mut out = std::io::repeat(b'O').take(n(0));
                std::io::copy(&mut out, &mut std::io::stdout()).expect("write stdout");
                let mut err = std::io::repeat(b'E').take(n(1));
                std::io::copy(&mut err, &mut std::io::stderr()).expect("write stderr");
            }
            // Prints the template in `rest[0]` to stderr, each `@` replaced by the secret.
            "emit_secret" => {
                eprint!(
                    "{}",
                    rest[0].replace('@', &std::env::var(SECRET_VAR).unwrap())
                );
            }
            "exit_code" => {
                let code: i32 = rest[0].parse().expect("exit code");
                eprint!("exiting with {code}");
                std::process::exit(code);
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
    async fn train_external_security_redacts_secret_straddling_log_cap() {
        // log_cap = 20; 16 'A's + 24-byte secret + 20 'B's -> secret crosses byte 20.
        let template = format!("{}@{}", "A".repeat(16), "B".repeat(20));
        let output = emit_secret("TOPSECRET_BOUNDARY_VALUE", &template, 20).await;
        assert!(output.truncated[1]);
        assert_eq!(output.stderr, format!("{}***B", "A".repeat(16)));
    }

    #[tokio::test]
    async fn train_external_security_redacts_secret_cut_at_capture_cap() {
        // Redacting the first occurrence pulls the second, cut at log_cap + secret len, into
        // the log_cap window: no byte of it may survive.
        let pad = |n| "A".repeat(n);
        for (secret, template) in [
            ("0123456789".repeat(10), "@@".to_owned()),
            ("0123456789".to_owned(), format!("{}@{}@", pad(10), pad(45))),
        ] {
            let output = emit_secret(&secret, &template, 64).await;
            assert!(output.truncated[1]);
            assert!(
                !output.stderr.contains(|c: char| c.is_ascii_digit()),
                "{template}: secret bytes leaked: {}",
                output.stderr
            );
        }
    }

    #[tokio::test]
    async fn train_external_timeout_and_cancel_precancelled_skips_spawn() {
        let dir = tempfile::tempdir().expect("tempdir");
        let argv = vec!["nonexistent-binary-that-must-not-spawn".to_string()];
        let cancel = CancelSource::new();
        cancel.cancel();

        let err = run_child(spec(dir.path(), argv, None, 1024), &cancel.token())
            .await
            .expect_err("pre-cancelled token must fail immediately");
        assert!(matches!(err, Error::Cancelled));
    }

    #[tokio::test]
    async fn train_external_timeout_and_cancel_caps_and_drains_streams() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Write 256 KiB per stream (well above typical OS pipe buffers) with a 256 B cap.
        let argv = child_argv("emit_bytes", &["262144", "262144"]);
        let spec = spec(dir.path(), argv, Some(Duration::from_secs(15)), 256);
        let output = tokio::time::timeout(
            Duration::from_secs(15),
            run_child(spec, &CancelSource::new().token()),
        )
        .await
        .expect("draining large streams must not deadlock")
        .expect("child should succeed");

        assert!(output.status.success());
        assert_eq!(output.truncated, [true, true]);
        assert!(output.stdout.len() <= 256);
        assert_eq!(output.stderr, "E".repeat(256));
    }

    #[tokio::test]
    async fn train_external_timeout_and_cancel_nonzero_exit_is_ok() {
        let dir = tempfile::tempdir().expect("tempdir");
        let argv = child_argv("exit_code", &["42"]);
        let spec = spec(dir.path(), argv, Some(Duration::from_secs(10)), 4096);
        let output = run_child(spec, &CancelSource::new().token())
            .await
            .expect("non-zero exit must be returned in ChildOutput");

        assert_eq!(output.status.code(), Some(42));
        assert_eq!(output.stderr, "exiting with 42");
        assert!(!output.truncated[1]);
    }

    /// Runs a child that spawns a sleeping grandchild (`behavior`: `sleep` waits for it,
    /// `exit_immediately` leaves it behind), calls `act` once the grandchild runs, and checks the
    /// grandchild is killed with the group.
    #[cfg(unix)]
    async fn group_killed(
        behavior: &str,
        timeout: Option<Duration>,
        act: impl FnOnce(&CancelSource),
    ) -> Result<ChildOutput, Error> {
        let dir = tempfile::tempdir().expect("tempdir");
        let pid_path = dir.path().join("grandchild.pid");
        let argv = child_argv(
            "spawn_grandchild",
            &[pid_path.to_str().expect("utf8 path"), behavior],
        );
        let spec = spec(dir.path(), argv, timeout, 4096);
        let cancel = CancelSource::new();
        let token = cancel.token();
        let task = tokio::spawn(async move { run_child(spec, &token).await });

        let pid: u32 = wait_for_file(&pid_path, Duration::from_secs(10))
            .await
            .trim()
            .parse()
            .expect("parse grandchild pid");
        act(&cancel);
        let result = tokio::time::timeout(Duration::from_secs(10), task)
            .await
            .expect("run_child did not return")
            .expect("join run_child task");
        wait_until_pid_dead(pid, Duration::from_secs(5)).await;
        result
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn train_external_timeout_and_cancel_kills_process_group_on_cancel() {
        let err = group_killed("sleep", None, CancelSource::cancel).await;
        assert!(matches!(err, Err(Error::Cancelled)), "{err:?}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn train_external_timeout_and_cancel_kills_process_group_on_normal_exit() {
        let output = group_killed("exit_immediately", None, |_| {}).await;
        assert!(output.expect("child exited cleanly").status.success());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn train_external_timeout_and_cancel_kills_process_group_on_timeout() {
        let err = group_killed("sleep", Some(Duration::from_secs(2)), |_| {}).await;
        assert!(
            matches!(&err, Err(Error::Runtime(m)) if m == "timed out after 2s"),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn train_external_timeout_and_cancel_spawn_failure_returns_runtime_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let missing = dir.path().join("does-not-exist-binary");
        let argv = vec![missing.to_string_lossy().into_owned()];
        let err = run_child(
            spec(dir.path(), argv, None, 1024),
            &CancelSource::new().token(),
        )
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

        // A secret cut by the capture inside a multi-byte char is still redacted.
        let (out, _) = redact_and_truncate(&"x:ab€".as_bytes()[..5], true, &["ab€z"], 64);
        assert_eq!(out, "x:***");
    }
}
