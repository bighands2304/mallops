//! Child process execution for `std/train.external`: argv only, cleared environment, process
//! group, timeout/cancel, capped and redacted output. Knows nothing about `Context`.

use std::path::PathBuf;
use std::process::ExitStatus;
use std::time::Duration;

use mallops_core::{CancelToken, Error, Secret};
// ponytail: placeholder until 13.2 calls `killpg`.
#[cfg(unix)]
use nix as _;

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
    let _ = (spec, cancel);
    todo!("13.2")
}
