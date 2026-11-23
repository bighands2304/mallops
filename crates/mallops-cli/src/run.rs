//! `run` (with `runs/latest.json`) and `store verify` (task 10.2).

use std::path::Path;

/// Executes the batch DAG of `file`, prints a summary and records `runs/latest.json` on success.
pub(crate) fn run(_file: &Path, _jobs: Option<usize>, _no_cache: bool) -> anyhow::Result<()> {
    anyhow::bail!("todo: 10.2")
}

/// Verifies artifact `id` in the store configured by `file`.
pub(crate) fn verify(_file: &Path, _id: &str) -> anyhow::Result<()> {
    anyhow::bail!("todo: 10.2")
}
