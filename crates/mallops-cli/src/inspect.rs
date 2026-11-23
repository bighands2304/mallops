//! `init`, `validate`, `graph`, `blocks list`: never execute blocks or spawn processes (task 10.1).

use std::path::Path;

use crate::GraphFormat;

/// Writes a minimal runnable `mallops.yaml` into `dir`; fails if it already exists.
pub(crate) fn init(_dir: &Path) -> anyhow::Result<()> {
    anyhow::bail!("todo: 10.1")
}

/// Parses and validates `file`; prints `ok` or every validation error.
pub(crate) fn validate(file: &Path) -> anyhow::Result<()> {
    crate::load(file)?;
    anyhow::bail!("todo: 10.1")
}

/// Prints the validated DAG of `file` as Mermaid or DOT.
pub(crate) fn graph(_file: &Path, _format: GraphFormat) -> anyhow::Result<()> {
    anyhow::bail!("todo: 10.1")
}

/// Lists registered blocks (`id`, `version`, `stage`, ports), sorted by id.
pub(crate) fn blocks_list() -> anyhow::Result<()> {
    anyhow::bail!("todo: 10.1")
}
