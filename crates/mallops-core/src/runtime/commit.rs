//! Runtime-owned commit of block output drafts (ADR-0007 D4).

use std::collections::BTreeMap;
use std::path::Path;

use crate::{ArtifactId, ArtifactStore, Error, OutputDrafts, ResolvedPorts};

/// Validates `drafts` against the node's resolved output `ports` and commits them.
///
/// - every draft must name a declared output port and satisfy it (`Port::accepts_manifest`);
/// - when `produced` (outcome `Produced`), every required output port must have a draft;
///   for `Skipped` outcomes drafts are optional;
/// - `File`/`Dir` payloads must resolve inside `workdir` (no `..` or symlink escapes);
/// - `manifest.parents` is replaced by `parents` (the node's actual input ids);
/// - drafts are committed with [`ArtifactStore::put`] only if all checks pass.
///
/// Returns the committed ids by output port.
///
/// # Errors
///
/// [`Error::Validation`] with every violation, paths `out.<port>`; store errors as returned.
#[allow(dead_code)] // 7.1 wires it into the scheduler.
pub(crate) async fn commit_outputs(
    store: &dyn ArtifactStore,
    ports: &ResolvedPorts,
    drafts: OutputDrafts,
    parents: &[ArtifactId],
    workdir: &Path,
    produced: bool,
) -> Result<BTreeMap<String, ArtifactId>, Error> {
    // 7.2: implementation.
    let _ = (store, ports, drafts, parents, workdir, produced);
    Err(Error::Runtime(
        "commit_outputs not implemented yet".to_string(),
    ))
}
