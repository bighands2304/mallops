//! Versioned file contract v1 between `std/train.external` and the external command.
//!
//! mallops writes `inputs.json` and `config.json` into the node workdir (the child's cwd); the
//! command writes `outputs.json` there. Every file carries `protocol_version: 1`.

use std::collections::BTreeMap;
use std::path::Path;

use mallops_core::{ArtifactKind, DraftPayload, Error, Manifest};
use serde::{Deserialize, Serialize};

/// Contract version written to and required in every contract file.
pub(crate) const PROTOCOL_VERSION: u32 = 1;
/// Input description written by mallops.
pub(crate) const INPUTS_FILE: &str = "inputs.json";
/// Block configuration written by mallops (never contains secrets).
pub(crate) const CONFIG_FILE: &str = "config.json";
/// Output description written by the command.
pub(crate) const OUTPUTS_FILE: &str = "outputs.json";

/// `inputs.json`: input port name → materialized, read-only copy in the workdir.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct InputsFile {
    pub(crate) protocol_version: u32,
    pub(crate) inputs: BTreeMap<String, InputEntry>,
}

/// One materialized input artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct InputEntry {
    /// Artifact id (hex).
    pub(crate) id: String,
    /// Payload directory relative to the workdir, `/`-separated (e.g. `inputs/train`).
    pub(crate) path: String,
    /// Artifact manifest.
    pub(crate) manifest: Manifest,
}

/// `config.json`: `with.flavor`, `with.params` and the run id. Never contains resolved secrets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ConfigFile {
    pub(crate) protocol_version: u32,
    pub(crate) flavor: Option<String>,
    /// `with.params` (an object; `{}` when absent).
    pub(crate) params: serde_json::Value,
    pub(crate) run_id: String,
}

/// `outputs.json` written by the command.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OutputsFile {
    pub(crate) protocol_version: u32,
    pub(crate) status: OutputStatus,
    #[serde(default)]
    pub(crate) skip_reason: Option<String>,
    #[serde(default)]
    pub(crate) outputs: BTreeMap<String, OutputEntry>,
}

/// Whether the command produced its outputs or chose to skip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum OutputStatus {
    Produced,
    Skipped,
}

/// One output artifact declared by the command.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OutputEntry {
    /// File or directory relative to the workdir (validated as `RelPath`, no symlinks).
    pub(crate) path: String,
    pub(crate) kind: ArtifactKind,
    pub(crate) format: String,
    /// Absent → the configured `with.flavor` (model ports only).
    #[serde(default)]
    pub(crate) flavor: Option<String>,
    #[serde(default)]
    pub(crate) io_schema: Option<serde_json::Value>,
    #[serde(default)]
    pub(crate) metadata: BTreeMap<String, serde_json::Value>,
}

/// Reads and parses `outputs.json` from `workdir`, requiring `protocol_version == 1`.
///
/// # Errors
///
/// [`Error::Runtime`] if the file is missing, malformed or has another protocol version.
pub(crate) fn read_outputs(workdir: &Path) -> Result<OutputsFile, Error> {
    let _ = workdir;
    todo!("13.1")
}

/// Validates an `outputs.json` path and returns the matching draft payload.
///
/// `rel` must parse as `RelPath`; no component under `workdir` may be a symlink; the target must be
/// a regular file ([`DraftPayload::File`]) or a directory ([`DraftPayload::Dir`]).
///
/// # Errors
///
/// [`Error::Runtime`] naming the offending path.
pub(crate) fn output_payload(workdir: &Path, rel: &str) -> Result<DraftPayload, Error> {
    let _ = (workdir, rel);
    todo!("13.1")
}
