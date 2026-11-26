//! Versioned file contract v1 between `std/train.external` and the external command.
//!
//! mallops writes `inputs.json` and `config.json` into the node workdir (the child's cwd); the
//! command writes `outputs.json` there. Every file carries `protocol_version: 1`.

use std::collections::BTreeMap;
use std::path::Path;

use mallops_core::{
    ArtifactKind, BlockOutcome, DraftPayload, Error, Manifest, OutputDraft, OutputDrafts, RelPath,
};
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
#[derive(Debug, Serialize)]
pub(crate) struct InputsFile {
    pub(crate) protocol_version: u32,
    pub(crate) inputs: BTreeMap<String, InputEntry>,
}

/// One materialized input artifact.
#[derive(Debug, Serialize)]
pub(crate) struct InputEntry {
    /// Artifact id (hex).
    pub(crate) id: String,
    /// Payload directory relative to the workdir, `/`-separated (e.g. `inputs/train`).
    pub(crate) path: String,
    /// Artifact manifest.
    pub(crate) manifest: Manifest,
}

/// `config.json`: `with.flavor`, `with.params` and the run id. Never contains resolved secrets.
#[derive(Debug, Serialize)]
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
/// The version is checked before the strict parse, so a future version is reported as such
/// rather than as an unknown field.
///
/// # Errors
///
/// [`Error::Runtime`] if the file is missing, not a regular file, malformed or has another
/// protocol version.
pub(crate) fn read_outputs(workdir: &Path) -> Result<OutputsFile, Error> {
    let path = workdir.join(OUTPUTS_FILE);
    let err = |msg: String| Error::Runtime(format!("{OUTPUTS_FILE}: {msg}"));
    match std::fs::symlink_metadata(&path) {
        Ok(meta) if meta.is_file() => {}
        Ok(_) => return Err(err("must be a regular file".into())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(err("not written by the command".into()));
        }
        Err(e) => return Err(err(e.to_string())),
    }
    let bytes = std::fs::read(&path).map_err(|e| err(e.to_string()))?;
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| err(format!("invalid JSON: {e}")))?;
    let version = &value["protocol_version"];
    if version != PROTOCOL_VERSION {
        return Err(err(format!(
            "protocol_version must be {PROTOCOL_VERSION}, got {version}"
        )));
    }
    serde_json::from_value(value).map_err(|e| err(e.to_string()))
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
    let err = |msg: String| Error::Runtime(format!("output path `{rel}`: {msg}"));
    let rel_path = RelPath::parse(rel).map_err(|e| err(e.to_string()))?;
    let mut path = workdir.to_path_buf();
    let mut meta = None;
    for segment in rel_path.as_str().split('/') {
        path.push(segment);
        let m = std::fs::symlink_metadata(&path).map_err(|e| err(e.to_string()))?;
        if m.file_type().is_symlink() {
            return Err(err(format!("`{segment}` is a symlink")));
        }
        meta = Some(m);
    }
    // Belt and braces for what the symlink walk cannot see (e.g. NTFS junctions).
    let root = std::fs::canonicalize(workdir).map_err(|e| err(e.to_string()))?;
    let canon = std::fs::canonicalize(&path).map_err(|e| err(e.to_string()))?;
    if !canon.starts_with(&root) {
        return Err(err("resolves outside the workdir".into()));
    }
    match meta {
        Some(m) if m.is_file() => Ok(DraftPayload::File(path)),
        Some(m) if m.is_dir() => Ok(DraftPayload::Dir(path)),
        _ => Err(err("not a regular file or directory".into())),
    }
}

impl OutputsFile {
    /// Turns the parsed outputs into a block outcome; `flavor` (the configured `with.flavor`)
    /// fills a missing flavor on `model` outputs. Port names, kinds and required outputs are
    /// checked by the runtime when it commits the drafts.
    ///
    /// # Errors
    ///
    /// [`Error::Runtime`] from [`output_payload`], prefixed with the port.
    pub(crate) fn into_outcome(
        self,
        workdir: &Path,
        flavor: Option<&str>,
    ) -> Result<BlockOutcome, Error> {
        let mut drafts = OutputDrafts::new();
        for (port, entry) in self.outputs {
            let payload = output_payload(workdir, &entry.path).map_err(|e| match e {
                Error::Runtime(m) => Error::Runtime(format!("output `{port}`: {m}")),
                other => other,
            })?;
            let mut manifest =
                Manifest::new(entry.kind, entry.format).with_metadata(entry.metadata);
            if let Some(io_schema) = entry.io_schema {
                manifest = manifest.with_io_schema(io_schema);
            }
            let default_flavor = (entry.kind == ArtifactKind::Model)
                .then_some(flavor)
                .flatten();
            if let Some(f) = entry.flavor.as_deref().or(default_flavor) {
                manifest = manifest.with_flavor(f);
            }
            drafts = drafts.with(port, OutputDraft::new(manifest, payload));
        }
        Ok(match self.status {
            OutputStatus::Produced => BlockOutcome::Produced(drafts),
            OutputStatus::Skipped => BlockOutcome::Skipped {
                reason: self
                    .skip_reason
                    .unwrap_or_else(|| "skipped by the command".into()),
                outputs: drafts,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use serde_json::json;

    use super::*;

    fn write_outputs(dir: &Path, value: &serde_json::Value) {
        fs::write(dir.join(OUTPUTS_FILE), value.to_string()).unwrap();
    }

    fn runtime_msg<T: std::fmt::Debug>(r: Result<T, Error>) -> String {
        match r {
            Err(Error::Runtime(m)) => m,
            other => panic!("expected Error::Runtime, got {other:?}"),
        }
    }

    #[test]
    fn train_external_contract_v1_read_outputs_rejects_wrong_version() {
        let dir = tempfile::tempdir().unwrap();
        write_outputs(
            dir.path(),
            &json!({ "protocol_version": 2, "status": "produced", "new": 1 }),
        );
        assert!(runtime_msg(read_outputs(dir.path())).contains("protocol_version must be 1"));
        write_outputs(dir.path(), &json!({ "status": "produced" }));
        assert!(runtime_msg(read_outputs(dir.path())).contains("protocol_version must be 1"));
    }

    #[test]
    fn train_external_contract_v1_read_outputs_rejects_unknown_fields() {
        let dir = tempfile::tempdir().unwrap();
        write_outputs(
            dir.path(),
            &json!({ "protocol_version": 1, "status": "produced", "x": 1 }),
        );
        assert!(runtime_msg(read_outputs(dir.path())).contains("unknown field"));
        let entry = json!({ "path": "m", "kind": "model", "format": "f", "extra": true });
        write_outputs(
            dir.path(),
            &json!({ "protocol_version": 1, "status": "produced", "outputs": { "model": entry } }),
        );
        assert!(runtime_msg(read_outputs(dir.path())).contains("unknown field"));
    }

    #[test]
    fn train_external_contract_v1_read_outputs_missing_or_malformed() {
        let dir = tempfile::tempdir().unwrap();
        assert!(runtime_msg(read_outputs(dir.path())).contains("not written"));
        fs::write(dir.path().join(OUTPUTS_FILE), "{").unwrap();
        assert!(runtime_msg(read_outputs(dir.path())).contains("invalid JSON"));
    }

    #[test]
    fn train_external_contract_v1_read_outputs_skipped_into_outcome() {
        let dir = tempfile::tempdir().unwrap();
        write_outputs(
            dir.path(),
            &json!({ "protocol_version": 1, "status": "skipped", "skip_reason": "no new data" }),
        );
        let out = read_outputs(dir.path()).unwrap();
        assert_eq!(out.status, OutputStatus::Skipped);
        match out.into_outcome(dir.path(), Some("onnx")).unwrap() {
            BlockOutcome::Skipped { reason, outputs } => {
                assert_eq!(reason, "no new data");
                assert!(outputs.iter().next().is_none());
            }
            other => panic!("expected Skipped, got {other:?}"),
        }
    }

    #[test]
    fn train_external_contract_v1_read_outputs_produced_into_outcome() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("out")).unwrap();
        fs::write(dir.path().join("out/model.onnx"), b"m").unwrap();
        fs::write(dir.path().join("metrics.json"), b"{}").unwrap();
        write_outputs(
            dir.path(),
            &json!({ "protocol_version": 1, "status": "produced", "outputs": {
                "model": { "path": "out", "kind": "model", "format": "onnx",
                           "metadata": { "epochs": 3 } },
                "other": { "path": "out/model.onnx", "kind": "model", "format": "onnx",
                           "flavor": "custom" },
                "metrics": { "path": "metrics.json", "kind": "metrics", "format": "json" },
            }}),
        );
        let BlockOutcome::Produced(drafts) = read_outputs(dir.path())
            .unwrap()
            .into_outcome(dir.path(), Some("onnx"))
            .unwrap()
        else {
            panic!("expected Produced");
        };
        let model = drafts.get("model").unwrap();
        assert_eq!(model.manifest.flavor.as_deref(), Some("onnx"));
        assert_eq!(model.manifest.metadata["epochs"], json!(3));
        assert_eq!(model.payload, DraftPayload::Dir(dir.path().join("out")));
        let other = drafts.get("other").unwrap();
        assert_eq!(other.manifest.flavor.as_deref(), Some("custom"));
        assert_eq!(
            other.payload,
            DraftPayload::File(dir.path().join("out/model.onnx"))
        );
        // The configured flavor only applies to model outputs.
        assert_eq!(drafts.get("metrics").unwrap().manifest.flavor, None);
    }

    #[test]
    fn train_external_security_output_payload_rejects_traversal_and_absolute() {
        let dir = tempfile::tempdir().unwrap();
        for bad in ["../x", "a/../b", "/etc/passwd", "./x", "", "a\\b", "C:/x"] {
            assert!(
                runtime_msg(output_payload(dir.path(), bad)).contains("output path"),
                "{bad:?} accepted"
            );
        }
    }

    #[test]
    fn train_external_security_output_payload_missing() {
        let dir = tempfile::tempdir().unwrap();
        runtime_msg(output_payload(dir.path(), "nope"));
    }

    #[cfg(unix)]
    #[test]
    fn train_external_security_output_payload_rejects_symlinks() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("secret"), b"s").unwrap();
        let w = dir.path();
        fs::create_dir(w.join("real")).unwrap();
        fs::write(w.join("real/f"), b"f").unwrap();
        symlink(outside.path().join("secret"), w.join("file_link")).unwrap();
        symlink(outside.path(), w.join("dir_link")).unwrap();
        symlink(w.join("real"), w.join("inner_link")).unwrap();
        for bad in ["file_link", "dir_link", "dir_link/secret", "inner_link/f"] {
            assert!(
                runtime_msg(output_payload(w, bad)).contains("is a symlink"),
                "{bad:?} accepted"
            );
        }
        assert_eq!(
            output_payload(w, "real/f").unwrap(),
            DraftPayload::File(w.join("real/f"))
        );
        assert_eq!(
            output_payload(w, "real").unwrap(),
            DraftPayload::Dir(w.join("real"))
        );
    }

    #[cfg(unix)]
    #[test]
    fn train_external_security_output_payload_rejects_special_files() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("sock");
        let _listener = std::os::unix::net::UnixListener::bind(&sock).unwrap();
        assert!(runtime_msg(output_payload(dir.path(), "sock")).contains("not a regular file"));
    }

    #[cfg(unix)]
    #[test]
    fn train_external_security_read_outputs_rejects_symlinked_file() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        write_outputs(
            outside.path(),
            &json!({ "protocol_version": 1, "status": "produced" }),
        );
        std::os::unix::fs::symlink(
            outside.path().join(OUTPUTS_FILE),
            dir.path().join(OUTPUTS_FILE),
        )
        .unwrap();
        assert!(runtime_msg(read_outputs(dir.path())).contains("regular file"));
    }
}
