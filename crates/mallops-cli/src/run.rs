//! `run` (with `runs/latest.json`) and `store verify` (task 10.2).

use std::collections::BTreeMap;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Arc;

use anyhow::Context as _;
use mallops_core::config::{LineageConfig, PipelineConfig};
use mallops_core::{
    ArtifactId, ArtifactStore, JsonSink, LineageSink, LocalStore, MemoryStore, RelPath, RunSummary,
    Runtime, RuntimeConfig, Status, ValidatedPipeline,
};

/// Store metadata key of the latest successful run pointer (SPEC §4.1, ARCHITECTURE §6.1).
const LATEST_PATH: &str = "runs/latest.json";

/// Schema version of `runs/latest.json`.
const LATEST_DOC_VERSION: u32 = 1;

/// Verifies that the store kind and lineage sinks configured in `cfg` are supported by the CLI.
pub(crate) fn check_backends(cfg: &PipelineConfig) -> anyhow::Result<()> {
    match cfg.store.kind.as_str() {
        "local" | "memory" => {}
        other => anyhow::bail!("unsupported store kind `{other}` (supported: `local`, `memory`)"),
    }
    match &cfg.lineage {
        LineageConfig::Off => {}
        LineageConfig::Sinks(sinks) => {
            for sink in sinks {
                if sink.sink != "json" {
                    anyhow::bail!(
                        "unsupported lineage sink `{}` (supported: `json`)",
                        sink.sink
                    );
                }
            }
        }
        _ => anyhow::bail!("unsupported lineage configuration"),
    }
    Ok(())
}

/// Directory containing `file`, defaulting to `.` when `file` has no parent component.
fn base_dir(file: &Path) -> &Path {
    match file.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    }
}

fn open_store(base: &Path, cfg: &PipelineConfig) -> anyhow::Result<Arc<dyn ArtifactStore>> {
    match cfg.store.kind.as_str() {
        "local" => Ok(Arc::new(LocalStore::new(base.join(&cfg.store.path))?)),
        "memory" => Ok(Arc::new(MemoryStore::new())),
        other => anyhow::bail!("unsupported store kind `{other}` (supported: `local`, `memory`)"),
    }
}

fn build_lineage(
    cfg: &PipelineConfig,
    store: &Arc<dyn ArtifactStore>,
) -> anyhow::Result<Vec<Arc<dyn LineageSink>>> {
    match &cfg.lineage {
        LineageConfig::Off => Ok(Vec::new()),
        LineageConfig::Sinks(sinks) => {
            let mut out: Vec<Arc<dyn LineageSink>> = Vec::with_capacity(sinks.len());
            for sink in sinks {
                match sink.sink.as_str() {
                    "json" => out.push(Arc::new(JsonSink::new(Arc::clone(store)))),
                    other => {
                        anyhow::bail!("unsupported lineage sink `{other}` (supported: `json`)")
                    }
                }
            }
            Ok(out)
        }
        _ => anyhow::bail!("unsupported lineage configuration"),
    }
}

/// Best-effort `git rev-parse HEAD` in `base`; only called by `run`.
fn git_commit(base: &Path) -> Option<String> {
    let out = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(base)
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let stdout = String::from_utf8(out.stdout).ok()?;
    let trimmed = stdout.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

fn status_label(status: Status) -> &'static str {
    match status {
        Status::Succeeded => "succeeded",
        Status::Failed => "failed",
        Status::Skipped => "skipped",
        Status::Cancelled => "cancelled",
        _ => "unknown",
    }
}

fn print_summary(pipeline: &ValidatedPipeline, summary: &RunSummary) {
    let mut executed = 0usize;
    let mut cached = 0usize;
    let mut failed = 0usize;

    for node in pipeline.nodes() {
        let Some(ns) = summary.nodes.get(&node.id) else {
            continue;
        };
        match ns.status {
            Status::Succeeded if ns.cached => cached += 1,
            Status::Succeeded => executed += 1,
            Status::Failed => failed += 1,
            _ => {}
        }
        let status = status_label(ns.status);
        let cached_tag = if ns.cached { " (cached)" } else { "" };
        let ms = ns.duration.as_millis();
        match (&ns.reason, &ns.error) {
            (Some(reason), Some(err)) => {
                println!("{} {status}{cached_tag} {ms}ms: {reason}: {err}", node.id);
            }
            (None, Some(err)) => {
                println!("{} {status}{cached_tag} {ms}ms: {err}", node.id);
            }
            (Some(reason), None) => {
                println!("{} {status}{cached_tag} {ms}ms: {reason}", node.id);
            }
            (None, None) => {
                println!("{} {status}{cached_tag} {ms}ms", node.id);
            }
        }
    }

    println!(
        "run {}: {executed} executed, {cached} cached, {failed} failed",
        summary.run_id
    );
}

fn latest_json(summary: &RunSummary) -> anyhow::Result<Vec<u8>> {
    let outputs: BTreeMap<&str, BTreeMap<&str, String>> = summary
        .nodes
        .iter()
        .map(|(node, ns)| {
            let ports = ns
                .outputs
                .iter()
                .map(|(port, id)| (port.as_str(), id.to_hex()))
                .collect();
            (node.as_str(), ports)
        })
        .collect();
    let doc = serde_json::json!({
        "v": LATEST_DOC_VERSION,
        "run": summary.run_id.to_string(),
        "outputs": outputs,
    });
    serde_json::to_vec(&doc).context("cannot serialize runs/latest.json")
}

/// Executes the batch DAG of `file`, prints a summary and records `runs/latest.json` on success.
pub(crate) fn run(file: &Path, jobs: Option<usize>, no_cache: bool) -> anyhow::Result<()> {
    let pipeline = crate::load(file)?;
    let cfg = pipeline.config();
    check_backends(cfg)?;

    let base = base_dir(file);
    let store = open_store(base, cfg)?;
    let lineage = build_lineage(cfg, &store)?;

    let mut rt_cfg = RuntimeConfig::new(Arc::clone(&store));
    if let Some(max_jobs) = jobs {
        rt_cfg.max_jobs = max_jobs;
    }
    rt_cfg.no_cache = no_cache;
    rt_cfg.lineage = lineage;
    rt_cfg.git_commit = git_commit(base);

    let rt = tokio::runtime::Runtime::new().context("cannot start tokio runtime")?;
    let summary = rt.block_on(Runtime::new(rt_cfg).run(&pipeline))?;

    print_summary(&pipeline, &summary);

    if !summary.is_success() {
        anyhow::bail!("run failed");
    }

    let latest_key = RelPath::parse(LATEST_PATH)?;
    let bytes = latest_json(&summary)?;
    rt.block_on(store.put_meta(&latest_key, &bytes))?;
    Ok(())
}

/// Verifies artifact `id` in the store configured by `file`.
pub(crate) fn verify(file: &Path, id: &str) -> anyhow::Result<()> {
    let artifact_id: ArtifactId = id
        .parse()
        .with_context(|| format!("invalid artifact id `{id}`"))?;
    let cfg = PipelineConfig::from_yaml_file(file, &|n| std::env::var(n).ok())?;
    check_backends(&cfg)?;

    let store = open_store(base_dir(file), &cfg)?;
    let rt = tokio::runtime::Runtime::new().context("cannot start tokio runtime")?;
    rt.block_on(store.verify(&artifact_id))?;
    println!("ok: {artifact_id}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_cfg(yaml: &str) -> PipelineConfig {
        PipelineConfig::from_yaml_str(yaml, "test.yaml", &|_| None).unwrap()
    }

    #[test]
    fn run_base_dir_handles_empty_and_nested_parents() {
        assert_eq!(base_dir(Path::new("mallops.yaml")), Path::new("."));
        assert_eq!(base_dir(Path::new("sub/mallops.yaml")), Path::new("sub"));
        assert_eq!(base_dir(Path::new("/a/b/mallops.yaml")), Path::new("/a/b"));
    }

    #[test]
    fn run_check_backends_accepts_supported_and_rejects_unknown() {
        let local_default = parse_cfg("version: 1\nname: p\nblocks: {}\n");
        assert!(check_backends(&local_default).is_ok());

        let mem_off =
            parse_cfg("version: 1\nname: p\nstore: { kind: memory }\nlineage: off\nblocks: {}\n");
        assert!(check_backends(&mem_off).is_ok());

        let bad_store = parse_cfg("version: 1\nname: p\nstore: { kind: s3 }\nblocks: {}\n");
        let err = check_backends(&bad_store).unwrap_err().to_string();
        assert!(err.contains("s3"), "{err}");

        let bad_sink = parse_cfg(
            "version: 1\nname: p\nlineage: [{ sink: json }, { sink: openlineage }]\nblocks: {}\n",
        );
        let err = check_backends(&bad_sink).unwrap_err().to_string();
        assert!(err.contains("openlineage"), "{err}");
    }
}
