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

enum StoreKind {
    Local,
    Memory,
}

fn parse_store_kind(kind: &str) -> anyhow::Result<StoreKind> {
    match kind {
        "local" => Ok(StoreKind::Local),
        "memory" => Ok(StoreKind::Memory),
        other => anyhow::bail!("unsupported store kind `{other}` (supported: `local`, `memory`)"),
    }
}

fn parse_lineage_sinks(lineage: &LineageConfig) -> anyhow::Result<usize> {
    match lineage {
        LineageConfig::Off => Ok(0),
        LineageConfig::Sinks(sinks) => {
            for sink in sinks {
                if sink.sink != "json" {
                    anyhow::bail!(
                        "unsupported lineage sink `{}` (supported: `json`)",
                        sink.sink
                    );
                }
            }
            Ok(sinks.len())
        }
        _ => anyhow::bail!("unsupported lineage configuration"),
    }
}

/// Verifies that the store kind and lineage sinks configured in `cfg` are supported by the CLI.
pub(crate) fn check_backends(cfg: &PipelineConfig) -> anyhow::Result<()> {
    parse_store_kind(&cfg.store.kind)?;
    parse_lineage_sinks(&cfg.lineage)?;
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
    match parse_store_kind(&cfg.store.kind)? {
        StoreKind::Local => Ok(Arc::new(LocalStore::new(base.join(&cfg.store.path))?)),
        StoreKind::Memory => Ok(Arc::new(MemoryStore::new())),
    }
}

fn build_lineage(
    cfg: &PipelineConfig,
    store: &Arc<dyn ArtifactStore>,
) -> anyhow::Result<Vec<Arc<dyn LineageSink>>> {
    let count = parse_lineage_sinks(&cfg.lineage)?;
    Ok((0..count)
        .map(|_| Arc::new(JsonSink::new(Arc::clone(store))) as Arc<dyn LineageSink>)
        .collect())
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
    let mut skipped = 0usize;
    let mut cancelled = 0usize;

    for node in pipeline.nodes() {
        let Some(ns) = summary.nodes.get(&node.id) else {
            continue;
        };
        match ns.status {
            Status::Succeeded if ns.cached => cached += 1,
            Status::Succeeded => executed += 1,
            Status::Failed => failed += 1,
            Status::Skipped => skipped += 1,
            Status::Cancelled => cancelled += 1,
            _ => {}
        }
        let status = status_label(ns.status);
        let cached_tag = if ns.cached { " (cached)" } else { "" };
        let ms = ns.duration.as_millis();
        let err_str = ns.error.as_ref().map(ToString::to_string);
        let detail: Vec<&str> = ns
            .reason
            .as_deref()
            .into_iter()
            .chain(err_str.as_deref())
            .collect();
        let suffix = if detail.is_empty() {
            String::new()
        } else {
            format!(": {}", detail.join(": "))
        };
        println!("{} {status}{cached_tag} {ms}ms{suffix}", node.id);
    }

    println!(
        "run {}: {executed} executed, {cached} cached, {failed} failed, {skipped} skipped, {cancelled} cancelled",
        summary.run_id
    );
}

fn latest_json(pipeline_name: &str, summary: &RunSummary) -> anyhow::Result<Vec<u8>> {
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
        "pipeline": pipeline_name,
        "run": summary.run_id.to_string(),
        "outputs": outputs,
    });
    serde_json::to_vec(&doc).context("cannot serialize runs/latest.json")
}

/// Executes the batch DAG of `file`, prints a summary and records `runs/latest.json` when every
/// node succeeds.
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

    let skipped = summary
        .nodes
        .values()
        .filter(|n| n.status == Status::Skipped)
        .count();
    if skipped > 0 {
        println!("latest.json not updated: {skipped} skipped");
        return Ok(());
    }

    let latest_key = RelPath::parse(LATEST_PATH)?;
    let bytes = latest_json(&cfg.name, &summary)?;
    rt.block_on(store.put_meta(&latest_key, &bytes))
        .context("all nodes succeeded but runs/latest.json could not be written")?;
    Ok(())
}

/// Verifies artifact `id` in the store configured by `file`.
pub(crate) fn verify(file: &Path, id: &str) -> anyhow::Result<()> {
    let artifact_id: ArtifactId = id
        .parse()
        .with_context(|| format!("invalid artifact id `{id}`"))?;
    let cfg = PipelineConfig::from_yaml_file(file, &|n| std::env::var(n).ok())?;
    let kind = parse_store_kind(&cfg.store.kind)?;
    parse_lineage_sinks(&cfg.lineage)?;

    let base = base_dir(file);
    if matches!(kind, StoreKind::Local) {
        let path = base.join(&cfg.store.path);
        if !path.exists() {
            anyhow::bail!("store not found: {}", path.display());
        }
    }

    let store = open_store(base, &cfg)?;
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
