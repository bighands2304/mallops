//! `std/train.external`: runs any external command as a training step through the versioned
//! file contract v1 (see [`contract`]), without Rust bindings (SPEC FR-TRN-3, ADR-0003).

pub(crate) mod contract;
pub(crate) mod process;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use mallops_core::{
    ArtifactKind, Block, BlockOutcome, BlockSpec, BoxFuture, CachePolicy, Context, Digest, Error,
    Inputs, Port, ResolvedPorts, Stage,
};
use serde::Deserialize;

use contract::{CONFIG_FILE, ConfigFile, INPUTS_FILE, InputEntry, InputsFile, PROTOCOL_VERSION};
use process::{ChildSpec, run_child};

/// Default required input port.
pub(crate) const TRAIN: &str = "train";
/// Default optional input port.
pub(crate) const VAL: &str = "val";
/// Default output port.
pub(crate) const MODEL: &str = "model";
/// Default `log_cap_bytes` (per stream).
pub(crate) const DEFAULT_LOG_CAP: usize = 1024 * 1024;

/// `with:` of `std/train.external`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TrainExternalConfig {
    /// argv; elements starting with `./` or `../` are made absolute against mallops' cwd.
    pub(crate) command: Vec<String>,
    #[serde(default)]
    pub(crate) flavor: Option<String>,
    /// Free-form object written to `config.json`.
    #[serde(default)]
    pub(crate) params: Option<serde_json::Value>,
    /// Host environment variable names passed to the child.
    #[serde(default)]
    pub(crate) env: Vec<String>,
    /// Secret names resolved via `ctx.secret` and passed as env vars of the same name.
    #[serde(default)]
    pub(crate) secrets: Vec<String>,
    #[serde(default)]
    pub(crate) timeout_secs: Option<u64>,
    #[serde(default)]
    pub(crate) log_cap_bytes: Option<usize>,
    /// Files (relative to mallops' cwd) whose path+content make the block cacheable.
    #[serde(default)]
    pub(crate) watch_files: Vec<PathBuf>,
    /// Input ports (name → kind); replaces the defaults `train` + optional `val`.
    #[serde(default)]
    pub(crate) inputs: Option<BTreeMap<String, ArtifactKind>>,
    /// Output ports (name → kind); replaces the default `model`.
    #[serde(default)]
    pub(crate) outputs: Option<BTreeMap<String, ArtifactKind>>,
}

impl TrainExternalConfig {
    /// Deserializes and checks `with:` (D13-9); every error is an [`Error::Config`].
    pub(crate) fn parse(config: &serde_json::Value) -> Result<Self, Error> {
        let cfg = Self::deserialize(config).map_err(|e| Error::Config(e.to_string()))?;
        cfg.check()?;
        Ok(cfg)
    }

    fn check(&self) -> Result<(), Error> {
        let err = |msg: String| Err(Error::Config(msg));
        let secret_ref = |s: &str| s.contains("${secret:");
        let no_ref = |field: &str| {
            Error::Config(format!(
                "{field}: `${{secret:...}}` is not resolved here, use `secrets:` to pass a secret \
                 as an environment variable"
            ))
        };
        match self.command.first() {
            None => return err("command must not be empty".into()),
            Some(program) if program.is_empty() => {
                return err("command[0] must not be empty".into());
            }
            _ => {}
        }
        for (i, arg) in self.command.iter().enumerate() {
            if arg.contains('\0') {
                return err(format!("command[{i}] contains a NUL byte"));
            }
            if secret_ref(arg) {
                return Err(no_ref(&format!("command[{i}]")));
            }
        }
        if let Some(params) = &self.params {
            if !params.is_object() {
                return err("params must be an object".into());
            }
            if json_strings_any(params, &secret_ref) {
                return Err(no_ref("params"));
            }
        }
        let mut seen = BTreeSet::new();
        for (field, names) in [("env", &self.env), ("secrets", &self.secrets)] {
            for name in names {
                if secret_ref(name) {
                    return Err(no_ref(field));
                }
                if name.contains('\0') {
                    return err(format!("{field}: `{name}` contains a NUL byte"));
                }
                if !is_env_name(name) {
                    return err(format!(
                        "{field}: `{name}` is not a valid variable name ([A-Za-z_][A-Za-z0-9_]*)"
                    ));
                }
                let upper = name.to_ascii_uppercase();
                if upper == "PATH" || upper == "SYSTEMROOT" || upper.starts_with("MALLOPS_") {
                    return err(format!("{field}: `{name}` is reserved and set by mallops"));
                }
                if !seen.insert(upper) {
                    return err(format!(
                        "{field}: `{name}` is listed more than once in env/secrets"
                    ));
                }
            }
        }
        if self.timeout_secs == Some(0) {
            return err("timeout_secs must be greater than 0".into());
        }
        if self.log_cap_bytes == Some(0) {
            return err("log_cap_bytes must be greater than 0".into());
        }
        for (field, ports) in [("inputs", &self.inputs), ("outputs", &self.outputs)] {
            for name in ports.iter().flat_map(BTreeMap::keys) {
                if !is_port_name(name) {
                    return err(format!("{field}: port `{name}` must match [a-z][a-z0-9_]*"));
                }
            }
        }
        Ok(())
    }

    /// Input and output ports: `inputs`/`outputs` replace the defaults; `flavor` is set on
    /// every `model` port.
    fn ports(&self, spec: &BlockSpec) -> ResolvedPorts {
        let custom = |ports: &BTreeMap<String, ArtifactKind>| {
            ports
                .iter()
                .map(|(name, kind)| Port::new(name.clone(), *kind))
                .collect::<Vec<_>>()
        };
        let flavored = |ports: Vec<Port>| {
            ports
                .into_iter()
                .map(|p| match &self.flavor {
                    Some(f) if p.kind == ArtifactKind::Model => p.with_flavor(f.clone()),
                    _ => p,
                })
                .collect()
        };
        ResolvedPorts::new(
            flavored(
                self.inputs
                    .as_ref()
                    .map_or_else(|| spec.inputs.clone(), custom),
            ),
            flavored(
                self.outputs
                    .as_ref()
                    .map_or_else(|| spec.outputs.clone(), custom),
            ),
        )
    }

    /// argv with `./` and `../` arguments made absolute against `pipeline_dir` (D13-8), rebuilt
    /// from their components so the separators are native on every OS.
    fn argv(&self, pipeline_dir: &Path) -> Result<Vec<String>, Error> {
        self.command
            .iter()
            .map(|arg| {
                if arg.starts_with("./") || arg.starts_with("../") {
                    let rel = Path::new(arg)
                        .components()
                        .filter(|c| *c != Component::CurDir);
                    path_string(&pipeline_dir.join(rel.collect::<PathBuf>()))
                } else {
                    Ok(arg.clone())
                }
            })
            .collect()
    }

    /// The child's environment without secrets (D13-4): `PATH` (+ `SYSTEMROOT` on Windows) and
    /// `MALLOPS_PIPELINE_DIR`, then each `env` name found by `lookup` (unset names are skipped).
    fn child_env(
        &self,
        pipeline_dir: &Path,
        lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Vec<(String, String)>, Error> {
        let fixed: &[&str] = if cfg!(windows) {
            &["PATH", "SYSTEMROOT"]
        } else {
            &["PATH"]
        };
        let mut env: Vec<(String, String)> = fixed
            .iter()
            .copied()
            .chain(self.env.iter().map(String::as_str))
            .filter_map(|name| lookup(name).map(|v| (name.to_owned(), v)))
            .collect();
        env.push(("MALLOPS_PIPELINE_DIR".into(), path_string(pipeline_dir)?));
        Ok(env)
    }
}

/// `[A-Za-z_][A-Za-z0-9_]*`
fn is_env_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// `[a-z][a-z0-9_]*`
fn is_port_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars.next().is_some_and(|c| c.is_ascii_lowercase())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

/// Whether any string (key or value) inside `value` satisfies `pred`.
fn json_strings_any(value: &serde_json::Value, pred: &dyn Fn(&str) -> bool) -> bool {
    match value {
        serde_json::Value::String(s) => pred(s),
        serde_json::Value::Array(items) => items.iter().any(|v| json_strings_any(v, pred)),
        serde_json::Value::Object(map) => map
            .iter()
            .any(|(k, v)| pred(k) || json_strings_any(v, pred)),
        _ => false,
    }
}

fn path_string(path: &Path) -> Result<String, Error> {
    path.to_str()
        .map(str::to_owned)
        .ok_or_else(|| Error::Runtime(format!("path is not UTF-8: {}", path.display())))
}

fn io_err(path: &Path) -> impl Fn(std::io::Error) -> Error + '_ {
    move |e| Error::Runtime(format!("{}: {e}", path.display()))
}

/// BLAKE3 of each watched file's path and content, in path order (D13-10).
fn hash_watch_files(files: &[PathBuf]) -> Result<Digest, Error> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"mallops/train.external/watch_files/v1");
    for path in files.iter().collect::<BTreeSet<_>>() {
        let meta = std::fs::metadata(path).map_err(io_err(path))?;
        if !meta.is_file() {
            return Err(Error::Runtime(format!(
                "watch_files: {} is not a regular file",
                path.display()
            )));
        }
        let bytes = std::fs::read(path).map_err(io_err(path))?;
        let name = path.as_os_str().as_encoded_bytes();
        hasher.update(&(name.len() as u64).to_le_bytes());
        hasher.update(name);
        hasher.update(&(bytes.len() as u64).to_le_bytes());
        hasher.update(&bytes);
    }
    Ok(Digest::from_bytes(*hasher.finalize().as_bytes()))
}

/// Sets or clears the read-only flag of every file under `dir`; directories stay writable.
fn set_files_readonly(dir: &Path, readonly: bool) -> Result<(), Error> {
    for entry in std::fs::read_dir(dir).map_err(io_err(dir))? {
        let path = entry.map_err(io_err(dir))?.path();
        let meta = std::fs::symlink_metadata(&path).map_err(io_err(&path))?;
        if meta.is_dir() {
            set_files_readonly(&path, readonly)?;
        } else if meta.is_file() {
            let mut perms = meta.permissions();
            // Clearing it only touches our own copies, inside the private (0o700) workdir.
            perms.set_readonly(readonly);
            std::fs::set_permissions(&path, perms).map_err(io_err(&path))?;
        }
    }
    Ok(())
}

/// Makes the files under `inputs/` writable again on drop, whatever the outcome of the run:
/// Windows cannot delete read-only files, so the run workdir cleanup would fail.
struct WritableOnDrop(PathBuf);

impl Drop for WritableOnDrop {
    fn drop(&mut self) {
        let _ = set_files_readonly(&self.0, false);
    }
}

/// Last `max` bytes of `s` (on a char boundary).
fn tail(s: &str, max: usize) -> &str {
    let mut start = s.len().saturating_sub(max);
    while !s.is_char_boundary(start) {
        start += 1;
    }
    &s[start..]
}

/// Bytes of stderr quoted in the error of a failed command.
const STDERR_TAIL: usize = 4096;

/// The `std/train.external` block.
pub(crate) struct TrainExternal(BlockSpec);

impl TrainExternal {
    pub(crate) fn new() -> Self {
        Self(
            BlockSpec::new("std/train.external", "0.1.0", Stage::Train)
                .with_input(Port::new(TRAIN, ArtifactKind::Features))
                .with_input(Port::new(VAL, ArtifactKind::Features).optional())
                .with_output(Port::new(MODEL, ArtifactKind::Model)),
        )
    }
}

impl Block for TrainExternal {
    fn spec(&self) -> &BlockSpec {
        &self.0
    }

    fn validate_config(&self, config: &serde_json::Value) -> Result<(), Error> {
        TrainExternalConfig::parse(config).map(|_| ())
    }

    fn resolve_ports(&self, config: &serde_json::Value) -> Result<ResolvedPorts, Error> {
        Ok(TrainExternalConfig::parse(config)?.ports(&self.0))
    }

    fn cache_policy<'a>(&'a self, ctx: &'a Context) -> BoxFuture<'a, Result<CachePolicy, Error>> {
        Box::pin(async move {
            let cfg = TrainExternalConfig::parse(ctx.config())?;
            if cfg.watch_files.is_empty() {
                return Ok(CachePolicy::Uncacheable);
            }
            let digest = tokio::task::spawn_blocking(move || hash_watch_files(&cfg.watch_files))
                .await
                .map_err(|e| Error::Runtime(format!("blocking task failed: {e}")))??;
            Ok(CachePolicy::Deterministic {
                extra_hash: Some(digest),
            })
        })
    }

    fn run<'a>(
        &'a self,
        ctx: &'a Context,
        inputs: Inputs,
    ) -> BoxFuture<'a, Result<BlockOutcome, Error>> {
        Box::pin(async move {
            if ctx.cancel().is_cancelled() {
                return Err(Error::Cancelled);
            }
            let cfg: TrainExternalConfig = ctx.config_as()?;
            cfg.check()?;
            // ponytail: sync fs calls on the async thread, a handful of small files and dirs.
            let workdir = ctx.workdir();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(workdir, std::fs::Permissions::from_mode(0o700))
                    .map_err(io_err(workdir))?;
            }
            // Already absolute; no `canonicalize`, its `\\?\` prefix on Windows disables `/` and
            // `..` handling in the paths joined to it.
            let cwd = std::env::current_dir()
                .map_err(|e| Error::Runtime(format!("cannot resolve the pipeline dir: {e}")))?;

            let inputs_dir = workdir.join("inputs");
            std::fs::create_dir_all(&inputs_dir).map_err(io_err(&inputs_dir))?;
            let _writable = WritableOnDrop(inputs_dir.clone());
            let mut entries = BTreeMap::new();
            for (port, artifact) in inputs.iter() {
                // Library callers bypass the pipeline validation: never join an unchecked name.
                if !is_port_name(port) {
                    return Err(Error::Runtime(format!("invalid input port name `{port}`")));
                }
                let rel = format!("inputs/{port}");
                let dest = inputs_dir.join(port);
                ctx.store().fetch(&artifact.id(), &dest).await?;
                set_files_readonly(&dest, true)?;
                entries.insert(
                    port.to_owned(),
                    InputEntry {
                        id: artifact.id().to_hex(),
                        path: rel,
                        manifest: artifact.manifest().clone(),
                    },
                );
            }
            let inputs_file = InputsFile {
                protocol_version: PROTOCOL_VERSION,
                inputs: entries,
            };
            let config_file = ConfigFile {
                protocol_version: PROTOCOL_VERSION,
                flavor: cfg.flavor.clone(),
                params: cfg.params.clone().unwrap_or_else(|| serde_json::json!({})),
                run_id: ctx.run_id().to_string(),
            };
            write_json(&workdir.join(INPUTS_FILE), &inputs_file)?;
            write_json(&workdir.join(CONFIG_FILE), &config_file)?;

            let secret_env = cfg
                .secrets
                .iter()
                .map(|name| Ok((name.clone(), ctx.secret(name)?)))
                .collect::<Result<Vec<_>, Error>>()?;
            let spec = ChildSpec {
                argv: cfg.argv(&cwd)?,
                cwd: workdir.to_path_buf(),
                env: cfg.child_env(&cwd, &|name| std::env::var(name).ok())?,
                secret_env,
                timeout: cfg.timeout_secs.map(Duration::from_secs),
                log_cap: cfg.log_cap_bytes.unwrap_or(DEFAULT_LOG_CAP),
            };
            let program = spec.argv[0].clone();
            let out = run_child(spec, ctx.cancel()).await?;
            for (stream, text, truncated) in [
                ("stdout", &out.stdout, out.truncated[0]),
                ("stderr", &out.stderr, out.truncated[1]),
            ] {
                if !text.is_empty() {
                    tracing::info!(program = %program, stream, truncated, "{text}");
                }
            }
            if !out.status.success() {
                return Err(Error::Runtime(format!(
                    "`{program}` failed ({}); stderr tail:\n{}",
                    out.status,
                    tail(&out.stderr, STDERR_TAIL)
                )));
            }
            contract::read_outputs(workdir)?.into_outcome(workdir, cfg.flavor.as_deref())
        })
    }
}

fn write_json(path: &Path, value: &impl serde::Serialize) -> Result<(), Error> {
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|e| Error::Runtime(format!("{}: {e}", path.display())))?;
    std::fs::write(path, bytes).map_err(io_err(path))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use mallops_core::{
        Artifact, ArtifactStore, CancelSource, Manifest, MemorySecretProvider, MemoryStore,
        PayloadSource, RunId,
    };
    use serde_json::json;

    use super::*;

    fn config_err(config: serde_json::Value) -> String {
        match TrainExternal::new().validate_config(&config) {
            Err(Error::Config(m)) => m,
            other => panic!("expected Error::Config for {config}, got {other:?}"),
        }
    }

    fn with(extra: serde_json::Value) -> serde_json::Value {
        let mut cfg = json!({ "command": ["python", "./train.py"] });
        cfg.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        cfg
    }

    #[test]
    fn train_external_validate_config_accepts_full_config() {
        let cfg = with(json!({
            "flavor": "onnx", "params": { "lr": 0.1 }, "env": ["HOME", "CUDA_VISIBLE_DEVICES"],
            "secrets": ["HF_TOKEN"], "timeout_secs": 60, "log_cap_bytes": 10,
            "watch_files": ["train.py"], "inputs": { "data": "dataset" },
            "outputs": { "model": "model", "report_1": "metrics" },
        }));
        TrainExternal::new().validate_config(&cfg).unwrap();
    }

    #[test]
    fn train_external_validate_config_command() {
        assert!(config_err(json!({})).contains("command"));
        assert!(config_err(json!({ "command": [] })).contains("must not be empty"));
        assert!(config_err(json!({ "command": [""] })).contains("command[0]"));
        assert!(config_err(json!({ "command": ["a", "b\u{0}"] })).contains("NUL"));
        assert!(config_err(json!({ "command": ["a"], "bogus": 1 })).contains("unknown field"));
        // An empty argument after the program is legitimate.
        TrainExternal::new()
            .validate_config(&json!({ "command": ["a", ""] }))
            .unwrap();
    }

    #[test]
    fn train_external_validate_config_rejects_secret_refs() {
        for cfg in [
            json!({ "command": ["a", "--token=${secret:T}"] }),
            with(json!({ "params": { "nested": ["${secret:T}"] } })),
            with(json!({ "params": { "${secret:T}": 1 } })),
            with(json!({ "env": ["${secret:T}"] })),
        ] {
            assert!(config_err(cfg).contains("use `secrets:`"));
        }
    }

    #[test]
    fn train_external_validate_config_env_and_secret_names() {
        for bad in ["1A", "A-B", "", "A B"] {
            assert!(config_err(with(json!({ "env": [bad] }))).contains("valid variable name"));
            assert!(config_err(with(json!({ "secrets": [bad] }))).contains("valid variable name"));
        }
        assert!(config_err(with(json!({ "env": ["A\u{0}"] }))).contains("NUL"));
        for reserved in [
            "PATH",
            "Path",
            "SYSTEMROOT",
            "MALLOPS_PIPELINE_DIR",
            "MALLOPS_X",
        ] {
            assert!(config_err(with(json!({ "env": [reserved] }))).contains("reserved"));
            assert!(config_err(with(json!({ "secrets": [reserved] }))).contains("reserved"));
        }
        assert!(
            config_err(with(json!({ "env": ["TOKEN"], "secrets": ["TOKEN"] })))
                .contains("more than once")
        );
        assert!(config_err(with(json!({ "env": ["A", "A"] }))).contains("more than once"));
    }

    #[test]
    fn train_external_validate_config_ports_limits_and_params() {
        for bad in ["Train", "1x", "a-b", ""] {
            assert!(config_err(with(json!({ "inputs": { bad: "features" } }))).contains("port"));
            assert!(config_err(with(json!({ "outputs": { bad: "model" } }))).contains("port"));
        }
        assert!(
            config_err(with(json!({ "outputs": { "m": "nope" } }))).contains("unknown variant")
        );
        assert!(config_err(with(json!({ "timeout_secs": 0 }))).contains("timeout_secs"));
        assert!(config_err(with(json!({ "log_cap_bytes": 0 }))).contains("log_cap_bytes"));
        assert!(config_err(with(json!({ "params": [1] }))).contains("params must be an object"));
    }

    #[test]
    fn train_external_ports_and_cache_default_ports() {
        let block = TrainExternal::new();
        let ports = block.resolve_ports(&with(json!({}))).unwrap();
        assert_eq!(ports, ResolvedPorts::from(block.spec()));
        let train = &ports.inputs[0];
        assert_eq!(
            (train.name.as_ref(), train.kind, train.required),
            (TRAIN, ArtifactKind::Features, true)
        );
        let val = &ports.inputs[1];
        assert_eq!(
            (val.name.as_ref(), val.kind, val.required),
            (VAL, ArtifactKind::Features, false)
        );
        assert_eq!(ports.outputs[0].name, MODEL);
        assert!(ports.outputs[0].flavors.is_empty());

        let ports = block
            .resolve_ports(&with(json!({ "flavor": "onnx" })))
            .unwrap();
        assert_eq!(ports.outputs[0].flavors, ["onnx"]);
        assert!(ports.inputs.iter().all(|p| p.flavors.is_empty()));
    }

    #[test]
    fn train_external_ports_and_cache_override_ports() {
        let cfg = with(json!({
            "flavor": "xgb",
            "inputs": { "data": "dataset", "base": "model" },
            "outputs": { "booster": "model", "report": "metrics" },
        }));
        let ports = TrainExternal::new().resolve_ports(&cfg).unwrap();
        let summary = |ps: &[Port]| {
            ps.iter()
                .map(|p| (p.name.to_string(), p.kind, p.required, p.flavors.clone()))
                .collect::<Vec<_>>()
        };
        let xgb = vec![std::borrow::Cow::Borrowed("xgb")];
        assert_eq!(
            summary(&ports.inputs),
            [
                ("base".into(), ArtifactKind::Model, true, xgb.clone()),
                ("data".into(), ArtifactKind::Dataset, true, vec![]),
            ]
        );
        assert_eq!(
            summary(&ports.outputs),
            [
                ("booster".into(), ArtifactKind::Model, true, xgb),
                ("report".into(), ArtifactKind::Metrics, true, vec![]),
            ]
        );
    }

    fn ctx_with(config: serde_json::Value, workdir: &Path) -> Context {
        Context::new(
            RunId::new(),
            workdir.to_path_buf(),
            Arc::new(MemoryStore::new()),
            config,
            CancelSource::new().token(),
            Arc::new(MemorySecretProvider::default()),
        )
    }

    /// Runs the block with one stored input bound to `port` and a command that cannot spawn.
    async fn run_one_input(port: &str, workdir: &Path) -> Result<BlockOutcome, Error> {
        let store = Arc::new(MemoryStore::new());
        let manifest = Manifest::new(ArtifactKind::Features, "csv");
        let id = store
            .put(manifest.clone(), PayloadSource::Bytes(b"x,y\n"))
            .await
            .unwrap();
        let missing = workdir.join("missing-binary");
        let ctx = Context::new(
            RunId::new(),
            workdir.to_path_buf(),
            store,
            json!({ "command": [missing] }),
            CancelSource::new().token(),
            Arc::new(MemorySecretProvider::default()),
        );
        let inputs = Inputs::new().with(port, Artifact::new(id, manifest));
        TrainExternal::new().run(&ctx, inputs).await
    }

    #[tokio::test]
    async fn train_external_contract_v1_inputs_writable_after_run() {
        // Not unix-only: on Windows read-only inputs would make the cleanup fail.
        let dir = tempfile::tempdir().unwrap();
        let workdir = dir.path().join("w");
        std::fs::create_dir(&workdir).unwrap();
        let err = run_one_input(TRAIN, &workdir).await.unwrap_err();
        assert!(
            matches!(&err, Error::Runtime(m) if m.contains("failed to spawn")),
            "{err:?}"
        );
        assert!(workdir.join("inputs/train").exists());
        std::fs::remove_dir_all(&workdir).unwrap();
    }

    #[tokio::test]
    async fn train_external_security_rejects_bad_input_port_names() {
        for port in ["../x", "/abs"] {
            let dir = tempfile::tempdir().unwrap();
            let err = run_one_input(port, dir.path()).await.unwrap_err();
            assert!(
                matches!(&err, Error::Runtime(m) if m.contains("invalid input port name")),
                "{port}: {err:?}"
            );
        }
    }

    #[tokio::test]
    async fn train_external_ports_and_cache_uncacheable_by_default() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ctx_with(with(json!({})), dir.path());
        assert_eq!(
            TrainExternal::new().cache_policy(&ctx).await.unwrap(),
            CachePolicy::Uncacheable
        );
    }

    #[tokio::test]
    async fn train_external_ports_and_cache_watch_files() {
        // Absolute paths: the test must not depend on (or change) the process cwd.
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("train.py");
        let b = dir.path().join("lib.py");
        std::fs::write(&a, "v1").unwrap();
        std::fs::write(&b, "x").unwrap();
        let block = TrainExternal::new();
        let policy = |files: serde_json::Value| {
            let ctx = ctx_with(with(json!({ "watch_files": files })), dir.path());
            let block = &block;
            async move { block.cache_policy(&ctx).await }
        };
        let first = policy(json!([a, b])).await.unwrap();
        assert!(matches!(
            first,
            CachePolicy::Deterministic {
                extra_hash: Some(_)
            }
        ));
        // Order of watch_files does not matter.
        assert_eq!(policy(json!([b, a])).await.unwrap(), first);
        std::fs::write(&a, "v2").unwrap();
        assert_ne!(policy(json!([a, b])).await.unwrap(), first);
        // Missing file or directory is an error.
        assert!(policy(json!([dir.path().join("missing")])).await.is_err());
        assert!(policy(json!([dir.path()])).await.is_err());
    }

    #[test]
    fn train_external_argv_resolves_relative_paths() {
        let cfg = TrainExternalConfig::parse(&json!({
            "command": ["./a/b.py", "../x", "python", "data/x", "--out=./m"],
        }))
        .unwrap();
        let base = Path::new("/pipe/line");
        let argv = cfg.argv(base).unwrap();
        // Native separators only: `/` is not a separator in Windows `\\?\` paths.
        assert_eq!(argv[0], path_string(&base.join("a").join("b.py")).unwrap());
        assert_eq!(argv[1], path_string(&base.join("..").join("x")).unwrap());
        assert_eq!(&argv[2..], ["python", "data/x", "--out=./m"]);
    }

    #[test]
    fn train_external_child_env_from_lookup() {
        let cfg = TrainExternalConfig::parse(&with(json!({
            "env": ["HOME", "UNSET"], "secrets": ["TOKEN"],
        })))
        .unwrap();
        let lookup = |name: &str| match name {
            "PATH" => Some("/usr/bin".to_owned()),
            "HOME" => Some("/home/u".to_owned()),
            "TOKEN" | "OTHER" => Some("leak".to_owned()),
            _ => None,
        };
        let env = cfg.child_env(Path::new("/pipe"), &lookup).unwrap();
        let names: Vec<&str> = env.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(names, ["PATH", "HOME", "MALLOPS_PIPELINE_DIR"]);
        assert_eq!(env[0].1, "/usr/bin");
        assert_eq!(env[2].1, "/pipe");
    }

    #[test]
    fn train_external_stderr_tail_respects_char_boundaries() {
        assert_eq!(tail("abcdef", 3), "def");
        assert_eq!(tail("ab", 10), "ab");
        assert_eq!(tail("aé", 1), "");
    }

    #[cfg(unix)]
    #[test]
    fn train_external_security_inputs_files_readonly_dirs_writable() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub/f"), "x").unwrap();
        set_files_readonly(dir.path(), true).unwrap();
        let ro = |p: &str| {
            std::fs::metadata(dir.path().join(p))
                .unwrap()
                .permissions()
                .readonly()
        };
        assert!(ro("sub/f"));
        assert!(!ro("sub"));
        // Cleanup still works.
        std::fs::remove_dir_all(dir.path().join("sub")).unwrap();
    }
}
