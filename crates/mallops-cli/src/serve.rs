//! `mallops serve`: HTTP inference over a `package` artifact (issue #17).

use std::path::Path;
use std::time::Duration;

use anyhow::Context as _;
use mallops_core::config::PipelineConfig;
use mallops_core::{ArtifactId, EnvSecretProvider, RelPath, SecretProvider as _};
use mallops_std::predictor::{PredictorRegistry, register_std_predictors};
use mallops_std::serve::{ServeOptions, Server};

use crate::run::{LATEST_PATH, base_dir, open_existing_store};

const NO_RUN: &str = "no successful run: run `mallops run` first";

/// Flags of `mallops serve`.
#[derive(clap::Args, Debug)]
pub(crate) struct Args {
    /// Package artifact id (default: the `serve.package` output of the latest successful run).
    package_id: Option<String>,
    /// Bind host (default: `serve.host`, then 127.0.0.1).
    #[arg(long)]
    host: Option<String>,
    /// Bind port, 0 = OS-assigned (default: `serve.port`, then 8080).
    #[arg(long)]
    port: Option<u16>,
    /// Allow model flavors that execute code from the package (e.g. `sklearn-pickle`).
    #[arg(long)]
    trust_unsafe_flavors: bool,
    /// Largest accepted request body in bytes (default 10485760).
    #[arg(long)]
    max_body_bytes: Option<usize>,
    /// Per-request timeout in seconds (default 30).
    #[arg(long, value_name = "SECS")]
    timeout: Option<u64>,
    /// Concurrent `/predict` requests (default 64).
    #[arg(long)]
    max_concurrency: Option<usize>,
    /// Require `Authorization: Bearer <token>`, token read from `MALLOPS_SECRET_<NAME>`.
    #[arg(long, value_name = "NAME")]
    auth_token_secret: Option<String>,
}

/// Which package to serve.
#[derive(Debug, PartialEq)]
enum Source {
    /// Given on the command line: `SIGHUP` is ignored.
    Explicit,
    /// Output `reference` (`<block>.<port>`) of the latest run of this pipeline.
    Latest { pipeline: String, reference: String },
}

/// Pure resolution of flags, `serve:` and defaults (flag > `serve:` > default) into the server
/// options (without the auth token); `latest` is the content of `runs/latest.json`, if any.
fn resolve(
    args: &Args,
    cfg: &PipelineConfig,
    latest: Option<&[u8]>,
) -> anyhow::Result<(ServeOptions, Source, ArtifactId)> {
    let mut o = ServeOptions::default();
    let from_yaml = cfg.serve.as_ref();
    let (source, id) = match &args.package_id {
        Some(s) => {
            let id = s
                .parse()
                .with_context(|| format!("invalid artifact id `{s}`"))?;
            (Source::Explicit, id)
        }
        None => {
            let serve = from_yaml.context(
                "no `serve:` section in the pipeline file (add one or pass a PACKAGE_ID)",
            )?;
            let id = latest_package(&cfg.name, &serve.package, latest)?;
            let source = Source::Latest {
                pipeline: cfg.name.clone(),
                reference: serve.package.clone(),
            };
            (source, id)
        }
    };
    if let Some(host) = args
        .host
        .clone()
        .or_else(|| from_yaml.map(|s| s.host.clone()))
    {
        o.host = host;
    }
    if let Some(port) = args.port.or(from_yaml.map(|s| s.port)) {
        o.port = port;
    }
    if let Some(v) = args.max_body_bytes {
        o.max_body_bytes = v;
    }
    if let Some(v) = args.timeout {
        o.timeout = Duration::from_secs(v);
    }
    if let Some(v) = args.max_concurrency {
        o.max_concurrency = v;
    }
    o.trust_unsafe_flavors = args.trust_unsafe_flavors;
    Ok((o, source, id))
}

/// Reads output `reference` of `latest` (`runs/latest.json`) if it belongs to pipeline `name`.
fn latest_package(
    name: &str,
    reference: &str,
    latest: Option<&[u8]>,
) -> anyhow::Result<ArtifactId> {
    let bytes = latest.context(NO_RUN)?;
    let doc: serde_json::Value =
        serde_json::from_slice(bytes).context("cannot parse runs/latest.json")?;
    if doc["pipeline"] != name {
        anyhow::bail!(NO_RUN);
    }
    let (block, port) = reference
        .split_once('.')
        .with_context(|| format!("invalid serve.package `{reference}`"))?;
    let id = doc["outputs"][block][port]
        .as_str()
        .with_context(|| format!("runs/latest.json has no output `{reference}`"))?;
    id.parse()
        .with_context(|| format!("invalid artifact id `{id}` in runs/latest.json"))
}

/// `SIGHUP` stream; on non-Unix it never fires.
struct Hangup(#[cfg(unix)] tokio::signal::unix::Signal);

impl Hangup {
    fn install() -> std::io::Result<Self> {
        #[cfg(unix)]
        return tokio::signal::unix::signal(tokio::signal::unix::SignalKind::hangup()).map(Self);
        #[cfg(not(unix))]
        Ok(Self())
    }

    async fn recv(&mut self) -> Option<()> {
        #[cfg(unix)]
        return self.0.recv().await;
        #[cfg(not(unix))]
        std::future::pending().await
    }
}

/// Resolves on Ctrl-C or (Unix) SIGTERM; both handlers are installed when this is called.
fn shutdown_signal() -> std::io::Result<impl Future<Output = ()> + Send + 'static> {
    #[cfg(unix)]
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    Ok(async move {
        #[cfg(unix)]
        let term = term.recv();
        #[cfg(not(unix))]
        let term = std::future::pending::<Option<()>>();
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term => {}
        }
    })
}

/// Runs the server until Ctrl-C / SIGTERM; `SIGHUP` reloads `runs/latest.json` (Unix).
pub(crate) fn serve(file: &Path, args: &Args) -> anyhow::Result<()> {
    let pipeline = crate::load(file)?;
    let cfg = pipeline.config();
    if cfg.store.kind == "memory" {
        anyhow::bail!("serve needs a persistent store");
    }
    let store = open_existing_store(base_dir(file), cfg)?;
    let rt = tokio::runtime::Runtime::new().context("cannot start tokio runtime")?;
    rt.block_on(async {
        // Installed before the (slow) load: an early SIGHUP must not kill the process.
        let mut hangup = Hangup::install().context("cannot install the SIGHUP handler")?;
        let shutdown = shutdown_signal().context("cannot install signal handlers")?;

        let latest_key = RelPath::parse(LATEST_PATH)?;
        let latest = store.get_meta(&latest_key).await?;
        let (mut opts, source, id) = resolve(args, cfg, latest.as_deref())?;
        opts.auth_token = args
            .auth_token_secret
            .as_deref()
            .map(|n| EnvSecretProvider.get(n))
            .transpose()?;
        let (host, port) = (opts.host.clone(), opts.port);

        let mut predictors = PredictorRegistry::new();
        register_std_predictors(&mut predictors)?;
        let server = Server::load(&*store, &id, &predictors, opts).await?;
        let handle = server.handle();
        let listener = tokio::net::TcpListener::bind((host.as_str(), port))
            .await
            .with_context(|| format!("cannot bind {host}:{port}"))?;
        println!("serving {id} on http://{}", listener.local_addr()?);

        let mut task = tokio::spawn(server.run(listener, shutdown));
        let res = loop {
            tokio::select! {
                res = &mut task => break res,
                Some(()) = hangup.recv() => {
                    // The reload runs inside this branch, so it is never cut off mid-swap.
                    let Source::Latest { pipeline, reference } = &source else {
                        eprintln!("warning: SIGHUP ignored: package id given on the command line");
                        continue;
                    };
                    let reloaded = async {
                        let latest = store.get_meta(&latest_key).await?;
                        let id = latest_package(pipeline, reference, latest.as_deref())?;
                        // `swap` logs its own failures.
                        let _ = handle.swap(&*store, &id, &predictors).await;
                        Ok::<_, anyhow::Error>(())
                    };
                    if let Err(e) = reloaded.await {
                        eprintln!("warning: SIGHUP reload failed: {e:#}");
                    }
                }
            }
        };
        drop(handle);
        res.context("server task failed")??;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser as _;

    const ID_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const ID_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn cfg(serve: &str) -> PipelineConfig {
        let yaml = format!("version: 1\nname: p\nblocks: {{}}\n{serve}");
        PipelineConfig::from_yaml_str(&yaml, "t.yaml", &|_| None).unwrap()
    }

    fn args(extra: &[&str]) -> Args {
        let mut v = vec!["mallops", "serve"];
        v.extend_from_slice(extra);
        match crate::Cli::try_parse_from(v).unwrap().cmd {
            crate::Cmd::Serve(a) => a,
            _ => unreachable!(),
        }
    }

    fn latest(pipeline: &str) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "v": 1, "pipeline": pipeline, "run": "r",
            "outputs": { "package": { "package": ID_A } },
        }))
        .unwrap()
    }

    #[test]
    fn cli_serve_flags_and_config() {
        let yaml = cfg("serve: { package: package.package, host: 0.0.0.0, port: 9000 }\n");
        let lj = latest("p");

        // serve: beats defaults; the package comes from latest.json.
        let (o, source, id) = resolve(&args(&[]), &yaml, Some(&lj)).unwrap();
        assert_eq!((o.host.as_str(), o.port), ("0.0.0.0", 9000));
        assert_eq!(id.to_string(), ID_A);
        assert_eq!(
            source,
            Source::Latest {
                pipeline: "p".into(),
                reference: "package.package".into()
            }
        );
        let d = ServeOptions::default();
        assert_eq!(o.max_body_bytes, d.max_body_bytes);
        assert_eq!(o.timeout, d.timeout);
        assert_eq!(o.max_concurrency, d.max_concurrency);
        assert!(!o.trust_unsafe_flavors);

        // flags beat serve:; an explicit id beats latest.json and is not reloadable.
        let a = args(&[
            ID_B,
            "--host",
            "127.0.0.2",
            "--port",
            "0",
            "--trust-unsafe-flavors",
            "--max-body-bytes",
            "5",
            "--timeout",
            "7",
            "--max-concurrency",
            "2",
        ]);
        let (o, source, id) = resolve(&a, &yaml, Some(&lj)).unwrap();
        assert_eq!((o.host.as_str(), o.port), ("127.0.0.2", 0));
        assert_eq!(id.to_string(), ID_B);
        assert_eq!(source, Source::Explicit);
        assert_eq!(
            (o.max_body_bytes, o.timeout, o.max_concurrency),
            (5, Duration::from_secs(7), 2)
        );
        assert!(o.trust_unsafe_flavors);

        // an explicit id needs neither serve: nor latest.json; defaults apply.
        let (o, ..) = resolve(&args(&[ID_B]), &cfg(""), None).unwrap();
        assert_eq!((o.host.as_str(), o.port), ("127.0.0.1", 8080));

        // failures
        let no_run = format!("{:#}", resolve(&args(&[]), &yaml, None).unwrap_err());
        assert_eq!(no_run, NO_RUN);
        let other = latest("other");
        let e = resolve(&args(&[]), &yaml, Some(&other)).unwrap_err();
        assert_eq!(format!("{e:#}"), NO_RUN);
        let e = resolve(&args(&[]), &cfg(""), Some(&lj)).unwrap_err();
        assert!(e.to_string().contains("serve:"), "{e}");
        let e = resolve(&args(&[]), &cfg("serve: { package: x.y }\n"), Some(&lj)).unwrap_err();
        assert!(e.to_string().contains("`x.y`"), "{e}");
        assert!(resolve(&args(&["nothex"]), &yaml, None).is_err());

        // a `serve` entry placed under `blocks:` is rejected by the loader `serve` uses.
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("mallops.yaml");
        std::fs::write(
            &file,
            "version: 1\nname: p\nblocks:\n  serve: { package: package.package, port: 8080 }\n",
        )
        .unwrap();
        let e = crate::load(&file).unwrap_err().to_string();
        assert!(e.contains("blocks.serve: unknown field `package`"), "{e}");
    }
}
