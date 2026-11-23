//! `mallops` command-line interface over `mallops-core`.
//!
//! `validate` and `graph` never execute blocks or spawn processes (SPEC §4.1).

mod inspect;
mod noop;
mod run;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use clap::{Parser, Subcommand, ValueEnum};
use mallops_core::{BlockRegistry, ValidatedPipeline, config::PipelineConfig};

#[derive(Parser)]
#[command(name = "mallops", version, about = "Modular MLOps pipelines")]
struct Cli {
    /// Debug logging on stderr.
    #[arg(short, long, global = true)]
    verbose: bool,
    /// Pipeline definition.
    #[arg(
        short = 'f',
        long = "file",
        global = true,
        default_value = "mallops.yaml"
    )]
    file: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Scaffold a minimal runnable `mallops.yaml` in DIR (default: current directory).
    Init { dir: Option<PathBuf> },
    /// Parse and validate the pipeline without executing anything.
    Validate,
    /// Execute the batch DAG.
    Run {
        /// Maximum concurrent blocks (default: available parallelism).
        #[arg(long)]
        jobs: Option<usize>,
        /// Bypass the cache.
        #[arg(long)]
        no_cache: bool,
    },
    /// Print the validated DAG without executing anything.
    Graph {
        #[arg(long, value_enum, default_value_t = GraphFormat::Mermaid)]
        format: GraphFormat,
    },
    /// Registered blocks.
    Blocks {
        #[command(subcommand)]
        cmd: BlocksCmd,
    },
    /// Artifact store operations.
    Store {
        #[command(subcommand)]
        cmd: StoreCmd,
    },
}

#[derive(Subcommand)]
enum BlocksCmd {
    /// List registered blocks.
    List,
}

#[derive(Subcommand)]
enum StoreCmd {
    /// Verify the integrity of a stored artifact.
    Verify { id: String },
}

/// Output format of `mallops graph`.
#[derive(Clone, Copy, ValueEnum)]
pub(crate) enum GraphFormat {
    Mermaid,
    Dot,
}

/// Blocks available to the CLI. `mallops_std::register_std` joins in #11.
pub(crate) fn registry() -> BlockRegistry {
    let mut r = BlockRegistry::new();
    // ponytail: a fresh registry cannot hold a duplicate id.
    if let Err(e) = r.register(Arc::new(noop::Noop::new())) {
        unreachable!("core/noop registration failed: {e}");
    }
    r
}

/// Parses `file` (with `${env:…}` interpolation) and validates it against [`registry`].
pub(crate) fn load(file: &Path) -> anyhow::Result<ValidatedPipeline> {
    let config = PipelineConfig::from_yaml_file(file, &|n| std::env::var(n).ok())?;
    Ok(config.validate(&registry())?)
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_max_level(if cli.verbose {
            tracing_subscriber::filter::LevelFilter::DEBUG
        } else {
            tracing_subscriber::filter::LevelFilter::WARN
        })
        .init();
    let res = match cli.cmd {
        Cmd::Init { dir } => inspect::init(dir.as_deref().unwrap_or(Path::new("."))),
        Cmd::Validate => inspect::validate(&cli.file),
        Cmd::Run { jobs, no_cache } => run::run(&cli.file, jobs, no_cache),
        Cmd::Graph { format } => inspect::graph(&cli.file, format),
        Cmd::Blocks {
            cmd: BlocksCmd::List,
        } => inspect::blocks_list(),
        Cmd::Store {
            cmd: StoreCmd::Verify { id },
        } => run::verify(&cli.file, &id),
    };
    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            if !e.is::<inspect::Reported>() {
                eprintln!("error: {e:#}");
            }
            ExitCode::FAILURE
        }
    }
}
