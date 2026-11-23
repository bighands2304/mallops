//! `init`, `validate`, `graph`, `blocks list`: never execute blocks or spawn processes (task 10.1).

use std::fmt::Write as _;
use std::fs::{self, OpenOptions};
use std::io::{self, Write as _};
use std::path::Path;

use mallops_core::{BlockSpec, Port, Stage, ValidatedPipeline};

use crate::GraphFormat;

const INIT_TEMPLATE: &str = "\
version: 1
name: hello

blocks:
  hello:
    use: core/noop
  world:
    use: core/noop
    in:
      in: hello.out
";

/// Writes a minimal runnable `mallops.yaml` into `dir`; fails if it already exists.
pub(crate) fn init(dir: &Path) -> anyhow::Result<()> {
    fs::create_dir_all(dir)?;
    let path = dir.join("mallops.yaml");
    let mut file = match OpenOptions::new().write(true).create_new(true).open(&path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
            anyhow::bail!("{} already exists", path.display());
        }
        Err(e) => return Err(e.into()),
    };
    file.write_all(INIT_TEMPLATE.as_bytes())?;
    println!("created {}", path.display());
    Ok(())
}

/// Parses and validates `file`; prints `ok` or returns every validation error.
pub(crate) fn validate(file: &Path) -> anyhow::Result<()> {
    let pipeline = crate::load(file)?;
    crate::run::check_backends(pipeline.config())?;
    println!(
        "ok: {} ({} blocks)",
        pipeline.config().name,
        pipeline.nodes().len()
    );
    Ok(())
}

/// Prints the validated DAG of `file` as Mermaid or DOT.
pub(crate) fn graph(file: &Path, format: GraphFormat) -> anyhow::Result<()> {
    let pipeline = crate::load(file)?;
    let out = match format {
        GraphFormat::Mermaid => render_mermaid(&pipeline),
        GraphFormat::Dot => render_dot(&pipeline),
    };
    print!("{out}");
    Ok(())
}

/// Lists registered blocks (`id`, `version`, `stage`, ports), sorted by id.
pub(crate) fn blocks_list() -> anyhow::Result<()> {
    let reg = crate::registry()?;
    print!("{}", format_blocks_table(&reg.list()));
    Ok(())
}

fn escape_mermaid(s: &str) -> String {
    s.replace('"', "&quot;")
}

fn escape_dot(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Yields `(upstream_node, input_port, downstream_node)` edges in topological node order.
fn edges(pipeline: &ValidatedPipeline) -> impl Iterator<Item = (&str, &str, &str)> {
    pipeline.nodes().iter().flat_map(|node| {
        node.config.r#in.iter().filter_map(move |(port, target)| {
            let (up, _) = target.split_once('.')?;
            Some((up, port.as_str(), node.id.as_str()))
        })
    })
}

fn render_mermaid(pipeline: &ValidatedPipeline) -> String {
    let mut out = String::from("flowchart TD\n");
    for node in pipeline.nodes() {
        let id = escape_mermaid(&node.id);
        let block_use = escape_mermaid(&node.config.r#use.to_string());
        let _ = writeln!(out, "  {id}[\"{id}<br/>{block_use}\"]");
    }
    for (up, port, down) in edges(pipeline) {
        let _ = writeln!(
            out,
            "  {} -->|{}| {}",
            escape_mermaid(up),
            escape_mermaid(port),
            escape_mermaid(down),
        );
    }
    out
}

fn render_dot(pipeline: &ValidatedPipeline) -> String {
    let mut out = String::new();
    let name = escape_dot(&pipeline.config().name);
    let _ = writeln!(out, "digraph \"{name}\" {{");
    for node in pipeline.nodes() {
        let id = escape_dot(&node.id);
        let block_use = escape_dot(&node.config.r#use.to_string());
        let _ = writeln!(out, "  \"{id}\" [label=\"{id}\\n{block_use}\"];");
    }
    for (up, port, down) in edges(pipeline) {
        let _ = writeln!(
            out,
            "  \"{}\" -> \"{}\" [label=\"{}\"];",
            escape_dot(up),
            escape_dot(down),
            escape_dot(port),
        );
    }
    out.push_str("}\n");
    out
}

fn stage_name(stage: Stage) -> &'static str {
    match stage {
        Stage::Ingest => "ingest",
        Stage::DataStorage => "data_storage",
        Stage::Preprocess => "preprocess",
        Stage::Train => "train",
        Stage::ModelEngineering => "model_engineering",
        Stage::Evaluate => "evaluate",
        Stage::Package => "package",
        Stage::Version => "version",
        Stage::Deploy => "deploy",
        Stage::Serve => "serve",
        Stage::Monitor => "monitor",
        Stage::Retrain => "retrain",
        Stage::Custom => "custom",
        _ => "custom",
    }
}

fn format_ports(ports: &[Port]) -> String {
    if ports.is_empty() {
        return "-".to_owned();
    }
    ports
        .iter()
        .map(|p| {
            if p.required {
                format!("{}:{}", p.name, p.kind.as_str())
            } else {
                format!("{}?:{}", p.name, p.kind.as_str())
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn format_blocks_table(specs: &[&BlockSpec]) -> String {
    let mut rows: Vec<[String; 5]> = Vec::with_capacity(specs.len() + 1);
    rows.push([
        "ID".to_owned(),
        "VERSION".to_owned(),
        "STAGE".to_owned(),
        "INPUTS".to_owned(),
        "OUTPUTS".to_owned(),
    ]);
    for spec in specs {
        rows.push([
            spec.id.to_string(),
            spec.version.to_string(),
            stage_name(spec.stage).to_owned(),
            format_ports(&spec.inputs),
            format_ports(&spec.outputs),
        ]);
    }
    let mut widths = [0usize; 4];
    for row in &rows {
        for (w, cell) in widths.iter_mut().zip(row.iter()) {
            *w = (*w).max(cell.len());
        }
    }
    let mut out = String::new();
    for row in &rows {
        for (cell, &w) in row[..4].iter().zip(&widths) {
            let _ = write!(out, "{cell:<w$}  ");
        }
        let _ = writeln!(out, "{}", row[4]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use mallops_core::{ArtifactKind, BlockRef, PipelineBuilder};

    #[test]
    fn inspect_escape_quotes_dot_and_mermaid() {
        assert_eq!(escape_dot("a\"b\\c"), "a\\\"b\\\\c");
        assert_eq!(escape_mermaid("a\"b"), "a&quot;b");

        let noop: BlockRef = "core/noop".parse().unwrap();
        let mut builder = PipelineBuilder::new("pipe \"quoted\"");
        builder.block("a", noop.clone());
        builder.block("b", noop).input("in", "a.out");
        let pipeline = builder.build(&crate::registry().unwrap()).unwrap();

        let dot = render_dot(&pipeline);
        assert!(dot.starts_with("digraph \"pipe \\\"quoted\\\"\" {\n"));
    }

    #[test]
    fn inspect_format_ports_and_table() {
        assert_eq!(format_ports(&[]), "-");
        let ports = [
            Port::new("data", ArtifactKind::Dataset),
            Port::new("opt", ArtifactKind::Blob).optional(),
        ];
        assert_eq!(format_ports(&ports), "data:dataset,opt?:blob");

        let s1 = BlockSpec::new("core/a", "1.0.0", Stage::Ingest)
            .with_output(Port::new("out", ArtifactKind::Dataset));
        let s2 = BlockSpec::new("std/train.fast", "0.2.0", Stage::Train)
            .with_input(Port::new("train", ArtifactKind::Features))
            .with_input(Port::new("val", ArtifactKind::Features).optional())
            .with_output(Port::new("model", ArtifactKind::Model));
        let table = format_blocks_table(&[&s1, &s2]);
        let expected = "\
ID              VERSION  STAGE   INPUTS                        OUTPUTS
core/a          1.0.0    ingest  -                             out:dataset
std/train.fast  0.2.0    train   train:features,val?:features  model:model
";
        assert_eq!(table, expected);
    }
}
