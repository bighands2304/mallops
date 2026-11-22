//! Programmatic [`PipelineBuilder`] for Rust library users (available without the `yaml` feature).

use std::collections::{BTreeMap, BTreeSet};

use crate::config::{
    BlockConfig, LineageConfig, PipelineConfig, ServeConfig, StoreConfig, default_host,
    default_port,
};
use crate::{BlockRef, BlockRegistry, Error, ValidatedPipeline, ValidationError};

/// Programmatic builder for a [`PipelineConfig`] that validates through
/// [`PipelineConfig::validate`].
///
/// Works with `--no-default-features` (no YAML dependency required).
///
/// # Example
///
/// ```
/// # use std::sync::Arc;
/// # use mallops_core::*;
/// # struct Stub(BlockSpec);
/// # impl Block for Stub {
/// #     fn spec(&self) -> &BlockSpec { &self.0 }
/// #     fn run<'a>(&'a self, _: &'a Context, _: Inputs) -> BoxFuture<'a, Result<BlockOutcome, Error>> {
/// #         Box::pin(async { Ok(BlockOutcome::Produced(OutputDrafts::new())) })
/// #     }
/// # }
/// # fn main() -> Result<(), Error> {
/// let mut registry = BlockRegistry::new();
/// registry.register(Arc::new(Stub(
///     BlockSpec::new("std/ingest.csv", "1", Stage::Ingest)
///         .with_output(Port::new("dataset", ArtifactKind::Dataset)),
/// )))?;
/// registry.register(Arc::new(Stub(
///     BlockSpec::new("std/package", "1", Stage::Package)
///         .with_input(Port::new("data", ArtifactKind::Dataset))
///         .with_output(Port::new("package", ArtifactKind::Package)),
/// )))?;
///
/// let mut builder = PipelineBuilder::new("demo");
/// builder.block("ingest", "std/ingest.csv".parse()?);
/// builder.block("pkg", "std/package".parse()?).input("data", "ingest.dataset");
/// builder.serve("pkg.package");
/// assert_eq!(builder.build(&registry)?.nodes().len(), 2);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct PipelineBuilder {
    name: String,
    blocks: Vec<(String, BlockConfig)>,
    serve: Option<ServeConfig>,
}

/// Fluent configurator for one node added via [`PipelineBuilder::block`].
#[derive(Debug)]
pub struct NodeBuilder<'a> {
    config: &'a mut BlockConfig,
}

impl NodeBuilder<'_> {
    /// Wires input port `port` to an upstream `<node>.<output>` reference (replaces any
    /// earlier wiring of the same port).
    pub fn input(&mut self, port: impl Into<String>, source: impl Into<String>) -> &mut Self {
        self.config.r#in.insert(port.into(), source.into());
        self
    }

    /// Sets the block-specific `with:` configuration value.
    pub fn with(&mut self, config: serde_json::Value) -> &mut Self {
        self.config.with = config;
        self
    }

    /// Sets whether caching is enabled for this node (`cache:`).
    pub fn cache(&mut self, enabled: bool) -> &mut Self {
        self.config.cache = Some(enabled);
        self
    }

    /// Sets the deterministic seed for this node (`seed:`).
    pub fn seed(&mut self, seed: u64) -> &mut Self {
        self.config.seed = Some(seed);
        self
    }
}

impl PipelineBuilder {
    /// Starts a new pipeline with schema version 1, default store/lineage, and no `serve:` section.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            blocks: Vec::new(),
            serve: None,
        }
    }

    /// Adds a DAG node with the given `id` and [`BlockRef`], returning a [`NodeBuilder`] to
    /// configure its inputs, `with:` payload, caching and seed.
    ///
    /// Adding two nodes with the same `id` is reported as [`Error::Validation`] in [`Self::build`].
    pub fn block(&mut self, id: impl Into<String>, block: BlockRef) -> NodeBuilder<'_> {
        let idx = self.blocks.len();
        self.blocks.push((
            id.into(),
            BlockConfig {
                r#use: block,
                r#in: BTreeMap::new(),
                with: serde_json::Value::Null,
                cache: None,
                seed: None,
            },
        ));
        NodeBuilder {
            config: &mut self.blocks[idx].1,
        }
    }

    /// Sets the top-level `serve:` package target (`<node>.<output>`) with default bind address
    /// (`127.0.0.1:8080`).
    pub fn serve(&mut self, package: impl Into<String>) -> &mut Self {
        self.serve = Some(ServeConfig {
            package: package.into(),
            host: default_host(),
            port: default_port(),
        });
        self
    }

    /// Builds the [`PipelineConfig`] and validates it against `registry`.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] with one error per block id added more than once (path
    /// `blocks.<id>`) followed by every error of [`PipelineConfig::validate`].
    pub fn build(&self, registry: &BlockRegistry) -> Result<ValidatedPipeline, Error> {
        let mut blocks = BTreeMap::new();
        let (mut dups, mut errs) = (BTreeSet::new(), Vec::new());
        for (id, cfg) in &self.blocks {
            if blocks.insert(id.clone(), cfg.clone()).is_some() && dups.insert(id) {
                errs.push(ValidationError::new(
                    format!("blocks.{id}"),
                    format!("duplicate block id `{id}`"),
                ));
            }
        }
        let config = PipelineConfig {
            version: 1,
            name: self.name.clone(),
            store: StoreConfig::default(),
            lineage: LineageConfig::default(),
            blocks,
            serve: self.serve.clone(),
        };
        match config.validate(registry) {
            Ok(v) if errs.is_empty() => Ok(v),
            Ok(_) => Err(Error::Validation(errs)),
            Err(Error::Validation(v)) => {
                errs.extend(v);
                Err(Error::Validation(errs))
            }
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{
        ArtifactKind, Block, BlockOutcome, BlockSpec, BoxFuture, Context, Inputs, OutputDrafts,
        Port, Stage,
    };

    struct TestBlock(BlockSpec);

    impl Block for TestBlock {
        fn spec(&self) -> &BlockSpec {
            &self.0
        }

        fn run<'a>(
            &'a self,
            _: &'a Context,
            _: Inputs,
        ) -> BoxFuture<'a, Result<BlockOutcome, Error>> {
            Box::pin(async { Ok(BlockOutcome::Produced(OutputDrafts::new())) })
        }
    }

    fn test_registry() -> BlockRegistry {
        let mut reg = BlockRegistry::new();
        reg.register(Arc::new(TestBlock(
            BlockSpec::new("std/ingest.csv", "1", Stage::Ingest)
                .with_output(Port::new("dataset", ArtifactKind::Dataset)),
        )))
        .unwrap();
        reg.register(Arc::new(TestBlock(
            BlockSpec::new("std/train.mock", "1", Stage::Train)
                .with_input(Port::new("train", ArtifactKind::Dataset))
                .with_output(Port::new("model", ArtifactKind::Model)),
        )))
        .unwrap();
        reg.register(Arc::new(TestBlock(
            BlockSpec::new("std/package", "1", Stage::Package)
                .with_input(Port::new("model", ArtifactKind::Model))
                .with_output(Port::new("package", ArtifactKind::Package)),
        )))
        .unwrap();
        reg
    }

    #[test]
    fn builder_constructs_and_validates_multi_node_dag() {
        let reg = test_registry();
        let mut builder = PipelineBuilder::new("churn");
        // added in reverse order: the result must be topological.
        builder
            .block("package", "std/package".parse().unwrap())
            .input("model", "model.model");
        builder
            .block("model", "std/train.mock".parse().unwrap())
            .input("train", "ingest.dataset")
            .cache(false);
        builder.block("ingest", "std/ingest.csv".parse().unwrap());
        builder.serve("package.package");

        let validated = builder.build(&reg).unwrap();
        let ids: Vec<&str> = validated.nodes().iter().map(|n| n.id.as_str()).collect();
        assert_eq!(ids, ["ingest", "model", "package"]);
        assert_eq!(validated.nodes()[1].config.cache, Some(false));
        let serve = validated.config().serve.as_ref().unwrap();
        assert_eq!(
            (serve.package.as_str(), serve.host.as_str(), serve.port),
            ("package.package", "127.0.0.1", 8080)
        );
    }

    #[test]
    fn builder_duplicate_ids_reported_once_with_validation_errors() {
        let reg = test_registry();
        let mut builder = PipelineBuilder::new("dup");
        for _ in 0..3 {
            builder.block("ingest", "std/ingest.csv".parse().unwrap());
        }
        builder.block("raw", "std/nope".parse().unwrap());

        match builder.build(&reg).unwrap_err() {
            Error::Validation(errs) => {
                let paths: Vec<_> = errs.iter().map(|e| e.path.as_str()).collect();
                assert_eq!(paths, ["blocks.ingest", "blocks.raw.use"]);
                assert!(errs[0].message.contains("duplicate"));
            }
            other => panic!("expected Error::Validation, got {other:?}"),
        }
    }
}
