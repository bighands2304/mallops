//! Programmatic [`PipelineBuilder`] for Rust library users (available without the `yaml` feature).

use std::collections::BTreeMap;

use crate::config::{BlockConfig, LineageConfig, PipelineConfig, ServeConfig, StoreConfig};
use crate::{BlockRef, BlockRegistry, Error, ValidatedPipeline, ValidationError};

const DEFAULT_SERVE_HOST: &str = "127.0.0.1";
const DEFAULT_SERVE_PORT: u16 = 8080;

/// Programmatic builder for a [`PipelineConfig`] that validates through
/// [`PipelineConfig::validate`].
///
/// Works with `--no-default-features` (no YAML dependency required).
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use mallops_core::{
///     ArtifactKind, Block, BlockOutcome, BlockRegistry, BlockSpec, BoxFuture, Context, Error,
///     Inputs, OutputDrafts, PipelineBuilder, Port, Stage,
/// };
///
/// struct Stub(BlockSpec);
/// impl Block for Stub {
///     fn spec(&self) -> &BlockSpec { &self.0 }
///     fn run<'a>(&'a self, _: &'a Context, _: Inputs) -> BoxFuture<'a, Result<BlockOutcome, Error>> {
///         Box::pin(async { Ok(BlockOutcome::Produced(OutputDrafts::new())) })
///     }
/// }
///
/// # fn main() -> Result<(), Error> {
/// let mut registry = BlockRegistry::new();
/// registry.register(Arc::new(Stub(
///     BlockSpec::new("std/ingest.csv", "1", Stage::Ingest)
///         .with_output(Port::new("dataset", ArtifactKind::Dataset)),
/// )))?;
/// registry.register(Arc::new(Stub(
///     BlockSpec::new("std/train.demo", "1", Stage::Train)
///         .with_input(Port::new("train", ArtifactKind::Dataset))
///         .with_output(Port::new("model", ArtifactKind::Model)),
/// )))?;
/// registry.register(Arc::new(Stub(
///     BlockSpec::new("std/package", "1", Stage::Package)
///         .with_input(Port::new("model", ArtifactKind::Model))
///         .with_output(Port::new("package", ArtifactKind::Package)),
/// )))?;
///
/// let mut builder = PipelineBuilder::new("demo");
/// builder
///     .block("ingest", "std/ingest.csv".parse()?)
///     .with(serde_json::json!({ "path": "data.csv" }));
/// builder
///     .block("model", "std/train.demo".parse()?)
///     .input("train", "ingest.dataset")
///     .seed(42)
///     .cache(true);
/// builder
///     .block("package", "std/package".parse()?)
///     .input("model", "model.model");
/// builder.serve("package.package");
///
/// let pipeline = builder.build(&registry)?;
/// assert_eq!(pipeline.nodes().len(), 3);
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
    /// Wires input port `port` to an upstream `<node>.<output>` reference.
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
            host: DEFAULT_SERVE_HOST.to_owned(),
            port: DEFAULT_SERVE_PORT,
        });
        self
    }

    /// Builds the [`PipelineConfig`] and validates it against `registry`.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] if any block id was added more than once (path `blocks.<id>`) or if
    /// [`PipelineConfig::validate`] fails.
    pub fn build(&self, registry: &BlockRegistry) -> Result<ValidatedPipeline, Error> {
        let mut blocks = BTreeMap::new();
        let mut dup_errors = Vec::new();
        for (id, cfg) in &self.blocks {
            if blocks.contains_key(id) {
                dup_errors.push(ValidationError::new(
                    format!("blocks.{id}"),
                    format!("duplicate block id `{id}`"),
                ));
            } else {
                blocks.insert(id.clone(), cfg.clone());
            }
        }
        if !dup_errors.is_empty() {
            return Err(Error::Validation(dup_errors));
        }

        let config = PipelineConfig {
            version: 1,
            name: self.name.clone(),
            store: StoreConfig::default(),
            lineage: LineageConfig::default(),
            blocks,
            serve: self.serve.clone(),
        };
        config.validate(registry)
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
        builder
            .block("ingest", "std/ingest.csv".parse().unwrap())
            .with(serde_json::json!({ "path": "data/customers.csv" }));
        builder
            .block("model", "std/train.mock".parse().unwrap())
            .input("train", "ingest.dataset")
            .seed(42)
            .cache(false);
        builder
            .block("package", "std/package".parse().unwrap())
            .input("model", "model.model");
        builder.serve("package.package");

        let validated = builder.build(&reg).unwrap();
        let cfg = validated.config();
        assert_eq!(cfg.version, 1);
        assert_eq!(cfg.name, "churn");
        assert_eq!(cfg.store, StoreConfig::default());
        assert_eq!(cfg.lineage, LineageConfig::default());

        let ids: Vec<&str> = validated.nodes().iter().map(|n| n.id.as_str()).collect();
        assert_eq!(ids, ["ingest", "model", "package"]);

        let ingest = &validated.nodes()[0];
        assert_eq!(
            ingest.config.with,
            serde_json::json!({ "path": "data/customers.csv" })
        );
        assert_eq!(ingest.config.cache, None);
        assert_eq!(ingest.config.seed, None);

        let model = &validated.nodes()[1];
        assert_eq!(model.config.r#in["train"], "ingest.dataset");
        assert_eq!(model.config.seed, Some(42));
        assert_eq!(model.config.cache, Some(false));

        let pkg = &validated.nodes()[2];
        assert_eq!(pkg.config.r#in["model"], "model.model");

        let serve = validated.serve().unwrap();
        assert_eq!(serve.package, "package.package");
        assert_eq!(serve.host, "127.0.0.1");
        assert_eq!(serve.port, 8080);
    }

    #[test]
    fn builder_defaults_without_serve_or_options() {
        let reg = test_registry();
        let mut builder = PipelineBuilder::new("minimal");
        builder.block("ingest", "std/ingest.csv".parse().unwrap());

        let validated = builder.build(&reg).unwrap();
        assert!(validated.serve().is_none());
        assert_eq!(validated.nodes().len(), 1);
        let node = &validated.nodes()[0];
        assert_eq!(node.id, "ingest");
        assert!(node.config.r#in.is_empty());
        assert_eq!(node.config.with, serde_json::Value::Null);
        assert_eq!(node.config.cache, None);
        assert_eq!(node.config.seed, None);
    }

    #[test]
    fn builder_rejects_duplicate_block_id() {
        let reg = test_registry();
        let mut builder = PipelineBuilder::new("dup");
        builder.block("ingest", "std/ingest.csv".parse().unwrap());
        builder.block("ingest", "std/ingest.csv".parse().unwrap());

        let err = builder.build(&reg).unwrap_err();
        match err {
            Error::Validation(errs) => {
                assert_eq!(errs.len(), 1);
                assert_eq!(errs[0].path, "blocks.ingest");
                assert!(errs[0].message.contains("duplicate"), "{}", errs[0].message);
            }
            other => panic!("expected Error::Validation, got {other:?}"),
        }
    }

    #[test]
    fn builder_rejects_unknown_block() {
        let reg = BlockRegistry::new();
        let mut builder = PipelineBuilder::new("missing");
        builder.block("raw", "std/ingest.csv".parse().unwrap());

        let err = builder.build(&reg).unwrap_err();
        match err {
            Error::Validation(errs) => {
                assert_eq!(errs.len(), 1);
                assert_eq!(errs[0].path, "blocks.raw.use");
            }
            other => panic!("expected Error::Validation, got {other:?}"),
        }
    }
}
