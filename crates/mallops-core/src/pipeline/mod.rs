//! Pipeline validation and programmatic construction (SPEC §5, §6; ARCHITECTURE §4).
//!
//! [`PipelineConfig::validate`] checks the DAG, resolved ports, block configs and the
//! `serve:` target in one pass and never runs a block.

use std::sync::Arc;

use crate::config::{BlockConfig, PipelineConfig, ServeConfig};
use crate::{Block, BlockRegistry, Error, ResolvedPorts};

/// One validated DAG node.
#[derive(Clone)]
#[non_exhaustive]
pub struct ValidatedNode {
    /// Node id (key in `blocks:`).
    pub id: String,
    /// Block resolved from the registry.
    pub block: Arc<dyn Block>,
    /// Node configuration as written in the pipeline.
    pub config: BlockConfig,
    /// Ports resolved for `config.with` (`Block::resolve_ports`).
    pub ports: ResolvedPorts,
}

impl std::fmt::Debug for ValidatedNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValidatedNode")
            .field("id", &self.id)
            .field("block", &self.block.spec().id)
            .field("config", &self.config)
            .field("ports", &self.ports)
            .finish()
    }
}

/// A pipeline that passed validation, with nodes in topological order.
#[derive(Debug, Clone)]
pub struct ValidatedPipeline {
    config: PipelineConfig,
    nodes: Vec<ValidatedNode>,
}

impl ValidatedPipeline {
    /// The validated configuration.
    #[must_use]
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Nodes in topological order (ties broken by node id).
    #[must_use]
    pub fn nodes(&self) -> &[ValidatedNode] {
        &self.nodes
    }

    /// The validated `serve:` section, if any.
    #[must_use]
    pub fn serve(&self) -> Option<&ServeConfig> {
        self.config.serve.as_ref()
    }
}

impl PipelineConfig {
    /// Validates this pipeline against `registry` without running any block.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] with every problem found, each with a dotted path
    /// (e.g. `blocks.eval.in.test`).
    pub fn validate(&self, registry: &BlockRegistry) -> Result<ValidatedPipeline, Error> {
        // ponytail: skeleton from task 6.0, the real checks land in 6.1.
        let mut nodes = Vec::new();
        for (id, cfg) in &self.blocks {
            let block = registry
                .get(&cfg.r#use)
                .ok_or_else(|| Error::validation(format!("blocks.{id}.use"), "unknown block"))?;
            let ports = block.resolve_ports(&cfg.with)?;
            nodes.push(ValidatedNode {
                id: id.clone(),
                block,
                config: cfg.clone(),
                ports,
            });
        }
        Ok(ValidatedPipeline {
            config: self.clone(),
            nodes,
        })
    }
}
