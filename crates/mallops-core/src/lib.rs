//! Core of `mallops`: format-agnostic artifact model, error types, and core contracts.
//!
//! See `docs/SPEC.md`, `docs/ARCHITECTURE.md`, and `docs/adr/0007-core-contracts-revision.md`.

mod artifact;
mod block;
pub mod config;
mod context;
mod error;
mod pipeline;
mod registry;
pub mod store;

pub use artifact::{Artifact, ArtifactId, ArtifactKind, Digest, Manifest, PayloadFile, RelPath};
pub use block::{
    Block, BlockOutcome, BlockSpec, CachePolicy, DraftPayload, Inputs, OutputDraft, OutputDrafts,
    Port, ResolvedPorts, Stage,
};
pub use context::{
    CancelSource, CancelToken, Context, EnvSecretProvider, MemorySecretProvider, RunId, Secret,
    SecretProvider,
};
pub use error::{Error, ValidationError};
pub use pipeline::{NodeBuilder, PipelineBuilder, ValidatedNode, ValidatedPipeline};
pub use registry::{BlockRef, BlockRegistry};
pub use store::{ArtifactStore, BoxFuture, LocalStore, MemoryStore, PayloadSource, StoreError};
