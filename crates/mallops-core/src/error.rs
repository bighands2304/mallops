//! Core error types for `mallops-core`.

use thiserror::Error;

use crate::StoreError;

/// Error type returned by `mallops-core` operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// An [`ArtifactId`](crate::ArtifactId) failed strict validation.
    #[error("invalid artifact id: {0}")]
    InvalidArtifactId(String),

    /// A [`Digest`](crate::Digest) failed strict validation.
    #[error("invalid digest: {0}")]
    InvalidDigest(String),

    /// A [`RelPath`](crate::RelPath) failed validation or was duplicated.
    #[error("invalid relative path: {0}")]
    InvalidPath(String),

    /// A [`Manifest`](crate::Manifest) declared an unsupported `schema_version`.
    #[error("unsupported manifest schema_version {found}, expected {expected}")]
    UnsupportedSchemaVersion {
        /// The unsupported schema version found on the manifest.
        found: u32,
        /// The supported schema version ([`Manifest::SCHEMA_VERSION`](crate::Manifest::SCHEMA_VERSION)).
        expected: u32,
    },

    /// An [`ArtifactStore`](crate::ArtifactStore) operation failed.
    #[error("artifact store error: {0}")]
    Store(#[from] StoreError),

    /// A pipeline or block configuration is malformed.
    #[error("config error: {0}")]
    Config(String),

    /// A pipeline, port wiring or block input failed validation.
    #[error("validation error: {0}")]
    Validation(String),

    /// The runtime failed while executing a pipeline.
    #[error("runtime error: {0}")]
    Runtime(String),

    /// The run was cancelled through its cancel token.
    #[error("cancelled")]
    Cancelled,

    /// A secret could not be resolved. The message never contains the secret value.
    #[error("secret error: {0}")]
    Secret(String),

    /// A block failed while running.
    #[error("block `{id}` failed: {source}")]
    Block {
        /// Id of the failing block (e.g. `std/ingest.csv`).
        id: String,
        /// Underlying error reported by the block.
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },
}
