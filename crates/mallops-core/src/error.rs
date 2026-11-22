//! Core error types for `mallops-core`.

use thiserror::Error;

use crate::StoreError;

/// One validation failure, located by a dotted path into the pipeline
/// (e.g. `blocks.eval.in.test`, `serve.package`); the path is empty when there is no location.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ValidationError {
    /// Dotted location of the failure (may be empty).
    pub path: String,
    /// Human-readable description.
    pub message: String,
}

impl ValidationError {
    /// Creates a validation error at `path`.
    #[must_use]
    pub fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.path.is_empty() {
            f.write_str(&self.message)
        } else {
            write!(f, "{}: {}", self.path, self.message)
        }
    }
}

fn join(errors: &[ValidationError]) -> String {
    errors
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("; ")
}

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

    /// A pipeline, port wiring or block input failed validation; all errors found in one pass.
    #[error("validation error: {}", join(.0))]
    Validation(Vec<ValidationError>),

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

impl Error {
    /// A single [`Error::Validation`] at `path` (empty for no location).
    #[must_use]
    pub fn validation(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Validation(vec![ValidationError::new(path, message)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_display_joins_errors() {
        let err = Error::Validation(vec![
            ValidationError::new("blocks.a.use", "unknown block"),
            ValidationError::new("", "no location"),
        ]);
        assert_eq!(
            err.to_string(),
            "validation error: blocks.a.use: unknown block; no location"
        );
    }
}
