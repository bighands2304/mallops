//! Core error types for `mallops-core`.

use thiserror::Error;

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
}
