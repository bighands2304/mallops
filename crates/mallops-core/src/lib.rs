//! Core of `mallops`: format-agnostic artifact model, error types, and core contracts.
//!
//! See `docs/SPEC.md`, `docs/ARCHITECTURE.md`, and `docs/adr/0007-core-contracts-revision.md`.

mod artifact;
mod error;
pub mod store;

pub use artifact::{Artifact, ArtifactId, ArtifactKind, Digest, Manifest, PayloadFile, RelPath};
pub use error::Error;
pub use store::{ArtifactStore, BoxFuture, MemoryStore, PayloadSource, StoreError};
