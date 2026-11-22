//! Content-addressed, immutable [`ArtifactStore`] contract and supporting types.
//!
//! See `docs/ARCHITECTURE.md` §6.1 and `docs/adr/0007-core-contracts-revision.md` (D3).

mod local;
pub(crate) mod payload;

pub use local::LocalStore;

use std::collections::BTreeMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;

use thiserror::Error;

use crate::{Artifact, ArtifactId, Error, Manifest, RelPath};

/// A heap-allocated, pinned, `Send` future used for dyn-compatible async trait methods.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Source of payload files to store alongside a [`Manifest`] in an [`ArtifactStore`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PayloadSource<'a> {
    /// A single regular file on the local filesystem.
    ///
    /// Stored under the validated [`RelPath`] derived from the file's final path
    /// component (its file name). Symlinks and special files are rejected.
    File(&'a Path),

    /// A directory on the local filesystem whose regular files are walked recursively.
    ///
    /// Every regular file's path relative to this directory is validated as a
    /// [`RelPath`]. Symlinks and special files anywhere within the tree are rejected.
    Dir(&'a Path),

    /// A single in-memory byte slice stored under [`PayloadSource::DEFAULT_BYTES_PATH`] (`"data"`).
    Bytes(&'a [u8]),

    /// Multiple in-memory files keyed by validated [`RelPath`].
    Files(BTreeMap<RelPath, &'a [u8]>),
}

impl PayloadSource<'_> {
    /// Canonical [`RelPath`] name under which [`PayloadSource::Bytes`] payloads are stored.
    pub const DEFAULT_BYTES_PATH: &'static str = "data";
}

/// Errors returned by [`ArtifactStore`] operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum StoreError {
    /// No artifact with the given [`ArtifactId`] exists in the store.
    #[error("artifact not found: {0}")]
    NotFound(ArtifactId),

    /// A stored artifact failed integrity verification or had malformed stored state.
    #[error("corrupted artifact {id}: {reason}")]
    Corrupted {
        /// Identifier of the corrupted artifact.
        id: ArtifactId,
        /// Human-readable description of the corruption or digest mismatch.
        reason: String,
    },

    /// A payload source contained a disallowed entry (such as a symlink or special file)
    /// or was otherwise invalid for ingestion.
    #[error("invalid payload: {0}")]
    InvalidPayload(String),

    /// A resolved filesystem path escaped the store root boundary.
    #[error("path escapes store root: {}", .0.display())]
    PathOutsideRoot(PathBuf),

    /// An underlying filesystem or I/O operation failed.
    #[error("store I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A backend-specific or internal store error occurred.
    #[error("store backend error: {0}")]
    Backend(String),
}

/// Content-addressed, immutable artifact store and auxiliary metadata contract.
///
/// # Immutability and content addressing
///
/// Artifacts are addressed by their canonical [`ArtifactId`] (BLAKE3 v1 digest
/// over the canonical [`Manifest`] and all payload files). Once stored, an
/// artifact is immutable: repeated or concurrent [`put`](Self::put) calls for
/// identical content deduplicate idempotently, and [`fetch`](Self::fetch) always
/// materializes a copy into the caller-supplied destination rather than exposing
/// a writable store-internal path.
///
/// # Auxiliary metadata
///
/// In addition to immutable artifacts, every store exposes a mutable key-value
/// namespace keyed by validated [`RelPath`] for runtime metadata such as cache
/// entries (`cache/<key>.json`), run lineage (`runs/<run-id>/lineage.json`), and
/// future registry records.
pub trait ArtifactStore: Send + Sync {
    /// Stores an immutable artifact and returns its canonical [`ArtifactId`].
    ///
    /// # Contract
    ///
    /// - **Immutability and deduplication**: computes the canonical [`ArtifactId`]
    ///   from `manifest` and `payload`. Repeated or concurrent calls with identical
    ///   content are idempotent and return the same [`ArtifactId`] without mutating
    ///   existing stored data.
    /// - **Atomicity and read-only storage**: filesystem-backed stores must stage
    ///   writes into a same-filesystem temporary directory, publish via idempotent
    ///   rename, enforce canonical root-boundary checks, and mark stored payload
    ///   files/directories read-only (`0444`/`0555` on Unix where supported).
    /// - **Input validation**: rejects unsupported `manifest.schema_version` with
    ///   [`Error::UnsupportedSchemaVersion`], symlinks and special files with
    ///   [`StoreError::InvalidPayload`], invalid file names with
    ///   [`Error::InvalidPath`], and payloads containing distinct [`RelPath`]s that
    ///   collide after ASCII case folding (e.g. `A.bin` and `a.bin`) with
    ///   [`Error::InvalidPath`]. Unicode normalization collisions (NFC/NFD) are not
    ///   detected in v0.1.
    fn put<'a>(
        &'a self,
        manifest: Manifest,
        payload: PayloadSource<'a>,
    ) -> BoxFuture<'a, Result<ArtifactId, Error>>;

    /// Returns the [`Artifact`] (`ArtifactId` + [`Manifest`]) for `id`, or `Ok(None)` if absent.
    ///
    /// # Contract
    ///
    /// - Returning `Ok(None)` for unknown identifiers avoids `exists` + `get` TOCTOU races.
    /// - Returns [`StoreError::Corrupted`] (wrapped in [`Error::Store`]) if the stored
    ///   manifest is present but malformed or unreadable as valid JSON.
    fn get<'a>(&'a self, id: &'a ArtifactId) -> BoxFuture<'a, Result<Option<Artifact>, Error>>;

    /// Materializes a copy of the artifact's payload files into `dest` and returns the
    /// materialized directory path (`dest.to_path_buf()`).
    ///
    /// # Contract
    ///
    /// - **Always materializes a copy**: must never return a writable path inside the
    ///   store itself. Modifying or deleting files in `dest` after `fetch` must not
    ///   mutate the stored artifact or break subsequent [`verify`](Self::verify) calls.
    /// - Creates `dest` and any nested subdirectories required by payload [`RelPath`]s.
    /// - Returns [`StoreError::NotFound`] (wrapped in [`Error::Store`]) if no artifact
    ///   with `id` exists.
    fn fetch<'a>(
        &'a self,
        id: &'a ArtifactId,
        dest: &'a Path,
    ) -> BoxFuture<'a, Result<PathBuf, Error>>;

    /// Recomputes the canonical [`ArtifactId`] from the stored manifest and payload
    /// files and verifies that it matches `id`.
    ///
    /// # Contract
    ///
    /// - Detects payload tampering, missing or extra files, unexpected symlinks or
    ///   special files inside the stored artifact, and manifest corruption.
    /// - Returns `Ok(())` when the recomputed [`ArtifactId`] matches `id`.
    /// - Returns [`StoreError::NotFound`] if the artifact does not exist, or
    ///   [`StoreError::Corrupted`] if recomputation fails or yields a different ID.
    fn verify<'a>(&'a self, id: &'a ArtifactId) -> BoxFuture<'a, Result<(), Error>>;

    /// Reports whether an artifact with `id` is present in the store.
    ///
    /// # Contract
    ///
    /// - Returns `Ok(true)` if the artifact exists and `Ok(false)` otherwise.
    /// - Does not re-hash payload contents; use [`verify`](Self::verify) for integrity
    ///   checks and [`get`](Self::get) to avoid `exists` + `get` TOCTOU races.
    fn exists<'a>(&'a self, id: &'a ArtifactId) -> BoxFuture<'a, Result<bool, Error>>;

    /// Stores or atomically replaces auxiliary metadata bytes at `key`.
    ///
    /// # Contract
    ///
    /// - Unlike artifacts, metadata entries are mutable (last-write-wins) and keyed by
    ///   validated [`RelPath`] (e.g. `cache/<key>.json`, `runs/<run-id>/lineage.json`).
    /// - Writes must be atomic so concurrent readers never observe partial data, and
    ///   filesystem implementations must enforce the store root boundary.
    fn put_meta<'a>(&'a self, key: &'a RelPath, data: &'a [u8])
    -> BoxFuture<'a, Result<(), Error>>;

    /// Retrieves auxiliary metadata bytes stored at `key`, or `Ok(None)` if absent.
    ///
    /// # Contract
    ///
    /// - Returns `Ok(None)` when no entry exists at `key`.
    /// - Filesystem implementations must enforce the store root boundary.
    fn get_meta<'a>(&'a self, key: &'a RelPath) -> BoxFuture<'a, Result<Option<Vec<u8>>, Error>>;

    /// Lists auxiliary metadata keys matching `prefix` on a path-segment boundary.
    ///
    /// # Contract
    ///
    /// - Matches keys where `key == prefix` or `key` begins with `"{prefix}/"` (so
    ///   prefix `"cache"` matches `"cache/a.json"` but not `"cache2/a.json"`).
    /// - Returns matching keys sorted in ascending [`RelPath`] order (bytewise UTF-8),
    ///   or an empty `Vec` if no keys match.
    fn list_meta<'a>(&'a self, prefix: &'a RelPath) -> BoxFuture<'a, Result<Vec<RelPath>, Error>>;

    /// Deletes the auxiliary metadata entry at `key`.
    ///
    /// # Contract
    ///
    /// - Idempotent: returns `Ok(())` if `key` was removed or did not exist.
    /// - Filesystem implementations must enforce the store root boundary.
    fn delete_meta<'a>(&'a self, key: &'a RelPath) -> BoxFuture<'a, Result<(), Error>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ArtifactKind;

    struct StubStore;

    impl ArtifactStore for StubStore {
        fn put<'a>(
            &'a self,
            _manifest: Manifest,
            _payload: PayloadSource<'a>,
        ) -> BoxFuture<'a, Result<ArtifactId, Error>> {
            Box::pin(async { Ok(ArtifactId::from_bytes([0u8; 32])) })
        }

        fn get<'a>(
            &'a self,
            _id: &'a ArtifactId,
        ) -> BoxFuture<'a, Result<Option<Artifact>, Error>> {
            Box::pin(async { Ok(None) })
        }

        fn fetch<'a>(
            &'a self,
            id: &'a ArtifactId,
            _dest: &'a Path,
        ) -> BoxFuture<'a, Result<PathBuf, Error>> {
            let id = *id;
            Box::pin(async move { Err(StoreError::NotFound(id).into()) })
        }

        fn verify<'a>(&'a self, _id: &'a ArtifactId) -> BoxFuture<'a, Result<(), Error>> {
            Box::pin(async { Ok(()) })
        }

        fn exists<'a>(&'a self, _id: &'a ArtifactId) -> BoxFuture<'a, Result<bool, Error>> {
            Box::pin(async { Ok(false) })
        }

        fn put_meta<'a>(
            &'a self,
            _key: &'a RelPath,
            _data: &'a [u8],
        ) -> BoxFuture<'a, Result<(), Error>> {
            Box::pin(async { Ok(()) })
        }

        fn get_meta<'a>(
            &'a self,
            _key: &'a RelPath,
        ) -> BoxFuture<'a, Result<Option<Vec<u8>>, Error>> {
            Box::pin(async { Ok(None) })
        }

        fn list_meta<'a>(
            &'a self,
            _prefix: &'a RelPath,
        ) -> BoxFuture<'a, Result<Vec<RelPath>, Error>> {
            Box::pin(async { Ok(Vec::new()) })
        }

        fn delete_meta<'a>(&'a self, _key: &'a RelPath) -> BoxFuture<'a, Result<(), Error>> {
            Box::pin(async { Ok(()) })
        }
    }

    fn assert_send_sync<T: Send + Sync>() {}

    #[tokio::test]
    async fn artifact_store_is_object_safe() {
        assert_send_sync::<Box<dyn ArtifactStore>>();

        let store: Box<dyn ArtifactStore> = Box::new(StubStore);
        let manifest = Manifest::new(ArtifactKind::Blob, "bin");
        let id = store
            .put(manifest, PayloadSource::Bytes(b"payload"))
            .await
            .unwrap();

        assert!(store.get(&id).await.unwrap().is_none());
        assert!(!store.exists(&id).await.unwrap());
        store.verify(&id).await.unwrap();

        let err = store.fetch(&id, Path::new("out")).await.unwrap_err();
        assert!(matches!(err, Error::Store(StoreError::NotFound(found)) if found == id));

        let key = RelPath::parse("cache/entry.json").unwrap();
        let prefix = RelPath::parse("cache").unwrap();
        store.put_meta(&key, b"{}").await.unwrap();
        assert!(store.get_meta(&key).await.unwrap().is_none());
        assert!(store.list_meta(&prefix).await.unwrap().is_empty());
        store.delete_meta(&key).await.unwrap();
    }

    #[test]
    fn default_bytes_path_is_valid_rel_path() {
        let path = RelPath::parse(PayloadSource::DEFAULT_BYTES_PATH).unwrap();
        assert_eq!(path.as_str(), "data");
    }
}
