//! In-memory [`ArtifactStore`] for tests and ephemeral runs.

use std::collections::{BTreeMap, HashMap};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use super::payload::{self, PayloadEntry};
use super::{ArtifactStore, BoxFuture, PayloadSource, StoreError};
use crate::{Artifact, ArtifactId, Error, Manifest, RelPath};

type Files = BTreeMap<RelPath, Vec<u8>>;

/// An [`ArtifactStore`] that keeps artifacts and metadata in process memory.
///
/// Nothing is persisted: all data is dropped with the store.
#[derive(Debug, Default)]
pub struct MemoryStore {
    artifacts: RwLock<HashMap<ArtifactId, (Manifest, Files)>>,
    meta: RwLock<BTreeMap<RelPath, Vec<u8>>>,
}

impl MemoryStore {
    /// Creates an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

fn read<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    // ponytail: a panic mid-write cannot leave these maps half-updated, so poison is safe to ignore.
    lock.read().unwrap_or_else(PoisonError::into_inner)
}

fn write<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(PoisonError::into_inner)
}

/// Blocking: reads a validated payload fully into memory.
fn read_all(payload: PayloadSource<'_>) -> Result<Files, Error> {
    payload::collect(payload)?
        .into_iter()
        .map(|(path, entry)| {
            let bytes = match entry {
                PayloadEntry::File { .. } => {
                    let mut file = entry.open_verified()?;
                    let mut buf = Vec::new();
                    file.read_to_end(&mut buf).map_err(StoreError::from)?;
                    buf
                }
                PayloadEntry::Bytes(b) => b.to_vec(),
            };
            Ok((path, bytes))
        })
        .collect()
}

async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> Result<T, Error> + Send + 'static,
) -> Result<T, Error> {
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| StoreError::Backend(e.to_string()))?
}

async fn load(payload: PayloadSource<'_>) -> Result<Files, Error> {
    match payload {
        PayloadSource::File(p) => {
            let p = p.to_path_buf();
            blocking(move || read_all(PayloadSource::File(&p))).await
        }
        PayloadSource::Dir(p) => {
            let p = p.to_path_buf();
            blocking(move || read_all(PayloadSource::Dir(&p))).await
        }
        other => read_all(other),
    }
}

fn is_segment_prefix(prefix: &str, path: &str) -> bool {
    path.strip_prefix(prefix)
        .is_some_and(|rest| rest.starts_with('/'))
}

fn id_of(manifest: &Manifest, files: &Files) -> Result<ArtifactId, Error> {
    ArtifactId::compute_from_bytes(
        manifest,
        files.iter().map(|(p, b)| (p.clone(), b.as_slice())),
    )
}

impl ArtifactStore for MemoryStore {
    fn put<'a>(
        &'a self,
        manifest: Manifest,
        payload: PayloadSource<'a>,
    ) -> BoxFuture<'a, Result<ArtifactId, Error>> {
        Box::pin(async move {
            let files = load(payload).await?;
            let id = id_of(&manifest, &files)?;
            write(&self.artifacts)
                .entry(id)
                .or_insert((manifest, files));
            Ok(id)
        })
    }

    fn get<'a>(&'a self, id: &'a ArtifactId) -> BoxFuture<'a, Result<Option<Artifact>, Error>> {
        Box::pin(async move {
            Ok(read(&self.artifacts)
                .get(id)
                .map(|(m, _)| Artifact::new(*id, m.clone())))
        })
    }

    fn fetch<'a>(
        &'a self,
        id: &'a ArtifactId,
        dest: &'a Path,
    ) -> BoxFuture<'a, Result<PathBuf, Error>> {
        Box::pin(async move {
            let files = read(&self.artifacts)
                .get(id)
                .map(|(_, f)| f.clone())
                .ok_or(StoreError::NotFound(*id))?;
            let dest = dest.to_path_buf();
            blocking(move || {
                payload::materialize(
                    &dest,
                    files
                        .iter()
                        .map(|(rel, bytes)| Ok((rel.clone(), bytes.as_slice()))),
                )?;
                Ok(dest)
            })
            .await
        })
    }

    fn verify<'a>(&'a self, id: &'a ArtifactId) -> BoxFuture<'a, Result<(), Error>> {
        Box::pin(async move {
            let guard = read(&self.artifacts);
            let (manifest, files) = guard.get(id).ok_or(StoreError::NotFound(*id))?;
            match id_of(manifest, files) {
                Ok(actual) if actual == *id => Ok(()),
                Ok(actual) => Err(StoreError::Corrupted {
                    id: *id,
                    reason: format!("recomputed id {actual} differs"),
                }
                .into()),
                Err(e) => Err(StoreError::Corrupted {
                    id: *id,
                    reason: e.to_string(),
                }
                .into()),
            }
        })
    }

    fn exists<'a>(&'a self, id: &'a ArtifactId) -> BoxFuture<'a, Result<bool, Error>> {
        Box::pin(async move { Ok(read(&self.artifacts).contains_key(id)) })
    }

    fn put_meta<'a>(
        &'a self,
        key: &'a RelPath,
        data: &'a [u8],
    ) -> BoxFuture<'a, Result<(), Error>> {
        Box::pin(async move {
            let mut meta = write(&self.meta);
            if meta.keys().any(|existing| {
                is_segment_prefix(key.as_str(), existing.as_str())
                    || is_segment_prefix(existing.as_str(), key.as_str())
            }) {
                return Err(Error::InvalidPath(format!(
                    "metadata key '{key}' conflicts with an existing key prefix"
                )));
            }
            meta.insert(key.clone(), data.to_vec());
            Ok(())
        })
    }

    fn get_meta<'a>(&'a self, key: &'a RelPath) -> BoxFuture<'a, Result<Option<Vec<u8>>, Error>> {
        Box::pin(async move { Ok(read(&self.meta).get(key).cloned()) })
    }

    fn list_meta<'a>(&'a self, prefix: &'a RelPath) -> BoxFuture<'a, Result<Vec<RelPath>, Error>> {
        Box::pin(async move {
            Ok(read(&self.meta)
                .keys()
                .filter(|k| *k == prefix || is_segment_prefix(prefix.as_str(), k.as_str()))
                .cloned()
                .collect())
        })
    }

    fn delete_meta<'a>(&'a self, key: &'a RelPath) -> BoxFuture<'a, Result<(), Error>> {
        Box::pin(async move {
            write(&self.meta).remove(key);
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::MemoryStore;

    super::super::conformance::conformance_tests!(|| (MemoryStore::new(), ()));
}
