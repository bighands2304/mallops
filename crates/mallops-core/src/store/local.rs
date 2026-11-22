//! Local filesystem-backed [`ArtifactStore`] implementation.
//!
//! Layout under the canonicalized store root (default [`LocalStore::DEFAULT_ROOT`]):
//! - `artifacts/<artifact-id>/manifest.json` — canonical JSON manifest (`0444` on Unix)
//! - `artifacts/<artifact-id>/payload/…` — read-only payload files (`0444` files / `0555` dirs on Unix)
//! - `meta/<rel-path>` — mutable auxiliary metadata (cache entries, run lineage, etc.)
//! - `.tmp/` — same-filesystem staging directory for atomic writes

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};

use crate::store::payload::{self, PayloadEntry};
use crate::{
    Artifact, ArtifactId, ArtifactStore, BoxFuture, Digest, Error, Manifest, PayloadFile,
    PayloadSource, RelPath, StoreError,
};

const ARTIFACTS_DIR: &str = "artifacts";
const META_DIR: &str = "meta";
const TMP_DIR: &str = ".tmp";
const MANIFEST_FILE: &str = "manifest.json";
const PAYLOAD_DIR: &str = "payload";
const IO_BUF_SIZE: usize = 64 * 1024;

/// Local filesystem-backed, immutable, content-addressed [`ArtifactStore`].
///
/// All filesystem I/O and BLAKE3 hashing execute on Tokio's blocking pool via
/// `tokio::task::spawn_blocking`. Writes are staged inside `<root>/.tmp` on the
/// same filesystem and published via atomic rename.
#[derive(Debug, Clone)]
pub struct LocalStore {
    root: PathBuf,
}

impl LocalStore {
    /// Default on-disk root directory (`./.mallops`).
    pub const DEFAULT_ROOT: &'static str = "./.mallops";

    /// Opens or initializes a [`LocalStore`] at `root`.
    ///
    /// Creates `root` (and any missing parent directories) if it does not exist,
    /// then canonicalizes it for root-boundary enforcement.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] wrapping [`StoreError::Io`] if `root` cannot be
    /// created or canonicalized, or is not a directory.
    pub fn new(root: impl AsRef<Path>) -> Result<Self, Error> {
        let root = root.as_ref();
        fs::create_dir_all(root).map_err(StoreError::from)?;
        let canonical = fs::canonicalize(root).map_err(StoreError::from)?;
        let meta = fs::metadata(&canonical).map_err(StoreError::from)?;
        if !meta.is_dir() {
            return Err(StoreError::Io(io::Error::new(
                io::ErrorKind::NotADirectory,
                format!("store root is not a directory: {}", canonical.display()),
            ))
            .into());
        }
        Ok(Self { root: canonical })
    }

    /// Opens or initializes a [`LocalStore`] at [`LocalStore::DEFAULT_ROOT`] (`./.mallops`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Store`] if the default root directory cannot be created or canonicalized.
    pub fn open_default() -> Result<Self, Error> {
        Self::new(Self::DEFAULT_ROOT)
    }

    /// Returns the canonicalized root directory of this store.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }
}

enum OwnedPayloadSource {
    File(PathBuf),
    Dir(PathBuf),
    Bytes(Vec<u8>),
    Files(BTreeMap<RelPath, Vec<u8>>),
}

impl OwnedPayloadSource {
    fn from_borrowed(source: PayloadSource<'_>) -> Self {
        match source {
            PayloadSource::File(path) => Self::File(path.to_path_buf()),
            PayloadSource::Dir(path) => Self::Dir(path.to_path_buf()),
            PayloadSource::Bytes(bytes) => Self::Bytes(bytes.to_vec()),
            PayloadSource::Files(files) => Self::Files(
                files
                    .into_iter()
                    .map(|(rel, bytes)| (rel, bytes.to_vec()))
                    .collect(),
            ),
        }
    }

    fn as_borrowed(&self) -> PayloadSource<'_> {
        match self {
            Self::File(path) => PayloadSource::File(path.as_path()),
            Self::Dir(path) => PayloadSource::Dir(path.as_path()),
            Self::Bytes(bytes) => PayloadSource::Bytes(bytes.as_slice()),
            Self::Files(files) => PayloadSource::Files(
                files
                    .iter()
                    .map(|(rel, bytes)| (rel.clone(), bytes.as_slice()))
                    .collect(),
            ),
        }
    }
}

async fn run_blocking<F, T>(f: F) -> Result<T, Error>
where
    F: FnOnce() -> Result<T, Error> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(f).await.map_err(|err| {
        Error::Store(StoreError::Backend(format!(
            "blocking store task failed: {err}"
        )))
    })?
}

impl ArtifactStore for LocalStore {
    fn put<'a>(
        &'a self,
        manifest: Manifest,
        payload: PayloadSource<'a>,
    ) -> BoxFuture<'a, Result<ArtifactId, Error>> {
        let root = self.root.clone();
        let owned_payload = OwnedPayloadSource::from_borrowed(payload);
        Box::pin(
            async move { run_blocking(move || put_sync(&root, manifest, owned_payload)).await },
        )
    }

    fn get<'a>(&'a self, id: &'a ArtifactId) -> BoxFuture<'a, Result<Option<Artifact>, Error>> {
        let root = self.root.clone();
        let id = *id;
        Box::pin(async move { run_blocking(move || get_sync(&root, id)).await })
    }

    fn fetch<'a>(
        &'a self,
        id: &'a ArtifactId,
        dest: &'a Path,
    ) -> BoxFuture<'a, Result<PathBuf, Error>> {
        let root = self.root.clone();
        let id = *id;
        let dest = dest.to_path_buf();
        Box::pin(async move { run_blocking(move || fetch_sync(&root, id, &dest)).await })
    }

    fn verify<'a>(&'a self, id: &'a ArtifactId) -> BoxFuture<'a, Result<(), Error>> {
        let root = self.root.clone();
        let id = *id;
        Box::pin(async move { run_blocking(move || verify_sync(&root, id)).await })
    }

    fn exists<'a>(&'a self, id: &'a ArtifactId) -> BoxFuture<'a, Result<bool, Error>> {
        let root = self.root.clone();
        let id = *id;
        Box::pin(async move { run_blocking(move || exists_sync(&root, id)).await })
    }

    fn put_meta<'a>(
        &'a self,
        key: &'a RelPath,
        data: &'a [u8],
    ) -> BoxFuture<'a, Result<(), Error>> {
        let root = self.root.clone();
        let key = key.clone();
        let data = data.to_vec();
        Box::pin(async move { run_blocking(move || put_meta_sync(&root, &key, &data)).await })
    }

    fn get_meta<'a>(&'a self, key: &'a RelPath) -> BoxFuture<'a, Result<Option<Vec<u8>>, Error>> {
        let root = self.root.clone();
        let key = key.clone();
        Box::pin(async move { run_blocking(move || get_meta_sync(&root, &key)).await })
    }

    fn list_meta<'a>(&'a self, prefix: &'a RelPath) -> BoxFuture<'a, Result<Vec<RelPath>, Error>> {
        let root = self.root.clone();
        let prefix = prefix.clone();
        Box::pin(async move { run_blocking(move || list_meta_sync(&root, &prefix)).await })
    }

    fn delete_meta<'a>(&'a self, key: &'a RelPath) -> BoxFuture<'a, Result<(), Error>> {
        let root = self.root.clone();
        let key = key.clone();
        Box::pin(async move { run_blocking(move || delete_meta_sync(&root, &key)).await })
    }
}

/// Ensures `candidate` stays lexically and canonically inside `root` without crossing symlinks.
fn ensure_within_root(root: &Path, candidate: &Path) -> Result<(), Error> {
    let rel = candidate
        .strip_prefix(root)
        .map_err(|_| StoreError::PathOutsideRoot(candidate.to_path_buf()))?;

    let mut cur = root.to_path_buf();
    for component in rel.components() {
        match component {
            Component::Normal(part) => cur.push(part),
            Component::CurDir => continue,
            _ => return Err(StoreError::PathOutsideRoot(candidate.to_path_buf()).into()),
        }

        match fs::symlink_metadata(&cur) {
            Ok(meta) => {
                if meta.file_type().is_symlink() {
                    return Err(StoreError::PathOutsideRoot(cur).into());
                }
                let canonical = fs::canonicalize(&cur).map_err(StoreError::from)?;
                if !canonical.starts_with(root) {
                    return Err(StoreError::PathOutsideRoot(canonical).into());
                }
            }
            Err(err) if err.kind() == io::ErrorKind::NotFound => break,
            Err(err) => return Err(StoreError::Io(err).into()),
        }
    }

    Ok(())
}

fn artifact_dir(root: &Path, id: ArtifactId) -> PathBuf {
    root.join(ARTIFACTS_DIR).join(id.to_hex())
}

fn ensure_dir_within_root(root: &Path, dir: &Path) -> Result<(), Error> {
    ensure_within_root(root, dir)?;
    fs::create_dir_all(dir).map_err(StoreError::from)?;
    ensure_within_root(root, dir)
}

fn put_sync(
    root: &Path,
    manifest: Manifest,
    owned_payload: OwnedPayloadSource,
) -> Result<ArtifactId, Error> {
    if manifest.schema_version != Manifest::SCHEMA_VERSION {
        return Err(Error::UnsupportedSchemaVersion {
            found: manifest.schema_version,
            expected: Manifest::SCHEMA_VERSION,
        });
    }

    let entries = payload::collect(owned_payload.as_borrowed())?;

    let tmp_root = root.join(TMP_DIR);
    let artifacts_root = root.join(ARTIFACTS_DIR);
    ensure_dir_within_root(root, &tmp_root)?;
    ensure_dir_within_root(root, &artifacts_root)?;

    let staging = tempfile::Builder::new()
        .prefix("stage-")
        .tempdir_in(&tmp_root)
        .map_err(StoreError::from)?;

    let payload_stage = staging.path().join(PAYLOAD_DIR);
    fs::create_dir(&payload_stage).map_err(StoreError::from)?;

    let mut payload_files = Vec::with_capacity(entries.len());
    let mut buf = vec![0u8; IO_BUF_SIZE];

    for (rel, entry) in &entries {
        let dest_path = payload_stage.join(rel.as_path());
        if let Some(parent) = dest_path.parent() {
            fs::create_dir_all(parent).map_err(StoreError::from)?;
        }

        let (len, digest) = match entry {
            PayloadEntry::Bytes(bytes) => {
                fs::write(&dest_path, bytes).map_err(StoreError::from)?;
                (bytes.len() as u64, Digest::hash(bytes))
            }
            PayloadEntry::File(src_path) => copy_and_hash_file(src_path, &dest_path, &mut buf)?,
        };

        payload_files.push(PayloadFile::new(rel.clone(), len, digest));
    }

    let id = ArtifactId::compute(&manifest, &payload_files)?;
    let manifest_bytes = manifest.canonical_json();
    let manifest_stage = staging.path().join(MANIFEST_FILE);
    fs::write(&manifest_stage, &manifest_bytes).map_err(StoreError::from)?;

    let final_dir = artifact_dir(root, id);
    ensure_within_root(root, &final_dir)?;

    // Fast-path dedup before locking permissions and renaming.
    if fs::symlink_metadata(&final_dir).is_ok_and(|meta| meta.file_type().is_dir()) {
        return Ok(id);
    }

    // Take ownership of staging_path so read-only cleanup always runs if publishing fails.
    let staging_path = staging.keep();
    let publish_res = publish_staged_artifact(&staging_path, &final_dir, id);
    if publish_res.is_err() {
        let _ = remove_readonly_tree(&staging_path);
    }
    publish_res
}

fn publish_staged_artifact(
    staging_path: &Path,
    final_dir: &Path,
    id: ArtifactId,
) -> Result<ArtifactId, Error> {
    set_file_readonly(&staging_path.join(MANIFEST_FILE))?;
    set_tree_readonly(&staging_path.join(PAYLOAD_DIR))?;

    match fs::rename(staging_path, final_dir) {
        Ok(()) => {
            set_dir_readonly(final_dir)?;
            Ok(id)
        }
        Err(err) => {
            let _ = remove_readonly_tree(staging_path);
            if fs::symlink_metadata(final_dir).is_ok_and(|meta| meta.file_type().is_dir()) {
                Ok(id)
            } else {
                Err(StoreError::Io(err).into())
            }
        }
    }
}

fn copy_and_hash_file(src: &Path, dest: &Path, buf: &mut [u8]) -> Result<(u64, Digest), Error> {
    let meta = fs::symlink_metadata(src).map_err(StoreError::from)?;
    let ft = meta.file_type();
    if ft.is_symlink() || !ft.is_file() {
        return Err(StoreError::InvalidPayload(format!(
            "payload source changed to non-regular file: {}",
            src.display()
        ))
        .into());
    }

    let mut reader = File::open(src).map_err(StoreError::from)?;
    let mut writer = File::create(dest).map_err(StoreError::from)?;
    let mut hasher = blake3::Hasher::new();
    let mut total_len = 0u64;

    loop {
        let n = reader.read(buf).map_err(StoreError::from)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        writer.write_all(&buf[..n]).map_err(StoreError::from)?;
        total_len += n as u64;
    }

    writer.flush().map_err(StoreError::from)?;
    Ok((total_len, Digest::from_bytes(*hasher.finalize().as_bytes())))
}

fn hash_file_streaming(path: &Path, buf: &mut [u8]) -> Result<(u64, Digest), io::Error> {
    let mut reader = File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut total_len = 0u64;

    loop {
        let n = reader.read(buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total_len += n as u64;
    }

    Ok((total_len, Digest::from_bytes(*hasher.finalize().as_bytes())))
}

fn get_sync(root: &Path, id: ArtifactId) -> Result<Option<Artifact>, Error> {
    let dir = artifact_dir(root, id);
    ensure_within_root(root, &dir)?;

    let meta = match fs::symlink_metadata(&dir) {
        Ok(meta) => meta,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(StoreError::Io(err).into()),
    };
    if !meta.file_type().is_dir() {
        return Err(StoreError::Corrupted {
            id,
            reason: "stored artifact entry is not a directory".to_owned(),
        }
        .into());
    }

    let manifest = read_stored_manifest(&dir, id)?;
    Ok(Some(Artifact::new(id, manifest)))
}

fn exists_sync(root: &Path, id: ArtifactId) -> Result<bool, Error> {
    let dir = artifact_dir(root, id);
    ensure_within_root(root, &dir)?;

    match fs::symlink_metadata(&dir) {
        Ok(meta) => Ok(meta.file_type().is_dir()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(err) => Err(StoreError::Io(err).into()),
    }
}

fn fetch_sync(root: &Path, id: ArtifactId, dest: &Path) -> Result<PathBuf, Error> {
    let dir = artifact_dir(root, id);
    ensure_within_root(root, &dir)?;

    let dir_meta = match fs::symlink_metadata(&dir) {
        Ok(meta) => meta,
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            return Err(StoreError::NotFound(id).into());
        }
        Err(err) => return Err(StoreError::Io(err).into()),
    };
    if !dir_meta.file_type().is_dir() {
        return Err(StoreError::Corrupted {
            id,
            reason: "stored artifact entry is not a directory".to_owned(),
        }
        .into());
    }

    let payload_dir = dir.join(PAYLOAD_DIR);
    let payload_meta = fs::symlink_metadata(&payload_dir).map_err(|err| StoreError::Corrupted {
        id,
        reason: format!("missing or unreadable payload directory: {err}"),
    })?;
    if payload_meta.file_type().is_symlink() || !payload_meta.file_type().is_dir() {
        return Err(StoreError::Corrupted {
            id,
            reason: "stored payload entry is not a regular directory".to_owned(),
        }
        .into());
    }

    let stored_files = walk_stored_payload(&payload_dir, id)?;
    fs::create_dir_all(dest).map_err(StoreError::from)?;

    for (rel, src_path) in stored_files {
        let out_path = dest.join(rel.as_path());
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent).map_err(StoreError::from)?;
        }
        // Stream copy instead of `fs::copy` so materialized files do not inherit `0444` mode bits.
        let mut reader = File::open(&src_path).map_err(StoreError::from)?;
        let mut writer = File::create(&out_path).map_err(StoreError::from)?;
        io::copy(&mut reader, &mut writer).map_err(StoreError::from)?;
        writer.flush().map_err(StoreError::from)?;
    }

    Ok(dest.to_path_buf())
}

fn verify_sync(root: &Path, id: ArtifactId) -> Result<(), Error> {
    let dir = artifact_dir(root, id);
    ensure_within_root(root, &dir)?;

    let dir_meta = match fs::symlink_metadata(&dir) {
        Ok(meta) => meta,
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            return Err(StoreError::NotFound(id).into());
        }
        Err(err) => return Err(StoreError::Io(err).into()),
    };
    if !dir_meta.file_type().is_dir() {
        return Err(StoreError::Corrupted {
            id,
            reason: "stored artifact entry is not a directory".to_owned(),
        }
        .into());
    }

    // Reject unexpected extra entries at the artifact root directory.
    let mut saw_manifest = false;
    let mut saw_payload = false;
    let read_dir = fs::read_dir(&dir).map_err(|err| StoreError::Corrupted {
        id,
        reason: format!("failed to read artifact directory: {err}"),
    })?;
    for entry_res in read_dir {
        let entry = entry_res.map_err(|err| StoreError::Corrupted {
            id,
            reason: format!("failed to read artifact directory entry: {err}"),
        })?;
        let name = entry.file_name();
        match name.to_str() {
            Some(MANIFEST_FILE) => saw_manifest = true,
            Some(PAYLOAD_DIR) => saw_payload = true,
            _ => {
                return Err(StoreError::Corrupted {
                    id,
                    reason: format!(
                        "unexpected entry in artifact directory: {}",
                        name.to_string_lossy()
                    ),
                }
                .into());
            }
        }
    }
    if !saw_manifest || !saw_payload {
        return Err(StoreError::Corrupted {
            id,
            reason: "artifact directory is missing manifest.json or payload/".to_owned(),
        }
        .into());
    }

    let manifest = read_stored_manifest(&dir, id)?;

    let payload_dir = dir.join(PAYLOAD_DIR);
    let payload_meta = fs::symlink_metadata(&payload_dir).map_err(|err| StoreError::Corrupted {
        id,
        reason: format!("failed to inspect payload directory: {err}"),
    })?;
    if payload_meta.file_type().is_symlink() || !payload_meta.file_type().is_dir() {
        return Err(StoreError::Corrupted {
            id,
            reason: "stored payload entry is not a regular directory".to_owned(),
        }
        .into());
    }

    let stored_files = walk_stored_payload(&payload_dir, id)?;
    let mut payload_files = Vec::with_capacity(stored_files.len());
    let mut buf = vec![0u8; IO_BUF_SIZE];

    for (rel, path) in stored_files {
        let (len, digest) =
            hash_file_streaming(&path, &mut buf).map_err(|err| StoreError::Corrupted {
                id,
                reason: format!("failed to read stored payload file '{rel}': {err}"),
            })?;
        payload_files.push(PayloadFile::new(rel, len, digest));
    }

    let computed =
        ArtifactId::compute(&manifest, &payload_files).map_err(|err| StoreError::Corrupted {
            id,
            reason: format!("failed to recompute canonical ArtifactId: {err}"),
        })?;

    if computed != id {
        return Err(StoreError::Corrupted {
            id,
            reason: format!("digest mismatch: expected {id}, computed {computed}"),
        }
        .into());
    }

    Ok(())
}

fn read_stored_manifest(artifact_dir: &Path, id: ArtifactId) -> Result<Manifest, Error> {
    let manifest_path = artifact_dir.join(MANIFEST_FILE);
    let meta = fs::symlink_metadata(&manifest_path).map_err(|err| StoreError::Corrupted {
        id,
        reason: format!("missing or unreadable manifest.json: {err}"),
    })?;
    let ft = meta.file_type();
    if ft.is_symlink() || !ft.is_file() {
        return Err(StoreError::Corrupted {
            id,
            reason: "manifest.json is not a regular file".to_owned(),
        }
        .into());
    }

    let bytes = fs::read(&manifest_path).map_err(|err| StoreError::Corrupted {
        id,
        reason: format!("failed to read manifest.json: {err}"),
    })?;
    let manifest: Manifest =
        serde_json::from_slice(&bytes).map_err(|err| StoreError::Corrupted {
            id,
            reason: format!("malformed manifest.json: {err}"),
        })?;
    if manifest.schema_version != Manifest::SCHEMA_VERSION {
        return Err(StoreError::Corrupted {
            id,
            reason: format!(
                "unsupported manifest schema_version {}, expected {}",
                manifest.schema_version,
                Manifest::SCHEMA_VERSION
            ),
        }
        .into());
    }
    Ok(manifest)
}

fn walk_stored_payload(
    payload_dir: &Path,
    id: ArtifactId,
) -> Result<BTreeMap<RelPath, PathBuf>, Error> {
    let mut files = BTreeMap::new();
    let mut stack: Vec<(PathBuf, String)> = vec![(payload_dir.to_path_buf(), String::new())];

    while let Some((dir_path, rel_prefix)) = stack.pop() {
        let read_dir = fs::read_dir(&dir_path).map_err(|err| StoreError::Corrupted {
            id,
            reason: format!("failed to read payload directory: {err}"),
        })?;

        for entry_res in read_dir {
            let entry = entry_res.map_err(|err| StoreError::Corrupted {
                id,
                reason: format!("failed to read payload entry: {err}"),
            })?;
            let entry_path = entry.path();
            let meta = fs::symlink_metadata(&entry_path).map_err(|err| StoreError::Corrupted {
                id,
                reason: format!("failed to inspect payload entry: {err}"),
            })?;
            let ft = meta.file_type();

            if ft.is_symlink() {
                return Err(StoreError::Corrupted {
                    id,
                    reason: format!(
                        "unexpected symlink in stored payload: {}",
                        entry_path.display()
                    ),
                }
                .into());
            }
            if !ft.is_file() && !ft.is_dir() {
                return Err(StoreError::Corrupted {
                    id,
                    reason: format!(
                        "unexpected special file in stored payload: {}",
                        entry_path.display()
                    ),
                }
                .into());
            }

            let file_name = entry.file_name();
            let name_str = file_name.to_str().ok_or_else(|| StoreError::Corrupted {
                id,
                reason: format!(
                    "non-UTF-8 path component in stored payload: {}",
                    entry_path.display()
                ),
            })?;

            let rel_str = if rel_prefix.is_empty() {
                name_str.to_owned()
            } else {
                format!("{rel_prefix}/{name_str}")
            };

            let rel = RelPath::parse(&rel_str).map_err(|err| StoreError::Corrupted {
                id,
                reason: format!("invalid stored payload path '{rel_str}': {err}"),
            })?;

            if ft.is_dir() {
                stack.push((entry_path, rel_str));
            } else {
                files.insert(rel, entry_path);
            }
        }
    }

    Ok(files)
}

fn put_meta_sync(root: &Path, key: &RelPath, data: &[u8]) -> Result<(), Error> {
    let tmp_root = root.join(TMP_DIR);
    let meta_root = root.join(META_DIR);
    ensure_dir_within_root(root, &tmp_root)?;
    ensure_dir_within_root(root, &meta_root)?;

    let dest = meta_root.join(key.as_path());
    ensure_within_root(root, &dest)?;

    if let Some(parent) = dest.parent() {
        ensure_dir_within_root(root, parent)?;
    }
    ensure_within_root(root, &dest)?;

    let mut tmp = tempfile::Builder::new()
        .prefix("meta-")
        .tempfile_in(&tmp_root)
        .map_err(StoreError::from)?;
    tmp.write_all(data).map_err(StoreError::from)?;
    tmp.flush().map_err(StoreError::from)?;

    tmp.persist(&dest)
        .map_err(|err| StoreError::Io(err.error))?;
    Ok(())
}

fn get_meta_sync(root: &Path, key: &RelPath) -> Result<Option<Vec<u8>>, Error> {
    let dest = root.join(META_DIR).join(key.as_path());
    ensure_within_root(root, &dest)?;

    let meta = match fs::symlink_metadata(&dest) {
        Ok(meta) => meta,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(StoreError::Io(err).into()),
    };

    let ft = meta.file_type();
    if ft.is_symlink() {
        return Err(StoreError::PathOutsideRoot(dest).into());
    }
    if ft.is_dir() {
        return Ok(None);
    }
    if !ft.is_file() {
        return Err(StoreError::InvalidPayload(format!(
            "metadata entry is not a regular file: {}",
            dest.display()
        ))
        .into());
    }

    match fs::read(&dest) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(StoreError::Io(err).into()),
    }
}

fn list_meta_sync(root: &Path, prefix: &RelPath) -> Result<Vec<RelPath>, Error> {
    let meta_root = root.join(META_DIR);
    let target = meta_root.join(prefix.as_path());
    ensure_within_root(root, &meta_root)?;
    ensure_within_root(root, &target)?;

    let meta = match fs::symlink_metadata(&target) {
        Ok(meta) => meta,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(StoreError::Io(err).into()),
    };

    let ft = meta.file_type();
    if ft.is_symlink() {
        return Err(StoreError::PathOutsideRoot(target).into());
    }
    if ft.is_file() {
        return Ok(vec![prefix.clone()]);
    }
    if !ft.is_dir() {
        return Ok(Vec::new());
    }

    let mut keys = Vec::new();
    let mut stack: Vec<(PathBuf, String)> = vec![(target, prefix.as_str().to_owned())];

    while let Some((dir_path, rel_prefix)) = stack.pop() {
        let read_dir = match fs::read_dir(&dir_path) {
            Ok(rd) => rd,
            Err(err) if err.kind() == io::ErrorKind::NotFound => continue,
            Err(err) => return Err(StoreError::Io(err).into()),
        };

        for entry_res in read_dir {
            let entry = entry_res.map_err(StoreError::from)?;
            let entry_path = entry.path();
            let entry_meta = match fs::symlink_metadata(&entry_path) {
                Ok(meta) => meta,
                Err(err) if err.kind() == io::ErrorKind::NotFound => continue,
                Err(err) => return Err(StoreError::Io(err).into()),
            };
            let entry_ft = entry_meta.file_type();

            if entry_ft.is_symlink() {
                return Err(StoreError::PathOutsideRoot(entry_path).into());
            }

            let name = entry.file_name();
            let Some(name_str) = name.to_str() else {
                return Err(Error::InvalidPath(format!(
                    "non-UTF-8 metadata entry: {}",
                    entry_path.display()
                )));
            };
            let rel_str = format!("{rel_prefix}/{name_str}");

            if entry_ft.is_dir() {
                stack.push((entry_path, rel_str));
            } else if entry_ft.is_file() {
                keys.push(RelPath::parse(&rel_str)?);
            }
        }
    }

    keys.sort();
    Ok(keys)
}

fn delete_meta_sync(root: &Path, key: &RelPath) -> Result<(), Error> {
    let meta_root = root.join(META_DIR);
    let dest = meta_root.join(key.as_path());
    ensure_within_root(root, &meta_root)?;
    ensure_within_root(root, &dest)?;

    let meta = match fs::symlink_metadata(&dest) {
        Ok(meta) => meta,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(err) => return Err(StoreError::Io(err).into()),
    };

    let ft = meta.file_type();
    if ft.is_symlink() {
        return Err(StoreError::PathOutsideRoot(dest).into());
    }
    if ft.is_dir() {
        return Ok(());
    }

    match fs::remove_file(&dest) {
        Ok(()) => {
            prune_empty_parents(&meta_root, dest.parent());
            Ok(())
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(StoreError::Io(err).into()),
    }
}

fn prune_empty_parents(stop_at: &Path, mut current: Option<&Path>) {
    while let Some(dir) = current {
        if dir == stop_at || !dir.starts_with(stop_at) {
            break;
        }
        if fs::remove_dir(dir).is_err() {
            break;
        }
        current = dir.parent();
    }
}

#[cfg(unix)]
fn set_file_readonly(path: &Path) -> Result<(), Error> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o444)).map_err(StoreError::from)?;
    Ok(())
}

#[cfg(not(unix))]
fn set_file_readonly(path: &Path) -> Result<(), Error> {
    let mut perms = fs::metadata(path).map_err(StoreError::from)?.permissions();
    perms.set_readonly(true);
    fs::set_permissions(path, perms).map_err(StoreError::from)?;
    Ok(())
}

#[cfg(unix)]
fn set_dir_readonly(path: &Path) -> Result<(), Error> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o555)).map_err(StoreError::from)?;
    Ok(())
}

#[cfg(not(unix))]
fn set_dir_readonly(_path: &Path) -> Result<(), Error> {
    Ok(())
}

fn set_tree_readonly(dir: &Path) -> Result<(), Error> {
    let mut dirs = vec![dir.to_path_buf()];
    let mut idx = 0;

    while idx < dirs.len() {
        let current = dirs[idx].clone();
        idx += 1;
        for entry_res in fs::read_dir(&current).map_err(StoreError::from)? {
            let entry = entry_res.map_err(StoreError::from)?;
            let path = entry.path();
            let ft = fs::symlink_metadata(&path)
                .map_err(StoreError::from)?
                .file_type();
            if ft.is_dir() {
                dirs.push(path);
            } else if ft.is_file() {
                set_file_readonly(&path)?;
            }
        }
    }

    for dir_path in dirs.into_iter().rev() {
        set_dir_readonly(&dir_path)?;
    }
    Ok(())
}

fn remove_readonly_tree(path: &Path) -> io::Result<()> {
    make_tree_writable(path);
    fs::remove_dir_all(path)
}

#[cfg(unix)]
fn make_tree_writable(path: &Path) {
    use std::os::unix::fs::PermissionsExt;

    let mut stack = vec![path.to_path_buf()];
    while let Some(cur) = stack.pop() {
        let Ok(meta) = fs::symlink_metadata(&cur) else {
            continue;
        };
        let ft = meta.file_type();
        if ft.is_dir() {
            let _ = fs::set_permissions(&cur, fs::Permissions::from_mode(0o755));
            if let Ok(read_dir) = fs::read_dir(&cur) {
                for entry in read_dir.flatten() {
                    stack.push(entry.path());
                }
            }
        } else if ft.is_file() {
            let _ = fs::set_permissions(&cur, fs::Permissions::from_mode(0o644));
        }
    }
}

#[cfg(not(unix))]
fn make_tree_writable(path: &Path) {
    let mut stack = vec![path.to_path_buf()];
    while let Some(cur) = stack.pop() {
        let Ok(meta) = fs::symlink_metadata(&cur) else {
            continue;
        };
        if meta.is_file() {
            let mut perms = meta.permissions();
            perms.set_readonly(false);
            let _ = fs::set_permissions(&cur, perms);
        } else if meta.is_dir() {
            if let Ok(read_dir) = fs::read_dir(&cur) {
                for entry in read_dir.flatten() {
                    stack.push(entry.path());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ArtifactKind;

    /// Guard that restores write permissions before `TempDir` drops so Unix `0555` dirs clean up.
    struct TestStore {
        _dir: tempfile::TempDir,
        store: LocalStore,
    }

    impl TestStore {
        fn new() -> Self {
            let dir = tempfile::tempdir().unwrap();
            let store = LocalStore::new(dir.path()).unwrap();
            Self { _dir: dir, store }
        }
    }

    impl Drop for TestStore {
        fn drop(&mut self) {
            make_tree_writable(self.store.root());
        }
    }

    fn sample_manifest() -> Manifest {
        Manifest::new(ArtifactKind::Blob, "bin")
            .with_metadata_entry("author", serde_json::json!("test"))
    }

    #[tokio::test]
    async fn put_get_exists_verify_round_trip() {
        let ts = TestStore::new();
        let store = &ts.store;

        let missing = ArtifactId::from_bytes([0x42; 32]);
        assert!(!store.exists(&missing).await.unwrap());
        assert!(store.get(&missing).await.unwrap().is_none());
        let err = store.verify(&missing).await.unwrap_err();
        assert!(matches!(err, Error::Store(StoreError::NotFound(id)) if id == missing));

        // Bytes payload
        let manifest = sample_manifest();
        let id = store
            .put(manifest.clone(), PayloadSource::Bytes(b"hello world"))
            .await
            .unwrap();
        assert!(store.exists(&id).await.unwrap());
        let artifact = store.get(&id).await.unwrap().unwrap();
        assert_eq!(artifact.id(), id);
        assert_eq!(artifact.manifest(), &manifest);
        store.verify(&id).await.unwrap();

        // Directory payload
        let src_dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(src_dir.path().join("nested")).unwrap();
        fs::write(src_dir.path().join("a.txt"), b"alpha").unwrap();
        fs::write(src_dir.path().join("nested/b.txt"), b"beta").unwrap();

        let dir_id = store
            .put(sample_manifest(), PayloadSource::Dir(src_dir.path()))
            .await
            .unwrap();
        store.verify(&dir_id).await.unwrap();

        // Single file payload
        let single_file = src_dir.path().join("a.txt");
        let file_id = store
            .put(sample_manifest(), PayloadSource::File(&single_file))
            .await
            .unwrap();
        store.verify(&file_id).await.unwrap();
    }

    #[tokio::test]
    async fn put_deduplicates_sequential_and_concurrent_writes_atomically() {
        let ts = TestStore::new();
        let store = ts.store.clone();
        let manifest = sample_manifest();

        let id1 = store
            .put(manifest.clone(), PayloadSource::Bytes(b"same content"))
            .await
            .unwrap();
        let id2 = store
            .put(manifest.clone(), PayloadSource::Bytes(b"same content"))
            .await
            .unwrap();
        assert_eq!(id1, id2);

        // Concurrent puts of a brand-new artifact must all succeed with the same ID.
        let mut handles = Vec::new();
        for _ in 0..8 {
            let s = store.clone();
            let m = manifest.clone();
            handles.push(tokio::spawn(async move {
                let mut files = BTreeMap::new();
                files.insert(
                    RelPath::parse("part1.bin").unwrap(),
                    b"payload-1".as_slice(),
                );
                files.insert(
                    RelPath::parse("sub/part2.bin").unwrap(),
                    b"payload-2".as_slice(),
                );
                s.put(m, PayloadSource::Files(files)).await
            }));
        }

        let mut ids = Vec::new();
        for handle in handles {
            ids.push(handle.await.unwrap().unwrap());
        }
        let expected = ids[0];
        assert!(ids.iter().all(|id| *id == expected));
        store.verify(&expected).await.unwrap();

        // Staging directory must have no leftover temp entries.
        let tmp_entries: Vec<_> = fs::read_dir(store.root().join(TMP_DIR)).unwrap().collect();
        assert!(
            tmp_entries.is_empty(),
            "expected empty .tmp dir, found {tmp_entries:?}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn put_sets_readonly_permissions_on_unix() {
        use std::os::unix::fs::PermissionsExt;

        let ts = TestStore::new();
        let store = &ts.store;

        let mut files = BTreeMap::new();
        files.insert(RelPath::parse("top.bin").unwrap(), b"top".as_slice());
        files.insert(RelPath::parse("sub/leaf.bin").unwrap(), b"leaf".as_slice());

        let id = store
            .put(sample_manifest(), PayloadSource::Files(files))
            .await
            .unwrap();

        let base = artifact_dir(store.root(), id);
        let mode = |p: PathBuf| fs::symlink_metadata(p).unwrap().permissions().mode() & 0o777;

        assert_eq!(mode(base.clone()), 0o555);
        assert_eq!(mode(base.join(MANIFEST_FILE)), 0o444);
        assert_eq!(mode(base.join(PAYLOAD_DIR)), 0o555);
        assert_eq!(mode(base.join(PAYLOAD_DIR).join("sub")), 0o555);
        assert_eq!(mode(base.join(PAYLOAD_DIR).join("top.bin")), 0o444);
        assert_eq!(mode(base.join(PAYLOAD_DIR).join("sub/leaf.bin")), 0o444);
    }

    #[tokio::test]
    async fn fetch_materializes_isolated_copy() {
        let ts = TestStore::new();
        let store = &ts.store;

        let missing = ArtifactId::from_bytes([0x99; 32]);
        let dest_missing = tempfile::tempdir().unwrap();
        let err = store
            .fetch(&missing, dest_missing.path())
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Store(StoreError::NotFound(id)) if id == missing));

        let mut files = BTreeMap::new();
        files.insert(
            RelPath::parse("weights.bin").unwrap(),
            b"original".as_slice(),
        );
        files.insert(
            RelPath::parse("nested/config.json").unwrap(),
            b"{\"k\":1}".as_slice(),
        );
        let id = store
            .put(sample_manifest(), PayloadSource::Files(files))
            .await
            .unwrap();

        let out_parent = tempfile::tempdir().unwrap();
        let dest = out_parent.path().join("fetched");
        let returned = store.fetch(&id, &dest).await.unwrap();
        assert_eq!(returned, dest);
        assert!(!returned.starts_with(store.root()));

        assert_eq!(fs::read(dest.join("weights.bin")).unwrap(), b"original");
        assert_eq!(
            fs::read(dest.join("nested/config.json")).unwrap(),
            b"{\"k\":1}"
        );

        // Mutating and deleting fetched files must succeed and leave stored artifact intact.
        fs::write(dest.join("weights.bin"), b"tampered-copy").unwrap();
        fs::remove_file(dest.join("nested/config.json")).unwrap();
        store.verify(&id).await.unwrap();

        // A second fetch must still produce the original content.
        let dest2 = out_parent.path().join("fetched2");
        store.fetch(&id, &dest2).await.unwrap();
        assert_eq!(fs::read(dest2.join("weights.bin")).unwrap(), b"original");
    }

    #[tokio::test]
    async fn verify_detects_tampering_on_disk() {
        let ts = TestStore::new();
        let store = &ts.store;

        let id = store
            .put(sample_manifest(), PayloadSource::Bytes(b"clean payload"))
            .await
            .unwrap();
        let dir = artifact_dir(store.root(), id);
        let data_file = dir.join(PAYLOAD_DIR).join("data");

        // 1. Tamper payload file contents.
        make_tree_writable(&dir);
        fs::write(&data_file, b"dirty payload").unwrap();
        let err = store.verify(&id).await.unwrap_err();
        assert!(
            matches!(err, Error::Store(StoreError::Corrupted { id: bad, .. }) if bad == id),
            "expected Corrupted on content tamper, got {err:?}"
        );

        // Restore valid content, then add an extra file.
        fs::write(&data_file, b"clean payload").unwrap();
        store.verify(&id).await.unwrap();
        let extra = dir.join(PAYLOAD_DIR).join("extra.txt");
        fs::write(&extra, b"sneaky").unwrap();
        let err = store.verify(&id).await.unwrap_err();
        assert!(
            matches!(err, Error::Store(StoreError::Corrupted { .. })),
            "expected Corrupted on extra file, got {err:?}"
        );
        fs::remove_file(&extra).unwrap();

        // Plant a symlink inside payload on Unix.
        #[cfg(unix)]
        {
            let link = dir.join(PAYLOAD_DIR).join("sneaky_link");
            std::os::unix::fs::symlink(&data_file, &link).unwrap();
            let err = store.verify(&id).await.unwrap_err();
            assert!(
                matches!(err, Error::Store(StoreError::Corrupted { .. })),
                "expected Corrupted on stored symlink, got {err:?}"
            );
            fs::remove_file(&link).unwrap();
        }

        // Corrupt manifest.json -> both get and verify must return Corrupted.
        fs::write(dir.join(MANIFEST_FILE), b"{not valid json").unwrap();
        let err = store.get(&id).await.unwrap_err();
        assert!(
            matches!(err, Error::Store(StoreError::Corrupted { .. })),
            "expected Corrupted on malformed manifest get, got {err:?}"
        );
        let err = store.verify(&id).await.unwrap_err();
        assert!(
            matches!(err, Error::Store(StoreError::Corrupted { .. })),
            "expected Corrupted on malformed manifest verify, got {err:?}"
        );
    }

    #[tokio::test]
    async fn put_rejects_symlinks_case_collisions_and_unsupported_schema() {
        let ts = TestStore::new();
        let store = &ts.store;

        // Unsupported manifest schema_version
        let mut bad_manifest = sample_manifest();
        bad_manifest.schema_version = 99;
        let err = store
            .put(bad_manifest, PayloadSource::Bytes(b"x"))
            .await
            .unwrap_err();
        assert!(
            matches!(
                err,
                Error::UnsupportedSchemaVersion {
                    found: 99,
                    expected: 1
                }
            ),
            "got {err:?}"
        );

        // Case collision (A.bin vs a.bin)
        let mut colliding = BTreeMap::new();
        colliding.insert(RelPath::parse("A.bin").unwrap(), b"1".as_slice());
        colliding.insert(RelPath::parse("a.bin").unwrap(), b"2".as_slice());
        let err = store
            .put(sample_manifest(), PayloadSource::Files(colliding))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::InvalidPath(_)), "got {err:?}");

        // Symlinks (internal and external) on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;

            let outside = tempfile::tempdir().unwrap();
            let secret = outside.path().join("secret.txt");
            fs::write(&secret, b"secret").unwrap();

            let dir = tempfile::tempdir().unwrap();
            let internal_target = dir.path().join("real.txt");
            fs::write(&internal_target, b"real").unwrap();
            symlink(&internal_target, dir.path().join("internal_link.txt")).unwrap();

            let err = store
                .put(sample_manifest(), PayloadSource::Dir(dir.path()))
                .await
                .unwrap_err();
            assert!(
                matches!(err, Error::Store(StoreError::InvalidPayload(_))),
                "expected InvalidPayload for internal symlink, got {err:?}"
            );

            let dir_ext = tempfile::tempdir().unwrap();
            symlink(&secret, dir_ext.path().join("escape.txt")).unwrap();
            let err = store
                .put(sample_manifest(), PayloadSource::Dir(dir_ext.path()))
                .await
                .unwrap_err();
            assert!(
                matches!(err, Error::Store(StoreError::InvalidPayload(_))),
                "expected InvalidPayload for external symlink, got {err:?}"
            );
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn root_boundary_rejects_symlinks_escaping_store_root() {
        use std::os::unix::fs::symlink;

        let ts = TestStore::new();
        let store = &ts.store;
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("target.txt"), b"outside").unwrap();

        // Plant a symlink under meta/ pointing outside store root.
        let meta_root = store.root().join(META_DIR);
        fs::create_dir_all(&meta_root).unwrap();
        symlink(outside.path(), meta_root.join("escape")).unwrap();

        let key = RelPath::parse("escape/target.txt").unwrap();
        let err = store.put_meta(&key, b"overwrite").await.unwrap_err();
        assert!(
            matches!(err, Error::Store(StoreError::PathOutsideRoot(_))),
            "expected PathOutsideRoot on put_meta, got {err:?}"
        );
        assert_eq!(
            fs::read(outside.path().join("target.txt")).unwrap(),
            b"outside"
        );

        let err = store.get_meta(&key).await.unwrap_err();
        assert!(matches!(err, Error::Store(StoreError::PathOutsideRoot(_))));

        let prefix = RelPath::parse("escape").unwrap();
        let err = store.list_meta(&prefix).await.unwrap_err();
        assert!(matches!(err, Error::Store(StoreError::PathOutsideRoot(_))));

        let err = store.delete_meta(&key).await.unwrap_err();
        assert!(matches!(err, Error::Store(StoreError::PathOutsideRoot(_))));

        // Plant a symlink under artifacts/<id> pointing outside store root.
        let id = ArtifactId::from_bytes([0xab; 32]);
        let artifacts_root = store.root().join(ARTIFACTS_DIR);
        fs::create_dir_all(&artifacts_root).unwrap();
        symlink(outside.path(), artifact_dir(store.root(), id)).unwrap();

        let err = store.get(&id).await.unwrap_err();
        assert!(matches!(err, Error::Store(StoreError::PathOutsideRoot(_))));
        let err = store.verify(&id).await.unwrap_err();
        assert!(matches!(err, Error::Store(StoreError::PathOutsideRoot(_))));
        let err = store.exists(&id).await.unwrap_err();
        assert!(matches!(err, Error::Store(StoreError::PathOutsideRoot(_))));
    }

    #[tokio::test]
    async fn meta_crud_and_segment_boundary_listing() {
        let ts = TestStore::new();
        let store = &ts.store;

        let k1 = RelPath::parse("cache/a.json").unwrap();
        let k2 = RelPath::parse("cache/sub/b.json").unwrap();
        let k3 = RelPath::parse("cache2/c.json").unwrap();
        let lineage = RelPath::parse("runs/01J/lineage.json").unwrap();

        assert!(store.get_meta(&k1).await.unwrap().is_none());
        store.delete_meta(&k1).await.unwrap(); // idempotent on absent key

        store.put_meta(&k1, br#"{"v":1}"#).await.unwrap();
        store.put_meta(&k2, br#"{"v":2}"#).await.unwrap();
        store.put_meta(&k3, br#"{"v":3}"#).await.unwrap();
        store.put_meta(&lineage, br#"{"events":[]}"#).await.unwrap();

        assert_eq!(store.get_meta(&k1).await.unwrap().unwrap(), br#"{"v":1}"#);
        // Intermediate directory is not a metadata value.
        let cache_prefix = RelPath::parse("cache").unwrap();
        assert!(store.get_meta(&cache_prefix).await.unwrap().is_none());

        // Atomic overwrite (last-write-wins).
        store.put_meta(&k1, br#"{"v":10}"#).await.unwrap();
        assert_eq!(store.get_meta(&k1).await.unwrap().unwrap(), br#"{"v":10}"#);

        // Segment-boundary listing (excludes cache2/c.json).
        let listed = store.list_meta(&cache_prefix).await.unwrap();
        assert_eq!(listed, vec![k1.clone(), k2.clone()]);

        // Exact file prefix listing.
        assert_eq!(
            store.list_meta(&lineage).await.unwrap(),
            vec![lineage.clone()]
        );

        // Delete and verify empty parent pruning allows reusing parent path.
        store.delete_meta(&lineage).await.unwrap();
        assert!(store.get_meta(&lineage).await.unwrap().is_none());
        let runs_key = RelPath::parse("runs").unwrap();
        store.put_meta(&runs_key, b"reused").await.unwrap();
        assert_eq!(store.get_meta(&runs_key).await.unwrap().unwrap(), b"reused");
    }
}
