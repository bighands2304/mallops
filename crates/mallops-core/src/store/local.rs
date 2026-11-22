//! Local filesystem-backed [`ArtifactStore`] implementation.
//!
//! Layout under the canonicalized store root (default [`LocalStore::DEFAULT_ROOT`]):
//! - `artifacts/<artifact-id>/manifest.json` — canonical JSON manifest (`0444` on Unix)
//! - `artifacts/<artifact-id>/payload/…` — read-only payload files (`0444` files / `0555` dirs on Unix)
//! - `meta/<rel-path>` — mutable auxiliary metadata (cache entries, run lineage, etc.)
//! - `.tmp/` — same-filesystem staging directory for atomic writes

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};

use crate::store::payload::{self, PayloadEntry, WalkEntry};
use crate::{
    Artifact, ArtifactId, ArtifactStore, BoxFuture, Digest, Error, Manifest, PayloadFile,
    PayloadSource, RelPath, StoreError,
};

const ARTIFACTS_DIR: &str = "artifacts";
const META_DIR: &str = "meta";
const TMP_DIR: &str = ".tmp";
const MANIFEST_FILE: &str = "manifest.json";
const PAYLOAD_DIR: &str = "payload";

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

impl LocalStore {
    fn blocking<'a, T, F>(&'a self, f: F) -> BoxFuture<'a, Result<T, Error>>
    where
        T: Send + 'static,
        F: FnOnce(&Path) -> Result<T, Error> + Send + 'static,
    {
        let root = self.root.clone();
        Box::pin(async move {
            tokio::task::spawn_blocking(move || f(&root))
                .await
                .map_err(|err| {
                    Error::Store(StoreError::Backend(format!(
                        "blocking store task failed: {err}"
                    )))
                })?
        })
    }
}

impl ArtifactStore for LocalStore {
    fn put<'a>(
        &'a self,
        manifest: Manifest,
        payload: PayloadSource<'a>,
    ) -> BoxFuture<'a, Result<ArtifactId, Error>> {
        let owned = OwnedPayloadSource::from_borrowed(payload);
        self.blocking(move |root| put_sync(root, manifest, owned))
    }

    fn get<'a>(&'a self, id: &'a ArtifactId) -> BoxFuture<'a, Result<Option<Artifact>, Error>> {
        let id = *id;
        self.blocking(move |root| get_sync(root, id))
    }

    fn fetch<'a>(
        &'a self,
        id: &'a ArtifactId,
        dest: &'a Path,
    ) -> BoxFuture<'a, Result<PathBuf, Error>> {
        let id = *id;
        let dest = dest.to_path_buf();
        self.blocking(move |root| fetch_sync(root, id, &dest))
    }

    fn verify<'a>(&'a self, id: &'a ArtifactId) -> BoxFuture<'a, Result<(), Error>> {
        let id = *id;
        self.blocking(move |root| verify_sync(root, id))
    }

    fn exists<'a>(&'a self, id: &'a ArtifactId) -> BoxFuture<'a, Result<bool, Error>> {
        let id = *id;
        self.blocking(move |root| exists_sync(root, id))
    }

    fn put_meta<'a>(
        &'a self,
        key: &'a RelPath,
        data: &'a [u8],
    ) -> BoxFuture<'a, Result<(), Error>> {
        let key = key.clone();
        let data = data.to_vec();
        self.blocking(move |root| put_meta_sync(root, &key, &data))
    }

    fn get_meta<'a>(&'a self, key: &'a RelPath) -> BoxFuture<'a, Result<Option<Vec<u8>>, Error>> {
        let key = key.clone();
        self.blocking(move |root| get_meta_sync(root, &key))
    }

    fn list_meta<'a>(&'a self, prefix: &'a RelPath) -> BoxFuture<'a, Result<Vec<RelPath>, Error>> {
        let prefix = prefix.clone();
        self.blocking(move |root| list_meta_sync(root, &prefix))
    }

    fn delete_meta<'a>(&'a self, key: &'a RelPath) -> BoxFuture<'a, Result<(), Error>> {
        let key = key.clone();
        self.blocking(move |root| delete_meta_sync(root, &key))
    }
}

/// Ensures `candidate` stays lexically inside `root` without traversing symlinks.
///
/// Because `root` is canonicalized on construction and each existing component is checked via
/// `lstat`, this protects against planted or accidental symlinks inside the store. It is a
/// check-then-use guard and does not protect against a concurrent attacker with write access to
/// the store directory (outside the v0.1 threat model, ADR-0007 D7).
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
                let ft = meta.file_type();
                if ft.is_symlink() {
                    return Err(StoreError::PathOutsideRoot(cur).into());
                }
                if !ft.is_dir() {
                    break;
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
    fs::create_dir_all(dir).map_err(StoreError::from)?;
    ensure_within_root(root, dir)
}

fn stat_artifact_dir(root: &Path, id: ArtifactId) -> Result<Option<PathBuf>, Error> {
    let dir = artifact_dir(root, id);
    ensure_within_root(root, &dir)?;

    match fs::symlink_metadata(&dir) {
        Ok(meta) if meta.file_type().is_dir() => Ok(Some(dir)),
        Ok(_) => Err(StoreError::Corrupted {
            id,
            reason: "stored artifact entry is not a directory".to_owned(),
        }
        .into()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(StoreError::Io(err).into()),
    }
}

fn check_payload_dir(artifact_dir: &Path, id: ArtifactId) -> Result<PathBuf, Error> {
    let payload_dir = artifact_dir.join(PAYLOAD_DIR);
    let meta = fs::symlink_metadata(&payload_dir).map_err(|err| StoreError::Corrupted {
        id,
        reason: format!("missing or unreadable payload directory: {err}"),
    })?;
    if meta.file_type().is_symlink() || !meta.file_type().is_dir() {
        return Err(StoreError::Corrupted {
            id,
            reason: "stored payload entry is not a regular directory".to_owned(),
        }
        .into());
    }
    Ok(payload_dir)
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

    for (rel, entry) in &entries {
        let dest_path = payload_stage.join(rel.as_path());
        if let Some(parent) = dest_path.parent() {
            fs::create_dir_all(parent).map_err(StoreError::from)?;
        }

        let (len, digest) = match entry {
            PayloadEntry::Bytes(bytes) => {
                let mut file = File::create(&dest_path).map_err(StoreError::from)?;
                file.write_all(bytes).map_err(StoreError::from)?;
                file.sync_all().map_err(StoreError::from)?;
                (bytes.len() as u64, Digest::hash(bytes))
            }
            PayloadEntry::File { .. } => copy_and_hash_entry(entry, &dest_path)?,
        };

        payload_files.push(PayloadFile::new(rel.clone(), len, digest));
    }

    let id = ArtifactId::compute(&manifest, &payload_files)?;
    let manifest_bytes = manifest.canonical_json();
    let manifest_stage = staging.path().join(MANIFEST_FILE);
    let mut manifest_file = File::create(&manifest_stage).map_err(StoreError::from)?;
    manifest_file
        .write_all(&manifest_bytes)
        .map_err(StoreError::from)?;
    manifest_file.sync_all().map_err(StoreError::from)?;

    let final_dir = artifact_dir(root, id);
    ensure_within_root(root, &final_dir)?;

    // Fast-path dedup before locking permissions and renaming.
    if fs::symlink_metadata(&final_dir).is_ok_and(|meta| meta.file_type().is_dir()) {
        return Ok(id);
    }

    // Take ownership of staging_path so read-only cleanup always runs if publishing fails or loses a race.
    let staging_path = staging.keep();
    let publish = (|| -> Result<ArtifactId, Error> {
        set_file_readonly(&staging_path.join(MANIFEST_FILE))?;
        set_tree_readonly(&staging_path.join(PAYLOAD_DIR))?;

        match fs::rename(&staging_path, &final_dir) {
            Ok(()) => {
                #[cfg(unix)]
                File::open(&artifacts_root)
                    .and_then(|dir| dir.sync_all())
                    .map_err(StoreError::from)?;
                // ponytail: best-effort chmod after publish; the artifact is already atomically
                // live, so failing here would abort a succeeded put(). Must run after rename
                // because moving a directory updates its `..` entry.
                let _ = set_dir_readonly(&final_dir);
                Ok(id)
            }
            Err(err) => {
                if fs::symlink_metadata(&final_dir).is_ok_and(|meta| meta.file_type().is_dir()) {
                    Ok(id)
                } else {
                    Err(StoreError::Io(err).into())
                }
            }
        }
    })();

    if publish.is_err() || fs::symlink_metadata(&staging_path).is_ok() {
        let _ = remove_readonly_tree(&staging_path);
    }
    publish
}

struct HashingWriter<W> {
    inner: W,
    hasher: blake3::Hasher,
}

impl<W: Write> Write for HashingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

fn copy_and_hash_entry(entry: &PayloadEntry<'_>, dest: &Path) -> Result<(u64, Digest), Error> {
    let mut reader = entry.open_verified()?;
    let mut writer = HashingWriter {
        inner: File::create(dest).map_err(StoreError::from)?,
        hasher: blake3::Hasher::new(),
    };
    let total_len = io::copy(&mut reader, &mut writer).map_err(StoreError::from)?;
    writer.inner.sync_all().map_err(StoreError::from)?;
    Ok((
        total_len,
        Digest::from_bytes(*writer.hasher.finalize().as_bytes()),
    ))
}

fn hash_file_streaming(path: &Path) -> Result<(u64, Digest), io::Error> {
    let mut reader = File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update_reader(&mut reader)?;
    Ok((
        hasher.count(),
        Digest::from_bytes(*hasher.finalize().as_bytes()),
    ))
}

fn get_sync(root: &Path, id: ArtifactId) -> Result<Option<Artifact>, Error> {
    let Some(dir) = stat_artifact_dir(root, id)? else {
        return Ok(None);
    };
    let manifest = read_stored_manifest(&dir, id)?;
    Ok(Some(Artifact::new(id, manifest)))
}

fn exists_sync(root: &Path, id: ArtifactId) -> Result<bool, Error> {
    Ok(stat_artifact_dir(root, id)?.is_some())
}

/// Rejects a `fetch` destination that resolves inside `root` (including non-existent paths whose
/// nearest existing ancestor canonicalizes inside `root`).
fn ensure_dest_outside_root(root: &Path, dest: &Path) -> Result<(), Error> {
    let abs = if dest.is_absolute() {
        dest.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(StoreError::from)?
            .join(dest)
    };

    let mut existing = abs.as_path();
    let mut tail = Vec::new();
    let mut resolved = loop {
        match fs::canonicalize(existing) {
            Ok(canon) => break canon,
            Err(err) if err.kind() == io::ErrorKind::NotFound => {
                let Some(last) = existing.components().next_back() else {
                    return Err(StoreError::Io(err).into());
                };
                tail.push(last.as_os_str().to_os_string());
                let Some(parent) = existing.parent() else {
                    return Err(StoreError::Io(err).into());
                };
                existing = parent;
            }
            Err(err) => return Err(StoreError::Io(err).into()),
        }
    };

    if resolved.starts_with(root) {
        return Err(StoreError::InvalidPayload(format!(
            "fetch destination must not be inside store root: {}",
            dest.display()
        ))
        .into());
    }

    for part in tail.into_iter().rev() {
        if part == ".." {
            resolved.pop();
        } else if part != "." {
            resolved.push(part);
        }
        if resolved.starts_with(root) {
            return Err(StoreError::InvalidPayload(format!(
                "fetch destination must not be inside store root: {}",
                dest.display()
            ))
            .into());
        }
    }

    Ok(())
}

fn fetch_sync(root: &Path, id: ArtifactId, dest: &Path) -> Result<PathBuf, Error> {
    ensure_dest_outside_root(root, dest)?;

    let dir = stat_artifact_dir(root, id)?.ok_or(StoreError::NotFound(id))?;
    let payload_dir = check_payload_dir(&dir, id)?;
    let stored_files = walk_stored_payload(&payload_dir, id)?;

    payload::materialize(
        dest,
        stored_files.into_iter().map(|(rel, src_path)| {
            let file = File::open(&src_path).map_err(StoreError::from)?;
            Ok((rel, file))
        }),
    )?;

    Ok(dest.to_path_buf())
}

fn verify_sync(root: &Path, id: ArtifactId) -> Result<(), Error> {
    let dir = stat_artifact_dir(root, id)?.ok_or(StoreError::NotFound(id))?;

    // Reject missing or unexpected extra entries at the artifact root directory.
    let expected = BTreeSet::from([OsString::from(MANIFEST_FILE), OsString::from(PAYLOAD_DIR)]);
    let actual: BTreeSet<OsString> = fs::read_dir(&dir)
        .and_then(|rd| rd.map(|e| e.map(|entry| entry.file_name())).collect())
        .map_err(|err| StoreError::Corrupted {
            id,
            reason: format!("failed to read artifact directory: {err}"),
        })?;
    if actual != expected {
        return Err(StoreError::Corrupted {
            id,
            reason: "artifact directory must contain exactly manifest.json and payload/".to_owned(),
        }
        .into());
    }

    let manifest = read_stored_manifest(&dir, id)?;
    let payload_dir = check_payload_dir(&dir, id)?;
    let stored_files = walk_stored_payload(&payload_dir, id)?;
    let mut payload_files = Vec::with_capacity(stored_files.len());

    for (rel, path) in stored_files {
        let (len, digest) = hash_file_streaming(&path).map_err(|err| StoreError::Corrupted {
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
    let entries = payload::walk_files(payload_dir, false).map_err(|err| StoreError::Corrupted {
        id,
        reason: format!("failed to walk payload directory: {err}"),
    })?;

    let mut files = BTreeMap::new();
    for WalkEntry { path, meta } in entries {
        let ft = meta.file_type();
        if ft.is_symlink() {
            return Err(StoreError::Corrupted {
                id,
                reason: format!("unexpected symlink in stored payload: {}", path.display()),
            }
            .into());
        }
        if !ft.is_file() && !ft.is_dir() {
            return Err(StoreError::Corrupted {
                id,
                reason: format!(
                    "unexpected special file in stored payload: {}",
                    path.display()
                ),
            }
            .into());
        }
        let rel = payload::rel_path_from_root(payload_dir, &path).map_err(|err| {
            StoreError::Corrupted {
                id,
                reason: format!("invalid stored payload path '{}': {err}", path.display()),
            }
        })?;
        if ft.is_file() {
            files.insert(rel, path);
        }
    }

    Ok(files)
}

fn put_meta_sync(root: &Path, key: &RelPath, data: &[u8]) -> Result<(), Error> {
    let tmp_root = root.join(TMP_DIR);
    let meta_root = root.join(META_DIR);
    ensure_dir_within_root(root, &tmp_root)?;

    let dest = meta_root.join(key.as_path());
    ensure_within_root(root, &dest)?;

    for ancestor in dest.ancestors().skip(1) {
        if ancestor == meta_root || !ancestor.starts_with(&meta_root) {
            break;
        }
        if fs::symlink_metadata(ancestor).is_ok_and(|meta| meta.file_type().is_file()) {
            return Err(Error::InvalidPath(format!(
                "metadata key '{key}' conflicts with an existing key prefix"
            )));
        }
    }
    if fs::symlink_metadata(&dest).is_ok_and(|meta| meta.file_type().is_dir()) {
        return Err(Error::InvalidPath(format!(
            "metadata key '{key}' conflicts with an existing directory prefix"
        )));
    }

    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            if matches!(
                err.kind(),
                io::ErrorKind::AlreadyExists | io::ErrorKind::NotADirectory
            ) {
                Error::InvalidPath(format!(
                    "metadata key '{key}' conflicts with an existing key prefix"
                ))
            } else {
                StoreError::Io(err).into()
            }
        })?;
    }

    let mut tmp = tempfile::Builder::new()
        .prefix("meta-")
        .tempfile_in(&tmp_root)
        .map_err(StoreError::from)?;
    tmp.write_all(data).map_err(StoreError::from)?;
    tmp.flush().map_err(StoreError::from)?;

    tmp.persist(&dest).map_err(|err| {
        if err.error.kind() == io::ErrorKind::IsADirectory
            || fs::symlink_metadata(&dest).is_ok_and(|meta| meta.file_type().is_dir())
        {
            Error::InvalidPath(format!(
                "metadata key '{key}' conflicts with an existing directory prefix"
            ))
        } else {
            StoreError::Io(err.error).into()
        }
    })?;
    Ok(())
}

fn get_meta_sync(root: &Path, key: &RelPath) -> Result<Option<Vec<u8>>, Error> {
    let dest = root.join(META_DIR).join(key.as_path());
    ensure_within_root(root, &dest)?;

    match fs::read(&dest) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(err)
            if matches!(
                err.kind(),
                io::ErrorKind::NotFound
                    | io::ErrorKind::IsADirectory
                    | io::ErrorKind::NotADirectory
            ) || fs::symlink_metadata(&dest).is_ok_and(|m| m.file_type().is_dir()) =>
        {
            Ok(None)
        }
        Err(err) => Err(StoreError::Io(err).into()),
    }
}

fn list_meta_sync(root: &Path, prefix: &RelPath) -> Result<Vec<RelPath>, Error> {
    let meta_root = root.join(META_DIR);
    let target = meta_root.join(prefix.as_path());
    ensure_within_root(root, &target)?;

    let meta = match fs::symlink_metadata(&target) {
        Ok(meta) => meta,
        Err(err)
            if matches!(
                err.kind(),
                io::ErrorKind::NotFound | io::ErrorKind::NotADirectory
            ) =>
        {
            return Ok(Vec::new());
        }
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
    for WalkEntry { path, meta } in payload::walk_files(&target, true).map_err(StoreError::from)? {
        let ft = meta.file_type();
        if ft.is_symlink() {
            return Err(StoreError::PathOutsideRoot(path).into());
        }
        if ft.is_file() {
            keys.push(payload::rel_path_from_root(&meta_root, &path)?);
        }
    }

    keys.sort();
    Ok(keys)
}

fn delete_meta_sync(root: &Path, key: &RelPath) -> Result<(), Error> {
    let dest = root.join(META_DIR).join(key.as_path());
    ensure_within_root(root, &dest)?;

    match fs::remove_file(&dest) {
        Ok(()) => Ok(()),
        Err(err)
            if matches!(
                err.kind(),
                io::ErrorKind::NotFound
                    | io::ErrorKind::IsADirectory
                    | io::ErrorKind::NotADirectory
            ) || fs::symlink_metadata(&dest).is_ok_and(|m| m.file_type().is_dir()) =>
        {
            Ok(())
        }
        Err(err) => Err(StoreError::Io(err).into()),
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
    for WalkEntry { path, meta } in payload::walk_files(dir, false).map_err(StoreError::from)? {
        let ft = meta.file_type();
        if ft.is_dir() {
            dirs.push(path);
        } else if ft.is_file() {
            set_file_readonly(&path)?;
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

        fn into_parts(self) -> (LocalStore, Self) {
            (self.store.clone(), self)
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

        // Restore valid content, then add an extra file in payload/ and at artifact root.
        fs::write(&data_file, b"clean payload").unwrap();
        store.verify(&id).await.unwrap();
        let extra = dir.join(PAYLOAD_DIR).join("extra.txt");
        fs::write(&extra, b"sneaky").unwrap();
        let err = store.verify(&id).await.unwrap_err();
        assert!(
            matches!(err, Error::Store(StoreError::Corrupted { .. })),
            "expected Corrupted on extra payload file, got {err:?}"
        );
        fs::remove_file(&extra).unwrap();

        let extra_root = dir.join("extra_root");
        fs::write(&extra_root, b"sneaky").unwrap();
        let err = store.verify(&id).await.unwrap_err();
        assert!(
            matches!(err, Error::Store(StoreError::Corrupted { .. })),
            "expected Corrupted on extra artifact root entry, got {err:?}"
        );
        fs::remove_file(&extra_root).unwrap();

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
    async fn fetch_rejects_dest_inside_store_root() {
        let ts = TestStore::new();
        let store = &ts.store;
        let id = store
            .put(sample_manifest(), PayloadSource::Bytes(b"payload"))
            .await
            .unwrap();

        for dest in [
            store.root().to_path_buf(),
            store.root().join(ARTIFACTS_DIR).join("injected"),
            store.root().join("new_dir/nested"),
        ] {
            let err = store.fetch(&id, &dest).await.unwrap_err();
            assert!(
                matches!(err, Error::Store(StoreError::InvalidPayload(_))),
                "expected InvalidPayload for dest {}, got {err:?}",
                dest.display()
            );
        }

        #[cfg(unix)]
        {
            let outside = tempfile::tempdir().unwrap();
            let link_to_root = outside.path().join("into_store");
            std::os::unix::fs::symlink(store.root(), &link_to_root).unwrap();
            let err = store
                .fetch(&id, &link_to_root.join("artifacts/fake"))
                .await
                .unwrap_err();
            assert!(
                matches!(err, Error::Store(StoreError::InvalidPayload(_))),
                "expected InvalidPayload for symlinked dest inside store root, got {err:?}"
            );
        }
    }

    mod conformance {
        use super::*;

        crate::store::conformance::conformance_tests!(|| TestStore::new().into_parts());
    }
}
