//! Shared synchronous payload collection and validation for [`ArtifactStore`](super::ArtifactStore)
//! implementations.
//!
//! Callers in async contexts should invoke [`collect`] inside `tokio::task::spawn_blocking`.

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};

use crate::{Error, Manifest, PayloadSource, RelPath, StoreError};

/// A single validated payload entry ready for ingestion into a store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PayloadEntry<'a> {
    /// A regular file on the local filesystem (verified non-symlink at collection time).
    File {
        path: PathBuf,
        #[cfg(unix)]
        dev: u64,
        #[cfg(unix)]
        ino: u64,
    },
    /// An in-memory byte slice.
    Bytes(&'a [u8]),
}

impl PayloadEntry<'_> {
    fn from_file(path: PathBuf, _meta: &fs::Metadata) -> Self {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            Self::File {
                path,
                dev: _meta.dev(),
                ino: _meta.ino(),
            }
        }
        #[cfg(not(unix))]
        {
            Self::File { path }
        }
    }

    /// Opens a collected file entry and verifies via `fstat` that the open descriptor still
    /// refers to a regular file with the same `(dev, ino)` observed during collection.
    ///
    /// ponytail: residual TOCTOU risks accepted for v0.1 without extra dependencies:
    /// (a) non-Unix targets do not expose a stable `(dev, ino)` pair in `std`;
    /// (b) a file swapped for a FIFO between `collect` and `open` can still block `File::open`,
    /// since passing `O_NONBLOCK` portably would require `libc` (not in the core allowlist);
    /// (c) swapping an intermediate directory *during* the walk (between `lstat` and `read_dir`)
    /// is not detected: it needs a same-user writer still active while the put commits (trusted,
    /// ADR-0007 D7). Full closure needs `openat(O_NOFOLLOW)` traversal, deferred to v0.2.
    pub(crate) fn open_verified(&self) -> Result<fs::File, Error> {
        let Self::File {
            path,
            #[cfg(unix)]
            dev,
            #[cfg(unix)]
            ino,
        } = self
        else {
            return Err(StoreError::InvalidPayload(
                "cannot open an in-memory payload entry as a file".to_owned(),
            )
            .into());
        };

        let file = fs::File::open(path).map_err(StoreError::from)?;
        let meta = file.metadata().map_err(StoreError::from)?;
        if !meta.file_type().is_file() {
            return Err(StoreError::InvalidPayload(format!(
                "payload source changed to non-regular file: {}",
                path.display()
            ))
            .into());
        }

        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if meta.dev() != *dev || meta.ino() != *ino {
                return Err(StoreError::InvalidPayload(format!(
                    "payload source changed after inspection: {}",
                    path.display()
                ))
                .into());
            }
        }

        Ok(file)
    }
}

/// Rejects a manifest whose `schema_version` is not supported (shared by every store's `put`).
pub(crate) fn check_schema(manifest: &Manifest) -> Result<(), Error> {
    if manifest.schema_version != Manifest::SCHEMA_VERSION {
        return Err(Error::UnsupportedSchemaVersion {
            found: manifest.schema_version,
            expected: Manifest::SCHEMA_VERSION,
        });
    }
    Ok(())
}

/// A single filesystem entry produced by [`walk_files`].
pub(crate) struct WalkEntry {
    pub(crate) path: PathBuf,
    pub(crate) meta: fs::Metadata,
}

/// Iteratively walks `root` with [`fs::symlink_metadata`] (never following symlinks).
///
/// Returns every entry discovered under `root` (excluding `root` itself) in parent-before-child
/// order. When `ignore_not_found` is `true`, entries or subdirectories that vanish concurrently
/// during the walk are skipped instead of failing.
pub(crate) fn walk_files(root: &Path, ignore_not_found: bool) -> io::Result<Vec<WalkEntry>> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let read_dir = match fs::read_dir(&dir) {
            Ok(rd) => rd,
            Err(err) if ignore_not_found && err.kind() == io::ErrorKind::NotFound => continue,
            Err(err) => return Err(err),
        };

        for entry_res in read_dir {
            let entry = match entry_res {
                Ok(e) => e,
                Err(err) if ignore_not_found && err.kind() == io::ErrorKind::NotFound => continue,
                Err(err) => return Err(err),
            };
            let path = entry.path();
            let meta = match fs::symlink_metadata(&path) {
                Ok(m) => m,
                Err(err) if ignore_not_found && err.kind() == io::ErrorKind::NotFound => continue,
                Err(err) => return Err(err),
            };
            let is_dir = meta.file_type().is_dir();
            out.push(WalkEntry {
                path: path.clone(),
                meta,
            });
            if is_dir {
                stack.push(path);
            }
        }
    }

    Ok(out)
}

/// Materializes payload entries into `dest`, which must either not exist or be an empty directory.
///
/// Subdirectories are created step-by-step with [`fs::create_dir`] and files are opened with
/// [`fs::OpenOptions::create_new`] (`O_EXCL`) so pre-existing files or planted symlinks cannot be
/// followed or overwritten.
///
/// This protects against content planted in `dest` *before* the call, not against a concurrent
/// writer in `dest`. On error `dest` may be left partially populated: retry with a new directory.
pub(crate) fn materialize<R, I>(dest: &Path, entries: I) -> Result<(), Error>
where
    R: io::Read,
    I: IntoIterator<Item = Result<(RelPath, R), Error>>,
{
    match fs::symlink_metadata(dest) {
        Ok(meta) => {
            let ft = meta.file_type();
            if ft.is_symlink() {
                return Err(StoreError::InvalidPayload(format!(
                    "fetch destination must not be a symlink: {}",
                    dest.display()
                ))
                .into());
            }
            if !ft.is_dir() {
                return Err(StoreError::InvalidPayload(format!(
                    "fetch destination must be a directory: {}",
                    dest.display()
                ))
                .into());
            }
            if fs::read_dir(dest)
                .map_err(StoreError::from)?
                .next()
                .is_some()
            {
                return Err(StoreError::InvalidPayload(format!(
                    "fetch destination must be empty: {}",
                    dest.display()
                ))
                .into());
            }
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            fs::create_dir_all(dest).map_err(StoreError::from)?;
        }
        Err(err) => return Err(StoreError::Io(err).into()),
    }

    let mut created_dirs: HashSet<PathBuf> = HashSet::new();

    for entry_res in entries {
        let (rel, mut reader) = entry_res?;
        let mut cur = dest.to_path_buf();
        if let Some(parent) = rel.as_path().parent() {
            for comp in parent.components() {
                if let Component::Normal(part) = comp {
                    cur.push(part);
                    if created_dirs.insert(cur.clone()) {
                        fs::create_dir(&cur).map_err(StoreError::from)?;
                    }
                }
            }
        }
        if let Some(file_name) = rel.as_path().file_name() {
            cur.push(file_name);
        }

        let mut writer = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&cur)
            .map_err(StoreError::from)?;
        io::copy(&mut reader, &mut writer).map_err(StoreError::from)?;
        writer.flush().map_err(StoreError::from)?;
    }

    Ok(())
}

/// Converts `path` under `root` into a validated [`RelPath`].
pub(crate) fn rel_path_from_root(root: &Path, path: &Path) -> Result<RelPath, Error> {
    let rel = path.strip_prefix(root).map_err(|_| {
        Error::InvalidPath(format!(
            "path '{}' is not under '{}'",
            path.display(),
            root.display()
        ))
    })?;
    let mut joined = String::new();
    for comp in rel.components() {
        let Component::Normal(part) = comp else {
            return Err(Error::InvalidPath(format!(
                "unexpected path component in '{}'",
                path.display()
            )));
        };
        let part_str = part.to_str().ok_or_else(|| {
            Error::InvalidPath(format!("non-UTF-8 path component: {}", path.display()))
        })?;
        if !joined.is_empty() {
            joined.push('/');
        }
        joined.push_str(part_str);
    }
    RelPath::parse(&joined)
}

/// Collects and validates all files described by `payload` into a deterministic map keyed by
/// [`RelPath`].
///
/// # Validation
///
/// - Walks [`PayloadSource::File`] and [`PayloadSource::Dir`] using [`fs::symlink_metadata`].
/// - Rejects symlinks (top-level or nested, internal or external) and special files with
///   [`StoreError::InvalidPayload`].
/// - Validates every relative path as a [`RelPath`] ([`Error::InvalidPath`]).
/// - Rejects payloads whose paths collide after ASCII case folding (e.g. `A.bin` vs `a.bin`,
///   or `Dir/a.bin` vs `dir/b.bin`) or conflict as both file and directory prefix with
///   [`Error::InvalidPath`]. Unicode normalization collisions (NFC/NFD) are not detected in v0.1.
pub(crate) fn collect<'a>(
    payload: PayloadSource<'a>,
) -> Result<BTreeMap<RelPath, PayloadEntry<'a>>, Error> {
    let mut entries = BTreeMap::new();

    match payload {
        PayloadSource::Bytes(bytes) => {
            let rel = RelPath::parse(PayloadSource::DEFAULT_BYTES_PATH)?;
            entries.insert(rel, PayloadEntry::Bytes(bytes));
        }
        PayloadSource::Files(files) => {
            for (rel, bytes) in files {
                entries.insert(rel, PayloadEntry::Bytes(bytes));
            }
        }
        PayloadSource::File(path) => {
            collect_single_file(path, &mut entries)?;
        }
        PayloadSource::Dir(root) => {
            collect_dir(root, &mut entries)?;
        }
    }

    check_path_collisions(&entries)?;
    Ok(entries)
}

fn regular_file(path: &Path) -> Result<fs::Metadata, Error> {
    let meta = fs::symlink_metadata(path).map_err(StoreError::from)?;
    let ft = meta.file_type();
    if ft.is_symlink() {
        return Err(StoreError::InvalidPayload(format!(
            "payload file must not be a symlink: {}",
            path.display()
        ))
        .into());
    }
    if !ft.is_file() {
        return Err(StoreError::InvalidPayload(format!(
            "payload file must be a regular file: {}",
            path.display()
        ))
        .into());
    }
    Ok(meta)
}

fn collect_single_file<'a>(
    path: &Path,
    entries: &mut BTreeMap<RelPath, PayloadEntry<'a>>,
) -> Result<(), Error> {
    let meta = regular_file(path)?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            Error::InvalidPath(format!(
                "payload file has missing or non-UTF-8 file name: {}",
                path.display()
            ))
        })?;
    let rel = RelPath::parse(file_name)?;
    entries.insert(rel, PayloadEntry::from_file(path.to_path_buf(), &meta));
    Ok(())
}

fn collect_dir<'a>(
    root: &Path,
    entries: &mut BTreeMap<RelPath, PayloadEntry<'a>>,
) -> Result<(), Error> {
    let root_meta = fs::symlink_metadata(root).map_err(StoreError::from)?;
    let root_ft = root_meta.file_type();
    if root_ft.is_symlink() {
        return Err(StoreError::InvalidPayload(format!(
            "payload directory must not be a symlink: {}",
            root.display()
        ))
        .into());
    }
    if !root_ft.is_dir() {
        return Err(StoreError::InvalidPayload(format!(
            "payload directory is not a directory: {}",
            root.display()
        ))
        .into());
    }

    for WalkEntry { path, meta } in walk_files(root, false).map_err(StoreError::from)? {
        let ft = meta.file_type();
        if ft.is_symlink() {
            return Err(StoreError::InvalidPayload(format!(
                "payload contains symlink: {}",
                path.display()
            ))
            .into());
        }
        if !ft.is_file() && !ft.is_dir() {
            return Err(StoreError::InvalidPayload(format!(
                "payload contains special file: {}",
                path.display()
            ))
            .into());
        }

        let rel = rel_path_from_root(root, &path)?;
        if ft.is_file() {
            entries.insert(rel, PayloadEntry::from_file(path, &meta));
        }
    }

    Ok(())
}

fn check_path_collisions(entries: &BTreeMap<RelPath, PayloadEntry<'_>>) -> Result<(), Error> {
    // Maps ASCII-lowercased path -> (original spelling, is_dir).
    let mut seen: HashMap<String, (&str, bool)> = HashMap::with_capacity(entries.len());

    for rel in entries.keys() {
        let raw = rel.as_str();
        if let Some((prev, _)) = seen.insert(raw.to_ascii_lowercase(), (raw, false)) {
            return Err(Error::InvalidPath(format!(
                "path collision: '{prev}' vs '{raw}'"
            )));
        }

        for (idx, _) in raw.match_indices('/') {
            let dir = &raw[..idx];
            match seen.entry(dir.to_ascii_lowercase()) {
                Entry::Vacant(slot) => {
                    slot.insert((dir, true));
                }
                Entry::Occupied(slot) => {
                    let &(prev, prev_is_dir) = slot.get();
                    if !prev_is_dir || prev != dir {
                        return Err(Error::InvalidPath(format!(
                            "path collision: '{prev}' vs '{dir}'"
                        )));
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collect_bytes_uses_default_path() {
        let map = collect(PayloadSource::Bytes(b"hello")).unwrap();
        assert_eq!(map.len(), 1);
        let key = RelPath::parse(PayloadSource::DEFAULT_BYTES_PATH).unwrap();
        assert_eq!(map.get(&key), Some(&PayloadEntry::Bytes(b"hello")));
    }

    #[test]
    fn collect_files_preserves_entries() {
        let mut files = BTreeMap::new();
        files.insert(RelPath::parse("a.txt").unwrap(), b"one".as_slice());
        files.insert(RelPath::parse("nested/b.txt").unwrap(), b"two".as_slice());

        let map = collect(PayloadSource::Files(files)).unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(
            map.get(&RelPath::parse("a.txt").unwrap()),
            Some(&PayloadEntry::Bytes(b"one"))
        );
        assert_eq!(
            map.get(&RelPath::parse("nested/b.txt").unwrap()),
            Some(&PayloadEntry::Bytes(b"two"))
        );
    }

    #[test]
    fn collect_rejects_ascii_case_collision_in_files() {
        let mut files = BTreeMap::new();
        files.insert(RelPath::parse("A.bin").unwrap(), b"upper".as_slice());
        files.insert(RelPath::parse("a.bin").unwrap(), b"lower".as_slice());

        let err = collect(PayloadSource::Files(files)).unwrap_err();
        assert!(matches!(err, Error::InvalidPath(_)), "got {err:?}");
    }

    #[test]
    fn collect_rejects_directory_case_collision_in_files() {
        let mut files = BTreeMap::new();
        files.insert(RelPath::parse("Dir/a.bin").unwrap(), b"1".as_slice());
        files.insert(RelPath::parse("dir/b.bin").unwrap(), b"2".as_slice());

        let err = collect(PayloadSource::Files(files)).unwrap_err();
        assert!(matches!(err, Error::InvalidPath(_)), "got {err:?}");
    }

    #[test]
    fn collect_rejects_file_and_directory_prefix_collision() {
        let mut files = BTreeMap::new();
        files.insert(RelPath::parse("foo").unwrap(), b"1".as_slice());
        files.insert(RelPath::parse("foo/bar").unwrap(), b"2".as_slice());

        let err = collect(PayloadSource::Files(files)).unwrap_err();
        assert!(matches!(err, Error::InvalidPath(_)), "got {err:?}");
    }

    #[test]
    fn collect_single_regular_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("model.bin");
        fs::write(&file_path, b"weights").unwrap();

        let map = collect(PayloadSource::File(&file_path)).unwrap();
        assert_eq!(map.len(), 1);
        assert!(matches!(
            map.get(&RelPath::parse("model.bin").unwrap()),
            Some(PayloadEntry::File { path, .. }) if *path == file_path
        ));
    }

    #[test]
    fn collect_single_file_rejects_directory() {
        let dir = tempfile::tempdir().unwrap();
        let err = collect(PayloadSource::File(dir.path())).unwrap_err();
        assert!(
            matches!(err, Error::Store(StoreError::InvalidPayload(_))),
            "got {err:?}"
        );
    }

    #[test]
    fn collect_dir_walks_recursively() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("sub").join("deep");
        fs::create_dir_all(&nested).unwrap();
        let f1 = dir.path().join("root.txt");
        let f2 = nested.join("leaf.bin");
        fs::write(&f1, b"root").unwrap();
        fs::write(&f2, b"leaf").unwrap();

        let map = collect(PayloadSource::Dir(dir.path())).unwrap();
        assert_eq!(map.len(), 2);
        assert!(matches!(
            map.get(&RelPath::parse("root.txt").unwrap()),
            Some(PayloadEntry::File { path, .. }) if *path == f1
        ));
        assert!(matches!(
            map.get(&RelPath::parse("sub/deep/leaf.bin").unwrap()),
            Some(PayloadEntry::File { path, .. }) if *path == f2
        ));
    }

    #[test]
    fn collect_dir_rejects_regular_file_as_root() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("not_a_dir.txt");
        fs::write(&f, b"x").unwrap();

        let err = collect(PayloadSource::Dir(&f)).unwrap_err();
        assert!(
            matches!(err, Error::Store(StoreError::InvalidPayload(_))),
            "got {err:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn collect_rejects_symlinks_and_special_files() {
        use std::os::unix::fs::symlink;
        use std::os::unix::net::UnixListener;

        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("real.txt");
        fs::write(&target, b"real").unwrap();

        // Top-level file symlink
        let file_link = dir.path().join("link.txt");
        symlink(&target, &file_link).unwrap();
        let err = collect(PayloadSource::File(&file_link)).unwrap_err();
        assert!(
            matches!(err, Error::Store(StoreError::InvalidPayload(_))),
            "expected InvalidPayload for file symlink, got {err:?}"
        );

        // Top-level dir symlink
        let real_dir = dir.path().join("real_dir");
        fs::create_dir(&real_dir).unwrap();
        let dir_link = dir.path().join("dir_link");
        symlink(&real_dir, &dir_link).unwrap();
        let err = collect(PayloadSource::Dir(&dir_link)).unwrap_err();
        assert!(
            matches!(err, Error::Store(StoreError::InvalidPayload(_))),
            "expected InvalidPayload for dir symlink, got {err:?}"
        );

        // Internal symlink inside directory
        let tree_internal = dir.path().join("tree_internal");
        fs::create_dir(&tree_internal).unwrap();
        let inner_file = tree_internal.join("a.txt");
        fs::write(&inner_file, b"a").unwrap();
        symlink(&inner_file, tree_internal.join("b.txt")).unwrap();
        let err = collect(PayloadSource::Dir(&tree_internal)).unwrap_err();
        assert!(
            matches!(err, Error::Store(StoreError::InvalidPayload(_))),
            "expected InvalidPayload for internal symlink, got {err:?}"
        );

        // External symlink inside directory
        let tree_external = dir.path().join("tree_external");
        fs::create_dir(&tree_external).unwrap();
        symlink(&target, tree_external.join("escape.txt")).unwrap();
        let err = collect(PayloadSource::Dir(&tree_external)).unwrap_err();
        assert!(
            matches!(err, Error::Store(StoreError::InvalidPayload(_))),
            "expected InvalidPayload for external symlink, got {err:?}"
        );

        // Special file (Unix domain socket)
        let sock_dir = tempfile::tempdir_in("/tmp").unwrap();
        let sock_path = sock_dir.path().join("s.sock");
        let _listener = UnixListener::bind(&sock_path).unwrap();

        let err = collect(PayloadSource::File(&sock_path)).unwrap_err();
        assert!(
            matches!(err, Error::Store(StoreError::InvalidPayload(_))),
            "expected InvalidPayload for special file, got {err:?}"
        );

        let err = collect(PayloadSource::Dir(sock_dir.path())).unwrap_err();
        assert!(
            matches!(err, Error::Store(StoreError::InvalidPayload(_))),
            "expected InvalidPayload for special file in dir, got {err:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn collect_rejects_invalid_rel_path_in_dir() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("bad:name.txt"), b"x").unwrap();

        let err = collect(PayloadSource::Dir(dir.path())).unwrap_err();
        assert!(matches!(err, Error::InvalidPath(_)), "got {err:?}");
    }

    #[test]
    fn materialize_refuses_to_overwrite_with_duplicate_entry() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("out");
        let entry = |bytes: &'static [u8]| Ok((RelPath::parse("sub/a.bin").unwrap(), bytes));

        let err = materialize(&dest, [entry(b"first"), entry(b"second")]).unwrap_err();
        assert!(
            matches!(err, Error::Store(StoreError::Io(_))),
            "got {err:?}"
        );
        assert_eq!(fs::read(dest.join("sub/a.bin")).unwrap(), b"first");
    }

    #[cfg(unix)]
    #[test]
    fn materialize_does_not_follow_planted_symlink_between_entries() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("out");
        let victim = dir.path().join("victim.txt");
        fs::write(&victim, b"untouched").unwrap();

        let planted = dest.clone();
        let victim2 = victim.clone();
        let entries = (0..2).map(move |i| {
            if i == 1 {
                symlink(&victim2, planted.join("b.bin")).unwrap();
            }
            let name = if i == 0 { "a.bin" } else { "b.bin" };
            Ok((RelPath::parse(name).unwrap(), b"new".as_slice()))
        });
        assert!(materialize(&dest, entries).is_err());
        assert_eq!(fs::read(&victim).unwrap(), b"untouched");
    }

    #[cfg(unix)]
    #[test]
    fn open_verified_rejects_symlink_swap_after_collect() {
        use std::io::Read;
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let secret = dir.path().join("secret.txt");
        fs::write(&secret, b"top-secret").unwrap();

        // Direct file swap: collect -> replace file with symlink -> open_verified Err.
        let src_file = dir.path().join("model.bin");
        fs::write(&src_file, b"legit").unwrap();
        let entries = collect(PayloadSource::File(&src_file)).unwrap();
        let key = RelPath::parse("model.bin").unwrap();

        let mut ok_file = entries[&key].open_verified().unwrap();
        let mut buf = Vec::new();
        ok_file.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, b"legit");

        fs::remove_file(&src_file).unwrap();
        symlink(&secret, &src_file).unwrap();
        let err = entries[&key].open_verified().unwrap_err();
        assert!(
            matches!(err, Error::Store(StoreError::InvalidPayload(_))),
            "expected InvalidPayload after file symlink swap, got {err:?}"
        );

        // Intermediate directory swap: collect -> replace subdir with symlink -> open_verified Err.
        let tree = dir.path().join("tree");
        let sub = tree.join("sub");
        fs::create_dir_all(&sub).unwrap();
        fs::write(sub.join("weights.bin"), b"ok").unwrap();

        let outside = dir.path().join("outside");
        fs::create_dir(&outside).unwrap();
        fs::write(outside.join("weights.bin"), b"exfiltrated").unwrap();

        let dir_entries = collect(PayloadSource::Dir(&tree)).unwrap();
        let sub_key = RelPath::parse("sub/weights.bin").unwrap();
        fs::remove_dir_all(&sub).unwrap();
        symlink(&outside, &sub).unwrap();

        let err = dir_entries[&sub_key].open_verified().unwrap_err();
        assert!(
            matches!(err, Error::Store(StoreError::InvalidPayload(_))),
            "expected InvalidPayload after subdir symlink swap, got {err:?}"
        );
    }
}
