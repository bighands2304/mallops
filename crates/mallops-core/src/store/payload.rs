//! Shared synchronous payload collection and validation for [`ArtifactStore`](super::ArtifactStore)
//! implementations.
//!
//! Callers in async contexts should invoke [`collect`] inside `tokio::task::spawn_blocking`.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use crate::{Error, PayloadSource, RelPath, StoreError};

/// A single validated payload entry ready for ingestion into a store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PayloadEntry<'a> {
    /// A regular file on the local filesystem (verified non-symlink at collection time).
    File(PathBuf),
    /// An in-memory byte slice.
    Bytes(&'a [u8]),
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

fn collect_single_file<'a>(
    path: &Path,
    entries: &mut BTreeMap<RelPath, PayloadEntry<'a>>,
) -> Result<(), Error> {
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
    entries.insert(rel, PayloadEntry::File(path.to_path_buf()));
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

    let mut stack: Vec<(PathBuf, String)> = vec![(root.to_path_buf(), String::new())];
    while let Some((dir_path, rel_prefix)) = stack.pop() {
        let read_dir = fs::read_dir(&dir_path).map_err(StoreError::from)?;
        for entry_res in read_dir {
            let entry = entry_res.map_err(StoreError::from)?;
            let entry_path = entry.path();
            let meta = fs::symlink_metadata(&entry_path).map_err(StoreError::from)?;
            let ft = meta.file_type();

            if ft.is_symlink() {
                return Err(StoreError::InvalidPayload(format!(
                    "payload contains symlink: {}",
                    entry_path.display()
                ))
                .into());
            }
            if !ft.is_file() && !ft.is_dir() {
                return Err(StoreError::InvalidPayload(format!(
                    "payload contains special file: {}",
                    entry_path.display()
                ))
                .into());
            }

            let file_name = entry.file_name();
            let name_str = file_name.to_str().ok_or_else(|| {
                Error::InvalidPath(format!(
                    "payload contains non-UTF-8 path component: {}",
                    entry_path.display()
                ))
            })?;

            let rel_str = if rel_prefix.is_empty() {
                name_str.to_owned()
            } else {
                format!("{rel_prefix}/{name_str}")
            };

            let rel = RelPath::parse(&rel_str)?;
            if ft.is_dir() {
                stack.push((entry_path, rel_str));
            } else {
                entries.insert(rel, PayloadEntry::File(entry_path));
            }
        }
    }

    Ok(())
}

fn check_path_collisions(entries: &BTreeMap<RelPath, PayloadEntry<'_>>) -> Result<(), Error> {
    let mut seen_files: HashMap<String, &RelPath> = HashMap::with_capacity(entries.len());
    let mut seen_dirs: HashMap<String, String> = HashMap::new();

    for rel in entries.keys() {
        let raw = rel.as_str();
        let folded = raw.to_ascii_lowercase();

        if let Some(existing) = seen_files.get(&folded) {
            return Err(Error::InvalidPath(format!(
                "case-insensitive path collision between '{existing}' and '{rel}'"
            )));
        }
        if let Some(existing_dir) = seen_dirs.get(&folded) {
            return Err(Error::InvalidPath(format!(
                "path collision between directory '{existing_dir}' and file '{rel}'"
            )));
        }
        seen_files.insert(folded, rel);

        for (idx, _) in raw.match_indices('/') {
            let dir_prefix = &raw[..idx];
            let folded_dir = dir_prefix.to_ascii_lowercase();

            if let Some(existing_file) = seen_files.get(&folded_dir) {
                return Err(Error::InvalidPath(format!(
                    "path collision between file '{existing_file}' and directory '{dir_prefix}'"
                )));
            }
            if let Some(existing_dir) = seen_dirs.get(&folded_dir) {
                if existing_dir != dir_prefix {
                    return Err(Error::InvalidPath(format!(
                        "case-insensitive directory collision between '{existing_dir}' and '{dir_prefix}'"
                    )));
                }
            } else {
                seen_dirs.insert(folded_dir, dir_prefix.to_owned());
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
        assert_eq!(
            map.get(&RelPath::parse("model.bin").unwrap()),
            Some(&PayloadEntry::File(file_path.clone()))
        );
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
        assert_eq!(
            map.get(&RelPath::parse("root.txt").unwrap()),
            Some(&PayloadEntry::File(f1))
        );
        assert_eq!(
            map.get(&RelPath::parse("sub/deep/leaf.bin").unwrap()),
            Some(&PayloadEntry::File(f2))
        );
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
}
