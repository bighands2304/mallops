//! Runtime-owned commit of block output drafts (ADR-0007 D4).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::ValidationError;
use crate::{ArtifactId, ArtifactStore, DraftPayload, Error, OutputDrafts, ResolvedPorts};

/// Resolves a `File`/`Dir` draft path (relative paths are joined to `workdir`) to its
/// canonical form, which must lie inside `root` (the canonical workdir) and be a regular
/// file (`dir == false`) or a directory.
fn confine(root: &Path, workdir: &Path, path: &Path, dir: bool) -> Result<PathBuf, String> {
    let shown = path.display();
    if path.as_os_str().is_empty() {
        return Err("payload path must not be empty".into());
    }
    let canonical = fs::canonicalize(workdir.join(path))
        .map_err(|e| format!("cannot resolve payload path `{shown}` inside workdir: {e}"))?;
    if !canonical.starts_with(root) {
        return Err(format!("payload path `{shown}` escapes workdir"));
    }
    let meta = fs::symlink_metadata(&canonical)
        .map_err(|e| format!("cannot stat payload path `{shown}`: {e}"))?;
    match (dir, meta.is_dir(), meta.is_file()) {
        (true, true, _) | (false, _, true) => Ok(canonical),
        (true, ..) => Err(format!("payload path `{shown}` is not a directory")),
        (false, ..) => Err(format!("payload path `{shown}` is not a regular file")),
    }
}

/// Validates `drafts` against the node's resolved output `ports` and commits them.
///
/// - every draft must name a declared output port and satisfy it (`Port::accepts_manifest`);
/// - when `produced` (outcome `Produced`), every required output port must have a draft;
///   for `Skipped` outcomes drafts are optional;
/// - `File`/`Dir` payloads must canonicalize inside `workdir` (no `..` or symlink escapes) to a
///   regular file / a directory; the canonical path is what `put` receives;
/// - `manifest.parents` is replaced by `parents` (the node's actual input ids, sorted and deduped);
/// - drafts are committed with [`ArtifactStore::put`] only if all checks pass.
///
/// Payload *content* (nested symlinks, special files inside a `Dir`) is checked by the store
/// during `put`; when that fails, outputs of the node committed before stay in the store as
/// unreferenced artifacts (immutable and harmless, left to a later GC).
///
/// Returns the committed ids by output port.
///
/// # Errors
///
/// [`Error::Validation`] with every violation, paths `out.<port>`; store errors as returned.
pub(crate) async fn commit_outputs(
    store: &dyn ArtifactStore,
    ports: &ResolvedPorts,
    drafts: OutputDrafts,
    parents: &[ArtifactId],
    workdir: &Path,
    produced: bool,
) -> Result<BTreeMap<String, ArtifactId>, Error> {
    // ponytail: sync canonicalize/stat on the async thread, a few syscalls per output
    // (like the per-node `create_dir` in the scheduler).
    let root = fs::canonicalize(workdir).map_err(|e| {
        Error::Runtime(format!(
            "cannot canonicalize workdir `{}`: {e}",
            workdir.display()
        ))
    })?;
    let mut errors = Vec::new();
    let mut checked = Vec::new();

    for (name, mut draft) in drafts {
        let at = format!("out.{name}");
        let Some(port) = ports.outputs.iter().find(|p| p.name == name.as_str()) else {
            errors.push(ValidationError::new(
                at,
                format!("unknown output port `{name}`"),
            ));
            continue;
        };
        if !port.accepts_manifest(&draft.manifest) {
            let m = &draft.manifest;
            errors.push(ValidationError::new(
                &at,
                format!(
                    "manifest (kind `{}`, format `{}`, flavor {:?}) does not satisfy output port \
                     `{name}` (kind `{}`, formats {:?}, flavors {:?})",
                    m.kind, m.format, m.flavor, port.kind, port.formats, port.flavors
                ),
            ));
        }
        let dir = matches!(draft.payload, DraftPayload::Dir(_));
        if let DraftPayload::File(path) | DraftPayload::Dir(path) = &mut draft.payload {
            // The path may change between this check and `put`: accepted by the trust model
            // (ADR-0007 D7; decision 2026-11-22 #2 B1-bis).
            match confine(&root, workdir, path, dir) {
                Ok(canonical) => *path = canonical,
                Err(msg) => errors.push(ValidationError::new(&at, msg)),
            }
        }
        checked.push((name, draft));
    }

    if produced {
        for port in ports.outputs.iter().filter(|p| p.required) {
            if checked.iter().all(|(name, _)| name != port.name.as_ref()) {
                errors.push(ValidationError::new(
                    format!("out.{}", port.name),
                    format!("missing required output `{}`", port.name),
                ));
            }
        }
    }

    if !errors.is_empty() {
        return Err(Error::Validation(errors));
    }

    let mut parents = parents.to_vec();
    parents.sort_unstable();
    parents.dedup();

    let mut committed = BTreeMap::new();
    for (port, draft) in checked {
        let manifest = draft.manifest.with_parents(parents.iter().copied());
        let id = store.put(manifest, draft.payload.as_source()).await?;
        committed.insert(port, id);
    }
    Ok(committed)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::{
        Artifact, ArtifactKind, BoxFuture, Manifest, MemoryStore, OutputDraft, PayloadSource, Port,
        RelPath,
    };

    /// Wraps [`MemoryStore`] and counts every `put` invocation so tests can prove
    /// no store write happened when validation fails.
    struct SpyStore {
        inner: MemoryStore,
        put_calls: AtomicUsize,
    }

    impl SpyStore {
        fn new() -> Self {
            Self {
                inner: MemoryStore::new(),
                put_calls: AtomicUsize::new(0),
            }
        }

        fn puts(&self) -> usize {
            self.put_calls.load(Ordering::SeqCst)
        }
    }

    impl ArtifactStore for SpyStore {
        fn put<'a>(
            &'a self,
            manifest: Manifest,
            payload: PayloadSource<'a>,
        ) -> BoxFuture<'a, Result<ArtifactId, Error>> {
            self.put_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.put(manifest, payload)
        }

        fn get<'a>(&'a self, id: &'a ArtifactId) -> BoxFuture<'a, Result<Option<Artifact>, Error>> {
            self.inner.get(id)
        }

        fn fetch<'a>(
            &'a self,
            id: &'a ArtifactId,
            dest: &'a Path,
        ) -> BoxFuture<'a, Result<PathBuf, Error>> {
            self.inner.fetch(id, dest)
        }

        fn verify<'a>(&'a self, id: &'a ArtifactId) -> BoxFuture<'a, Result<(), Error>> {
            self.inner.verify(id)
        }

        fn exists<'a>(&'a self, id: &'a ArtifactId) -> BoxFuture<'a, Result<bool, Error>> {
            self.inner.exists(id)
        }

        fn put_meta<'a>(
            &'a self,
            key: &'a RelPath,
            data: &'a [u8],
        ) -> BoxFuture<'a, Result<(), Error>> {
            self.inner.put_meta(key, data)
        }

        fn get_meta<'a>(
            &'a self,
            key: &'a RelPath,
        ) -> BoxFuture<'a, Result<Option<Vec<u8>>, Error>> {
            self.inner.get_meta(key)
        }

        fn list_meta<'a>(
            &'a self,
            prefix: &'a RelPath,
        ) -> BoxFuture<'a, Result<Vec<RelPath>, Error>> {
            self.inner.list_meta(prefix)
        }

        fn delete_meta<'a>(&'a self, key: &'a RelPath) -> BoxFuture<'a, Result<(), Error>> {
            self.inner.delete_meta(key)
        }
    }

    #[tokio::test]
    async fn commit_ok_sets_sorted_deduped_parents_and_returns_ids() {
        let store = MemoryStore::new();
        let workdir = tempfile::tempdir().unwrap();
        let rel_file = PathBuf::from("weights.onnx");
        fs::write(workdir.path().join(&rel_file), b"onnx-bytes").unwrap();
        let sub_dir = workdir.path().join("bundle");
        fs::create_dir(&sub_dir).unwrap();
        fs::write(sub_dir.join("meta.json"), b"{}").unwrap();

        let ports = ResolvedPorts::new(
            vec![],
            vec![
                Port::new("model", ArtifactKind::Model)
                    .with_format("onnx")
                    .with_flavor("tabular"),
                Port::new("metrics", ArtifactKind::Metrics).with_format("json"),
                Port::new("bundle", ArtifactKind::Package).with_format("dir"),
            ],
        );

        let fake_parent = ArtifactId::from_bytes([0x99; 32]);
        let p_low = ArtifactId::from_bytes([0x11; 32]);
        let p_high = ArtifactId::from_bytes([0x22; 32]);

        let drafts = OutputDrafts::new()
            .with(
                "model",
                OutputDraft::new(
                    Manifest::new(ArtifactKind::Model, "onnx")
                        .with_flavor("tabular")
                        .with_parent(fake_parent),
                    DraftPayload::File(rel_file),
                ),
            )
            .with(
                "metrics",
                OutputDraft::new(
                    Manifest::new(ArtifactKind::Metrics, "json"),
                    DraftPayload::Bytes(b"{\"acc\":0.9}".to_vec()),
                ),
            )
            .with(
                "bundle",
                OutputDraft::new(
                    Manifest::new(ArtifactKind::Package, "dir"),
                    DraftPayload::Dir(sub_dir),
                ),
            );

        let committed = commit_outputs(
            &store,
            &ports,
            drafts,
            &[p_high, p_low, p_high],
            workdir.path(),
            true,
        )
        .await
        .unwrap();

        assert_eq!(committed.len(), 3);
        for port in ["model", "metrics", "bundle"] {
            let id = committed.get(port).expect("port committed");
            let artifact = store.get(id).await.unwrap().expect("stored in store");
            assert_eq!(artifact.manifest().parents, vec![p_low, p_high]);
            store.verify(id).await.unwrap();
        }
    }

    #[tokio::test]
    async fn commit_rejects_unknown_output_port() {
        let store = SpyStore::new();
        let workdir = tempfile::tempdir().unwrap();
        let ports = ResolvedPorts::new(vec![], vec![Port::new("data", ArtifactKind::Dataset)]);
        let drafts = OutputDrafts::new()
            .with(
                "data",
                OutputDraft::new(
                    Manifest::new(ArtifactKind::Dataset, "parquet"),
                    DraftPayload::Bytes(b"ok".to_vec()),
                ),
            )
            .with(
                "surprise",
                OutputDraft::new(
                    Manifest::new(ArtifactKind::Blob, "bin"),
                    DraftPayload::Bytes(b"bad".to_vec()),
                ),
            );

        let err = commit_outputs(&store, &ports, drafts, &[], workdir.path(), true)
            .await
            .unwrap_err();
        let Error::Validation(errs) = err else {
            panic!("expected Validation error, got {err:?}");
        };
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].path, "out.surprise");
        assert!(errs[0].message.contains("unknown output port"));
        assert_eq!(store.puts(), 0);
    }

    #[tokio::test]
    async fn commit_rejects_manifest_violating_port_kind_format_and_flavor() {
        let store = SpyStore::new();
        let workdir = tempfile::tempdir().unwrap();
        let ports = ResolvedPorts::new(
            vec![],
            vec![
                Port::new("model", ArtifactKind::Model)
                    .with_format("onnx")
                    .with_flavor("tabular"),
            ],
        );

        // Wrong kind.
        let bad_kind = OutputDrafts::new().with(
            "model",
            OutputDraft::new(
                Manifest::new(ArtifactKind::Dataset, "onnx").with_flavor("tabular"),
                DraftPayload::Bytes(b"x".to_vec()),
            ),
        );
        let Error::Validation(errs) =
            commit_outputs(&store, &ports, bad_kind, &[], workdir.path(), true)
                .await
                .unwrap_err()
        else {
            panic!("expected Validation error for wrong kind");
        };
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].path, "out.model");
        assert!(errs[0].message.contains("kind"));

        // Wrong format.
        let bad_format = OutputDrafts::new().with(
            "model",
            OutputDraft::new(
                Manifest::new(ArtifactKind::Model, "pt").with_flavor("tabular"),
                DraftPayload::Bytes(b"x".to_vec()),
            ),
        );
        let Error::Validation(errs) =
            commit_outputs(&store, &ports, bad_format, &[], workdir.path(), true)
                .await
                .unwrap_err()
        else {
            panic!("expected Validation error for wrong format");
        };
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].path, "out.model");
        assert!(errs[0].message.contains("format"));

        // Missing/wrong flavor.
        let bad_flavor = OutputDrafts::new().with(
            "model",
            OutputDraft::new(
                Manifest::new(ArtifactKind::Model, "onnx"),
                DraftPayload::Bytes(b"x".to_vec()),
            ),
        );
        let Error::Validation(errs) =
            commit_outputs(&store, &ports, bad_flavor, &[], workdir.path(), true)
                .await
                .unwrap_err()
        else {
            panic!("expected Validation error for missing flavor");
        };
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].path, "out.model");
        assert!(errs[0].message.contains("flavor"));

        assert_eq!(store.puts(), 0);
    }

    #[tokio::test]
    async fn commit_missing_required_output_fails_when_produced_and_ok_when_skipped() {
        let store = SpyStore::new();
        let workdir = tempfile::tempdir().unwrap();
        let ports = ResolvedPorts::new(
            vec![],
            vec![
                Port::new("model", ArtifactKind::Model).with_format("onnx"),
                Port::new("report", ArtifactKind::Metrics).optional(),
            ],
        );

        // Produced with missing required output -> Validation error only on required port.
        let err = commit_outputs(
            &store,
            &ports,
            OutputDrafts::new(),
            &[],
            workdir.path(),
            true,
        )
        .await
        .unwrap_err();
        let Error::Validation(errs) = err else {
            panic!("expected Validation error, got {err:?}");
        };
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].path, "out.model");
        assert_eq!(store.puts(), 0);

        // Skipped (produced = false) with no drafts -> Ok, empty map.
        let skipped = commit_outputs(
            &store,
            &ports,
            OutputDrafts::new(),
            &[],
            workdir.path(),
            false,
        )
        .await
        .unwrap();
        assert!(skipped.is_empty());
        assert_eq!(store.puts(), 0);

        // Skipped with an invalid draft still fails validation without writing to store.
        let bad_skipped = OutputDrafts::new().with(
            "model",
            OutputDraft::new(
                Manifest::new(ArtifactKind::Model, "wrong"),
                DraftPayload::Bytes(b"m".to_vec()),
            ),
        );
        let err = commit_outputs(&store, &ports, bad_skipped, &[], workdir.path(), false)
            .await
            .unwrap_err();
        assert!(
            matches!(err, Error::Validation(errs) if errs.len() == 1 && errs[0].path == "out.model")
        );
        assert_eq!(store.puts(), 0);

        // Produced with required output present and optional omitted -> Ok.
        let drafts = OutputDrafts::new().with(
            "model",
            OutputDraft::new(
                Manifest::new(ArtifactKind::Model, "onnx"),
                DraftPayload::Bytes(b"m".to_vec()),
            ),
        );
        let committed = commit_outputs(&store, &ports, drafts, &[], workdir.path(), true)
            .await
            .unwrap();
        assert_eq!(committed.len(), 1);
        assert!(committed.contains_key("model"));
        assert_eq!(store.puts(), 1);
    }

    #[tokio::test]
    async fn commit_rejects_file_outside_workdir_absolute_and_parent_traversal() {
        let store = SpyStore::new();
        let outer = tempfile::tempdir().unwrap();
        let workdir = outer.path().join("work");
        let sibling = outer.path().join("work2");
        fs::create_dir(&workdir).unwrap();
        fs::create_dir(&sibling).unwrap();

        let outside_file = outer.path().join("secret.txt");
        let sibling_file = sibling.join("secret.txt");
        fs::write(&outside_file, b"nope").unwrap();
        fs::write(&sibling_file, b"nope").unwrap();

        let ports = ResolvedPorts::new(vec![], vec![Port::new("data", ArtifactKind::Blob)]);
        let make_draft = |payload| {
            OutputDrafts::new().with(
                "data",
                OutputDraft::new(Manifest::new(ArtifactKind::Blob, "text"), payload),
            )
        };

        for payload in [
            DraftPayload::File(outside_file),
            DraftPayload::File(PathBuf::from("../secret.txt")),
            DraftPayload::File(workdir.join("../secret.txt")),
            DraftPayload::File(sibling_file),
            DraftPayload::Dir(PathBuf::from("../work2")),
            DraftPayload::File(PathBuf::new()),
        ] {
            let err = commit_outputs(&store, &ports, make_draft(payload), &[], &workdir, true)
                .await
                .unwrap_err();
            let Error::Validation(errs) = err else {
                panic!("expected Validation error, got {err:?}");
            };
            assert_eq!(errs.len(), 1);
            assert_eq!(errs[0].path, "out.data");
        }
        assert_eq!(store.puts(), 0);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn commit_rejects_symlink_inside_workdir_pointing_outside() {
        use std::os::unix::fs::symlink;

        let store = SpyStore::new();
        let outer = tempfile::tempdir().unwrap();
        let workdir = outer.path().join("work");
        let outside_dir = outer.path().join("outside");
        fs::create_dir(&workdir).unwrap();
        fs::create_dir(&outside_dir).unwrap();

        let outside_file = outside_dir.join("secret.txt");
        fs::write(&outside_file, b"secret").unwrap();

        let file_link = workdir.join("escape.txt");
        symlink(&outside_file, &file_link).unwrap();

        let dir_link = workdir.join("escape_dir");
        symlink(&outside_dir, &dir_link).unwrap();

        let dangling_link = workdir.join("dangling.txt");
        symlink(outer.path().join("does_not_exist"), &dangling_link).unwrap();

        let ports = ResolvedPorts::new(vec![], vec![Port::new("data", ArtifactKind::Blob)]);
        let make_draft = |payload| {
            OutputDrafts::new().with(
                "data",
                OutputDraft::new(Manifest::new(ArtifactKind::Blob, "text"), payload),
            )
        };

        for payload in [
            DraftPayload::File(PathBuf::from("escape.txt")),
            DraftPayload::File(file_link),
            DraftPayload::Dir(PathBuf::from("escape_dir")),
            DraftPayload::File(PathBuf::from("escape_dir/secret.txt")),
            DraftPayload::File(dangling_link),
        ] {
            let err = commit_outputs(&store, &ports, make_draft(payload), &[], &workdir, true)
                .await
                .unwrap_err();
            let Error::Validation(errs) = err else {
                panic!("expected Validation error, got {err:?}");
            };
            assert_eq!(errs.len(), 1);
            assert_eq!(errs[0].path, "out.data");
        }
        assert_eq!(store.puts(), 0);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn commit_internal_symlinks_commit_canonical_target_nested_rejected_by_store() {
        use std::os::unix::fs::symlink;

        let store = MemoryStore::new();
        let workdir = tempfile::tempdir().unwrap();
        let sub = workdir.path().join("sub");
        fs::create_dir(&sub).unwrap();
        fs::write(sub.join("real.txt"), b"ok").unwrap();
        symlink(sub.join("real.txt"), workdir.path().join("alias.txt")).unwrap();
        symlink(&sub, workdir.path().join("alias_dir")).unwrap();
        let nested = workdir.path().join("nested");
        fs::create_dir(&nested).unwrap();
        symlink(sub.join("real.txt"), nested.join("link.txt")).unwrap();

        let ports = ResolvedPorts::new(vec![], vec![Port::new("data", ArtifactKind::Blob)]);
        let commit = |payload| {
            let drafts = OutputDrafts::new().with(
                "data",
                OutputDraft::new(Manifest::new(ArtifactKind::Blob, "text"), payload),
            );
            commit_outputs(&store, &ports, drafts, &[], workdir.path(), true)
        };

        // Top-level and intermediate symlinks inside the workdir: `put` gets the canonical
        // target, so both commit the same content as the real file.
        let real = commit(DraftPayload::File("sub/real.txt".into()))
            .await
            .unwrap();
        for alias in ["alias.txt", "alias_dir/real.txt"] {
            let via = commit(DraftPayload::File(alias.into())).await.unwrap();
            assert_eq!(via["data"], real["data"], "{alias}");
        }

        // Symlinks nested inside a `Dir` payload are rejected by the store at `put`.
        let err = commit(DraftPayload::Dir("nested".into()))
            .await
            .unwrap_err();
        assert!(
            matches!(err, Error::Store(crate::StoreError::InvalidPayload(_))),
            "expected InvalidPayload from store for nested symlink, got {err:?}"
        );
    }

    #[tokio::test]
    async fn commit_rejects_file_dir_type_mismatch_without_put() {
        let store = SpyStore::new();
        let workdir = tempfile::tempdir().unwrap();
        fs::create_dir(workdir.path().join("dir")).unwrap();
        fs::write(workdir.path().join("file.bin"), b"x").unwrap();
        let ports = ResolvedPorts::new(vec![], vec![Port::new("data", ArtifactKind::Blob)]);

        for (payload, expected) in [
            (DraftPayload::File("dir".into()), "not a regular file"),
            (DraftPayload::Dir("file.bin".into()), "not a directory"),
        ] {
            let drafts = OutputDrafts::new().with(
                "data",
                OutputDraft::new(Manifest::new(ArtifactKind::Blob, "bin"), payload),
            );
            let err = commit_outputs(&store, &ports, drafts, &[], workdir.path(), true)
                .await
                .unwrap_err();
            let Error::Validation(errs) = err else {
                panic!("expected Validation error, got {err:?}");
            };
            assert_eq!(errs.len(), 1);
            assert_eq!(errs[0].path, "out.data");
            assert!(errs[0].message.contains(expected), "{}", errs[0].message);
        }
        assert_eq!(store.puts(), 0);
    }

    #[tokio::test]
    async fn commit_performs_no_put_when_any_validation_fails() {
        let store = SpyStore::new();
        let outer = tempfile::tempdir().unwrap();
        let workdir = outer.path().join("work");
        fs::create_dir(&workdir).unwrap();
        let outside_file = outer.path().join("escape.bin");
        fs::write(&outside_file, b"outside").unwrap();

        let ports = ResolvedPorts::new(
            vec![],
            vec![
                Port::new("a", ArtifactKind::Blob).with_format("bin"),
                Port::new("b", ArtifactKind::Model).with_format("onnx"),
                Port::new("c", ArtifactKind::Blob),
                Port::new("missing", ArtifactKind::Metrics),
            ],
        );

        let valid_manifest = Manifest::new(ArtifactKind::Blob, "bin");
        let valid_bytes = b"valid-payload";
        let expected_a_id = ArtifactId::compute_from_bytes(
            &valid_manifest,
            [(
                RelPath::parse(PayloadSource::DEFAULT_BYTES_PATH).unwrap(),
                valid_bytes.as_slice(),
            )],
        )
        .unwrap();

        let drafts = OutputDrafts::new()
            .with(
                "a",
                OutputDraft::new(valid_manifest, DraftPayload::Bytes(valid_bytes.to_vec())),
            )
            .with(
                "b",
                OutputDraft::new(
                    Manifest::new(ArtifactKind::Model, "wrong"),
                    DraftPayload::Bytes(b"m".to_vec()),
                ),
            )
            .with(
                "c",
                OutputDraft::new(
                    Manifest::new(ArtifactKind::Blob, "bin"),
                    DraftPayload::File(outside_file),
                ),
            )
            .with(
                "unknown",
                OutputDraft::new(
                    Manifest::new(ArtifactKind::Blob, "bin"),
                    DraftPayload::Bytes(b"u".to_vec()),
                ),
            );

        let err = commit_outputs(&store, &ports, drafts, &[], &workdir, true)
            .await
            .unwrap_err();
        let Error::Validation(errs) = err else {
            panic!("expected aggregated Validation error, got {err:?}");
        };

        let paths: Vec<&str> = errs.iter().map(|e| e.path.as_str()).collect();
        assert_eq!(errs.len(), 4, "got errors: {errs:?}");
        assert!(paths.contains(&"out.b"));
        assert!(paths.contains(&"out.c"));
        assert!(paths.contains(&"out.unknown"));
        assert!(paths.contains(&"out.missing"));

        assert_eq!(store.puts(), 0);
        assert!(!store.exists(&expected_a_id).await.unwrap());
    }
}
