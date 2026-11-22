//! Runtime-owned commit of block output drafts (ADR-0007 D4).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::{
    ArtifactId, ArtifactStore, DraftPayload, Error, Manifest, OutputDraft, OutputDrafts, Port,
    ResolvedPorts, ValidationError,
};

fn describe_manifest_mismatch(port_name: &str, port: &Port, manifest: &Manifest) -> String {
    let mut dims = Vec::new();
    if port.kind != manifest.kind {
        dims.push("kind");
    }
    if !port.formats.is_empty()
        && !port
            .formats
            .iter()
            .any(|f| f.as_ref() == manifest.format.as_str())
    {
        dims.push("format");
    }
    if !port.flavors.is_empty()
        && !manifest
            .flavor
            .as_deref()
            .is_some_and(|fl| port.flavors.iter().any(|f| f.as_ref() == fl))
    {
        dims.push("flavor");
    }
    format!(
        "manifest violates output port `{port_name}` ({} mismatch): got kind `{}`, format `{}`, flavor {:?}; port requires kind `{}`, formats {:?}, flavors {:?}",
        dims.join("+"),
        manifest.kind,
        manifest.format,
        manifest.flavor,
        port.kind,
        port.formats,
        port.flavors,
    )
}

/// Canonicalizes `workdir` and every `File`/`Dir` draft path on Tokio's blocking pool,
/// rejecting paths that cannot be resolved or escape `workdir`.
///
/// Returns `(entry_index, resolved_path)` pairs for valid entries (preserving the resolved
/// non-canonical path so `ArtifactStore::put` still sees and rejects internal top-level
/// symlinks) alongside any collected [`ValidationError`]s.
async fn validate_fs_paths(
    workdir: PathBuf,
    targets: Vec<(usize, String, PathBuf)>,
) -> Result<(Vec<(usize, PathBuf)>, Vec<ValidationError>), Error> {
    tokio::task::spawn_blocking(move || {
        let canonical_workdir = fs::canonicalize(&workdir).map_err(|err| {
            Error::Runtime(format!(
                "cannot canonicalize workdir `{}`: {err}",
                workdir.display()
            ))
        })?;
        if !canonical_workdir.is_dir() {
            return Err(Error::Runtime(format!(
                "workdir `{}` is not a directory",
                workdir.display()
            )));
        }

        let mut resolved_paths = Vec::with_capacity(targets.len());
        let mut errors = Vec::new();

        for (idx, port_name, raw_path) in targets {
            let path_key = format!("out.{port_name}");
            if raw_path.as_os_str().is_empty() {
                errors.push(ValidationError::new(
                    path_key,
                    "payload path must not be empty",
                ));
                continue;
            }
            let resolved = workdir.join(&raw_path);
            match fs::canonicalize(&resolved) {
                Ok(canonical) if canonical.starts_with(&canonical_workdir) => {
                    resolved_paths.push((idx, resolved));
                }
                Ok(_) => {
                    errors.push(ValidationError::new(
                        path_key,
                        format!("payload path `{}` escapes workdir", raw_path.display()),
                    ));
                }
                Err(err) => {
                    errors.push(ValidationError::new(
                        path_key,
                        format!(
                            "cannot resolve payload path `{}` inside workdir: {err}",
                            raw_path.display()
                        ),
                    ));
                }
            }
        }

        Ok((resolved_paths, errors))
    })
    .await
    .map_err(|err| Error::Runtime(format!("blocking commit validation task failed: {err}")))?
}

/// Validates `drafts` against the node's resolved output `ports` and commits them.
///
/// - every draft must name a declared output port and satisfy it (`Port::accepts_manifest`);
/// - when `produced` (outcome `Produced`), every required output port must have a draft;
///   for `Skipped` outcomes drafts are optional;
/// - `File`/`Dir` payloads must resolve inside `workdir` (no `..` or symlink escapes);
/// - `manifest.parents` is replaced by `parents` (the node's actual input ids, sorted and deduped);
/// - drafts are committed with [`ArtifactStore::put`] only if all checks pass.
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
    let mut entries: Vec<(String, OutputDraft)> = drafts.into_iter().collect();
    let mut errors = Vec::new();
    let mut fs_targets = Vec::new();

    for (idx, (port_name, draft)) in entries.iter().enumerate() {
        let Some(port) = ports
            .outputs
            .iter()
            .find(|p| p.name.as_ref() == port_name.as_str())
        else {
            errors.push(ValidationError::new(
                format!("out.{port_name}"),
                format!("unknown output port `{port_name}`"),
            ));
            continue;
        };

        if !port.accepts_manifest(&draft.manifest) {
            errors.push(ValidationError::new(
                format!("out.{port_name}"),
                describe_manifest_mismatch(port_name, port, &draft.manifest),
            ));
        }

        if let DraftPayload::File(path) | DraftPayload::Dir(path) = &draft.payload {
            fs_targets.push((idx, port_name.clone(), path.clone()));
        }
    }

    if produced {
        for port in ports.outputs.iter().filter(|p| p.required) {
            if entries.iter().all(|(name, _)| name != port.name.as_ref()) {
                errors.push(ValidationError::new(
                    format!("out.{}", port.name),
                    format!("missing required output `{}`", port.name),
                ));
            }
        }
    }

    if !fs_targets.is_empty() {
        let (resolved_paths, fs_errors) =
            validate_fs_paths(workdir.to_path_buf(), fs_targets).await?;
        errors.extend(fs_errors);
        for (idx, resolved) in resolved_paths {
            if let DraftPayload::File(path) | DraftPayload::Dir(path) = &mut entries[idx].1.payload
            {
                *path = resolved;
            }
        }
    }

    if !errors.is_empty() {
        return Err(Error::Validation(errors));
    }

    let mut normalized_parents = parents.to_vec();
    normalized_parents.sort_unstable();
    normalized_parents.dedup();

    let mut committed = BTreeMap::new();
    for (port, draft) in entries {
        let manifest = draft
            .manifest
            .with_parents(normalized_parents.iter().copied());
        let id = store.put(manifest, draft.payload.as_source()).await?;
        committed.insert(port, id);
    }
    Ok(committed)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::{Artifact, ArtifactKind, BoxFuture, MemoryStore, PayloadSource, RelPath};

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
    async fn commit_internal_symlink_is_still_rejected_by_store() {
        use std::os::unix::fs::symlink;

        let store = MemoryStore::new();
        let workdir = tempfile::tempdir().unwrap();
        let real_file = workdir.path().join("real.txt");
        fs::write(&real_file, b"ok").unwrap();
        let link = workdir.path().join("alias.txt");
        symlink(&real_file, &link).unwrap();

        let ports = ResolvedPorts::new(vec![], vec![Port::new("data", ArtifactKind::Blob)]);
        let drafts = OutputDrafts::new().with(
            "data",
            OutputDraft::new(
                Manifest::new(ArtifactKind::Blob, "text"),
                DraftPayload::File(PathBuf::from("alias.txt")),
            ),
        );

        let err = commit_outputs(&store, &ports, drafts, &[], workdir.path(), true)
            .await
            .unwrap_err();
        assert!(
            matches!(err, Error::Store(crate::StoreError::InvalidPayload(_))),
            "expected InvalidPayload from store for internal symlink, got {err:?}"
        );
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
