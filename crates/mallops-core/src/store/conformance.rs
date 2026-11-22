//! Shared conformance suite for [`ArtifactStore`] implementations.
//!
//! Invoke `conformance_tests!(factory)` from an implementation's test module, where
//! `factory` is an expression callable as `factory()` returning `(store, guard)`.
//! `guard` is kept alive for the whole test (e.g. a `tempfile::TempDir` backing an
//! on-disk store). Each case becomes a `#[tokio::test]` named `store_*`.
//!
//! Implementation-specific checks (such as on-disk tamper detection) do not belong here.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::{ArtifactStore, PayloadSource, StoreError};
use crate::{ArtifactKind, Error, Manifest, RelPath};

fn manifest() -> Manifest {
    Manifest::new(ArtifactKind::Blob, "bin")
}

fn rel(s: &str) -> RelPath {
    RelPath::parse(s).unwrap()
}

fn files<'a>(entries: &[(&str, &'a [u8])]) -> PayloadSource<'a> {
    PayloadSource::Files(entries.iter().map(|(p, b)| (rel(p), *b)).collect())
}

pub(crate) async fn duplicate_put_is_idempotent<S: ArtifactStore>(store: &S) {
    let a = store
        .put(manifest(), PayloadSource::Bytes(b"payload"))
        .await
        .unwrap();
    let b = store
        .put(manifest(), PayloadSource::Bytes(b"payload"))
        .await
        .unwrap();
    assert_eq!(a, b);
    // Bytes is stored as the single file `data`, so the multi-file form is equivalent.
    let c = store
        .put(manifest(), files(&[("data", b"payload")]))
        .await
        .unwrap();
    assert_eq!(a, c);
    let other = store
        .put(manifest(), PayloadSource::Bytes(b"other"))
        .await
        .unwrap();
    assert_ne!(a, other);

    assert!(store.exists(&a).await.unwrap());
    let artifact = store.get(&a).await.unwrap().unwrap();
    assert_eq!(artifact.id(), a);
    assert_eq!(artifact.manifest(), &manifest());
    store.verify(&a).await.unwrap();
}

pub(crate) async fn concurrent_put_is_idempotent<S: ArtifactStore + 'static>(store: S) {
    let store = Arc::new(store);
    let tasks: Vec<_> = (0..8)
        .map(|_| {
            let store = Arc::clone(&store);
            tokio::spawn(async move {
                store
                    .put(
                        manifest(),
                        files(&[("a.bin", b"aaa"), ("sub/b.bin", b"bbb")]),
                    )
                    .await
            })
        })
        .collect();
    let mut ids = Vec::new();
    for t in tasks {
        ids.push(t.await.unwrap().unwrap());
    }
    assert!(ids.windows(2).all(|w| w[0] == w[1]));
    store.verify(&ids[0]).await.unwrap();
}

pub(crate) async fn unknown_id_is_none<S: ArtifactStore>(store: &S) {
    let id = crate::ArtifactId::from_bytes([7u8; 32]);
    assert!(store.get(&id).await.unwrap().is_none());
    assert!(!store.exists(&id).await.unwrap());
    let dest = tempfile::tempdir().unwrap();
    let err = store
        .fetch(&id, &dest.path().join("out"))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Store(StoreError::NotFound(f)) if f == id));
    let err = store.verify(&id).await.unwrap_err();
    assert!(matches!(err, Error::Store(StoreError::NotFound(f)) if f == id));
}

pub(crate) async fn fetch_is_an_isolated_copy<S: ArtifactStore>(store: &S) {
    let id = store
        .put(
            manifest(),
            files(&[("a.bin", b"aaa"), ("sub/dir/b.bin", b"bbb")]),
        )
        .await
        .unwrap();

    let tmp = tempfile::tempdir().unwrap();
    let dest = tmp.path().join("out");
    let got = store.fetch(&id, &dest).await.unwrap();
    assert_eq!(got, dest);
    assert_eq!(std::fs::read(dest.join("a.bin")).unwrap(), b"aaa");
    assert_eq!(std::fs::read(dest.join("sub/dir/b.bin")).unwrap(), b"bbb");

    // Mutate and delete the materialized copy.
    std::fs::write(dest.join("a.bin"), b"changed").unwrap();
    std::fs::remove_file(dest.join("sub/dir/b.bin")).unwrap();

    store.verify(&id).await.unwrap();
    let dest2 = tmp.path().join("out2");
    store.fetch(&id, &dest2).await.unwrap();
    assert_eq!(std::fs::read(dest2.join("a.bin")).unwrap(), b"aaa");
    assert_eq!(std::fs::read(dest2.join("sub/dir/b.bin")).unwrap(), b"bbb");
}

pub(crate) async fn put_rejects_symlinks<S: ArtifactStore>(store: &S) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::tempdir().unwrap();
        let outside = tmp.path().join("outside");
        std::fs::create_dir(&outside).unwrap();
        std::fs::write(outside.join("secret"), b"secret").unwrap();

        // Internal symlink (target inside the payload dir).
        let internal = tmp.path().join("internal");
        std::fs::create_dir(&internal).unwrap();
        std::fs::write(internal.join("real.bin"), b"x").unwrap();
        symlink("real.bin", internal.join("link.bin")).unwrap();

        // External symlink to a file outside the payload dir.
        let external = tmp.path().join("external");
        std::fs::create_dir(&external).unwrap();
        std::fs::write(external.join("real.bin"), b"x").unwrap();
        symlink(outside.join("secret"), external.join("link.bin")).unwrap();

        // Symlinked directory escaping to an ancestor.
        let escape = tmp.path().join("escape");
        std::fs::create_dir(&escape).unwrap();
        symlink("..", escape.join("up")).unwrap();

        for dir in [&internal, &external, &escape] {
            let err = store
                .put(manifest(), PayloadSource::Dir(dir))
                .await
                .unwrap_err();
            assert!(
                matches!(err, Error::Store(StoreError::InvalidPayload(_))),
                "{}: {err:?}",
                dir.display()
            );
        }

        // A symlink passed directly as a File payload.
        let err = store
            .put(manifest(), PayloadSource::File(&external.join("link.bin")))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Store(StoreError::InvalidPayload(_))));
    }
    #[cfg(not(unix))]
    let _ = store;
}

pub(crate) async fn put_rejects_case_collisions<S: ArtifactStore>(store: &S) {
    let mut map: BTreeMap<RelPath, &[u8]> = BTreeMap::new();
    map.insert(rel("A.bin"), b"1");
    map.insert(rel("a.bin"), b"2");
    let err = store
        .put(manifest(), PayloadSource::Files(map))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::InvalidPath(_)), "{err:?}");
}

pub(crate) async fn meta_round_trip<S: ArtifactStore>(store: &S) {
    let cache = rel("cache/abc.json");
    let lineage = rel("runs/r1/lineage.json");
    let lineage10 = rel("runs/r10/lineage.json");
    let cache2 = rel("cache2/x.json");

    assert!(store.get_meta(&cache).await.unwrap().is_none());

    store.put_meta(&cache, b"{\"v\":1}").await.unwrap();
    store.put_meta(&lineage, b"l1").await.unwrap();
    store.put_meta(&lineage10, b"l10").await.unwrap();
    store.put_meta(&cache2, b"c2").await.unwrap();
    assert_eq!(store.get_meta(&cache).await.unwrap().unwrap(), b"{\"v\":1}");
    assert_eq!(store.get_meta(&lineage).await.unwrap().unwrap(), b"l1");

    // Last write wins.
    store.put_meta(&cache, b"{\"v\":2}").await.unwrap();
    assert_eq!(store.get_meta(&cache).await.unwrap().unwrap(), b"{\"v\":2}");

    // Prefix matches on a segment boundary, sorted.
    assert_eq!(
        store.list_meta(&rel("cache")).await.unwrap(),
        std::slice::from_ref(&cache)
    );
    assert_eq!(
        store.list_meta(&rel("runs/r1")).await.unwrap(),
        std::slice::from_ref(&lineage)
    );
    assert_eq!(
        store.list_meta(&rel("runs")).await.unwrap(),
        [lineage.clone(), lineage10.clone()]
    );
    assert_eq!(
        store.list_meta(&cache).await.unwrap(),
        std::slice::from_ref(&cache)
    );
    assert!(store.list_meta(&rel("cac")).await.unwrap().is_empty());
    assert!(store.list_meta(&rel("nope")).await.unwrap().is_empty());

    // Delete is idempotent.
    store.delete_meta(&cache).await.unwrap();
    store.delete_meta(&cache).await.unwrap();
    store.delete_meta(&rel("never/existed.json")).await.unwrap();
    assert!(store.get_meta(&cache).await.unwrap().is_none());
    assert!(store.list_meta(&rel("cache")).await.unwrap().is_empty());
    assert!(store.get_meta(&lineage).await.unwrap().is_some());
}

/// Generates the conformance `#[tokio::test]`s for an [`ArtifactStore`] implementation.
macro_rules! conformance_tests {
    ($factory:expr) => {
        $crate::store::conformance::conformance_tests!(@case $factory, store_duplicate_put_is_idempotent, duplicate_put_is_idempotent, ref);
        $crate::store::conformance::conformance_tests!(@case $factory, store_concurrent_put_is_idempotent, concurrent_put_is_idempotent, owned);
        $crate::store::conformance::conformance_tests!(@case $factory, store_unknown_id_is_none, unknown_id_is_none, ref);
        $crate::store::conformance::conformance_tests!(@case $factory, store_fetch_is_an_isolated_copy, fetch_is_an_isolated_copy, ref);
        $crate::store::conformance::conformance_tests!(@case $factory, store_put_rejects_symlinks, put_rejects_symlinks, ref);
        $crate::store::conformance::conformance_tests!(@case $factory, store_put_rejects_case_collisions, put_rejects_case_collisions, ref);
        $crate::store::conformance::conformance_tests!(@case $factory, store_meta_round_trip, meta_round_trip, ref);
    };
    (@case $factory:expr, $name:ident, $case:ident, ref) => {
        #[tokio::test]
        async fn $name() {
            let (store, _guard) = ($factory)();
            $crate::store::conformance::$case(&store).await;
        }
    };
    (@case $factory:expr, $name:ident, $case:ident, owned) => {
        #[tokio::test]
        async fn $name() {
            let (store, _guard) = ($factory)();
            $crate::store::conformance::$case(store).await;
        }
    };
}
pub(crate) use conformance_tests;
