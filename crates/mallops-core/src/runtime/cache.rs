//! Content-hash cache of node results (SPEC NFR-2, ARCHITECTURE §4, §6.1).
//!
//! A node's cache key covers everything that determines its outputs; the entry
//! `cache/<key-hex>.json` in the store metadata maps output ports to committed artifact ids.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::artifact::canonical_json;
use crate::{ArtifactId, ArtifactStore, CachePolicy, Digest, Error, RelPath, ValidatedNode};

/// Domain-separation context for [`config_hash`].
const CONFIG_HASH_V1_CONTEXT: &str = "mallops config v1";

/// Domain-separation context for [`cache_key`].
const CACHE_KEY_V1_CONTEXT: &str = "mallops cache v1";

/// Format version stored in `cache/<key-hex>.json`.
const CACHE_ENTRY_VERSION: u32 = 1;

fn update_len_prefixed(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn entry_path(key: &Digest) -> Result<RelPath, Error> {
    RelPath::parse(&format!("cache/{key}.json"))
}

#[derive(Debug, Serialize, Deserialize)]
struct CacheEntry {
    v: u32,
    outputs: BTreeMap<String, ArtifactId>,
}

/// Domain-separated hash of a node's `with:` config, in canonical JSON (sorted keys).
///
/// `${secret:NAME}` references stay literal in the config, so secret values never enter it.
pub(crate) fn config_hash(with: &serde_json::Value) -> Digest {
    let bytes = canonical_json(with);
    Digest::from_bytes(blake3::derive_key(CONFIG_HASH_V1_CONTEXT, &bytes))
}

/// Cache key of `node` run with `inputs` (`(port, artifact id)` pairs) under `policy`.
///
/// `None` when the policy is [`CachePolicy::Uncacheable`].
pub(crate) fn cache_key(
    node: &ValidatedNode,
    inputs: &[(String, ArtifactId)],
    policy: &CachePolicy,
) -> Option<Digest> {
    let extra_hash = match policy {
        CachePolicy::Deterministic { extra_hash } => extra_hash,
        CachePolicy::Uncacheable => return None,
    };

    let spec = node.block.spec();
    let cfg_hash = config_hash(&node.config.with);

    let mut hasher = blake3::Hasher::new_derive_key(CACHE_KEY_V1_CONTEXT);
    update_len_prefixed(&mut hasher, spec.id.as_bytes());
    update_len_prefixed(&mut hasher, spec.version.as_bytes());
    hasher.update(cfg_hash.as_bytes());

    match node.config.seed {
        None => {
            hasher.update(&[0u8]);
        }
        Some(seed) => {
            hasher.update(&[1u8]);
            hasher.update(&seed.to_le_bytes());
        }
    }

    let mut sorted_inputs: Vec<(&str, &ArtifactId)> = inputs
        .iter()
        .map(|(port, id)| (port.as_str(), id))
        .collect();
    sorted_inputs.sort_unstable_by(|a, b| {
        a.0.as_bytes()
            .cmp(b.0.as_bytes())
            .then_with(|| a.1.cmp(b.1))
    });
    hasher.update(&(sorted_inputs.len() as u64).to_le_bytes());
    for (port, id) in sorted_inputs {
        update_len_prefixed(&mut hasher, port.as_bytes());
        hasher.update(id.as_bytes());
    }

    match extra_hash {
        None => {
            hasher.update(&[0u8]);
        }
        Some(digest) => {
            hasher.update(&[1u8]);
            hasher.update(digest.as_bytes());
        }
    }

    Some(Digest::from_bytes(*hasher.finalize().as_bytes()))
}

/// Outputs recorded under `key`, or `None` on a miss (no entry, unreadable entry,
/// or a referenced artifact no longer in the store).
pub(crate) async fn lookup(
    store: &dyn ArtifactStore,
    key: &Digest,
) -> Result<Option<BTreeMap<String, ArtifactId>>, Error> {
    let path = entry_path(key)?;
    let Some(bytes) = store.get_meta(&path).await? else {
        return Ok(None);
    };
    let Ok(entry) = serde_json::from_slice::<CacheEntry>(&bytes) else {
        return Ok(None);
    };
    if entry.v != CACHE_ENTRY_VERSION {
        return Ok(None);
    }
    for id in entry.outputs.values() {
        if !store.exists(id).await? {
            return Ok(None);
        }
    }
    Ok(Some(entry.outputs))
}

/// Records the committed `outputs` of a succeeded node under `key`.
pub(crate) async fn record(
    store: &dyn ArtifactStore,
    key: &Digest,
    outputs: &BTreeMap<String, ArtifactId>,
) -> Result<(), Error> {
    let path = entry_path(key)?;
    let entry = CacheEntry {
        v: CACHE_ENTRY_VERSION,
        outputs: outputs.clone(),
    };
    let bytes = serde_json::to_vec(&entry)
        .map_err(|e| Error::Runtime(format!("cannot serialize cache entry: {e}")))?;
    store.put_meta(&path, &bytes).await
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::config::BlockConfig;
    use crate::{
        Artifact, ArtifactKind, Block, BlockOutcome, BlockSpec, BoxFuture, CancelSource, Context,
        Inputs, LocalStore, Manifest, MemorySecretProvider, MemoryStore, OutputDrafts,
        PayloadSource, ResolvedPorts, RunId, Stage, StoreError,
    };

    struct DummyBlock(BlockSpec);

    impl Block for DummyBlock {
        fn spec(&self) -> &BlockSpec {
            &self.0
        }

        fn run<'a>(
            &'a self,
            _ctx: &'a Context,
            _inputs: Inputs,
        ) -> BoxFuture<'a, Result<BlockOutcome, Error>> {
            Box::pin(async { Ok(BlockOutcome::Produced(OutputDrafts::new())) })
        }
    }

    /// Base node `n` (block `std/step` 0.1.0, `with: {"k":1}`, no seed); tests change one field.
    fn make_node() -> ValidatedNode {
        with_spec("std/step", "0.1.0")
    }

    /// Base node with another block id/version.
    fn with_spec(id: &'static str, version: &'static str) -> ValidatedNode {
        let spec = BlockSpec::new(id, version, Stage::Custom);
        let ports = ResolvedPorts::from(&spec);
        ValidatedNode {
            id: "n".to_owned(),
            block: Arc::new(DummyBlock(spec)),
            config: BlockConfig {
                r#use: "std/step".parse().unwrap(),
                r#in: BTreeMap::new(),
                with: serde_json::json!({ "k": 1 }),
                cache: None,
                seed: None,
            },
            ports,
        }
    }

    /// Store wrapper that can hide selected artifacts from `exists` or inject backend errors.
    struct FaultyStore {
        inner: MemoryStore,
        hidden: Mutex<BTreeSet<ArtifactId>>,
        fail_get_meta: AtomicBool,
        fail_exists: AtomicBool,
        fail_put_meta: AtomicBool,
    }

    impl FaultyStore {
        fn new() -> Self {
            Self {
                inner: MemoryStore::new(),
                hidden: Mutex::new(BTreeSet::new()),
                fail_get_meta: AtomicBool::new(false),
                fail_exists: AtomicBool::new(false),
                fail_put_meta: AtomicBool::new(false),
            }
        }

        fn hide(&self, id: ArtifactId) {
            self.hidden.lock().unwrap().insert(id);
        }
    }

    impl ArtifactStore for FaultyStore {
        fn put<'a>(
            &'a self,
            manifest: Manifest,
            payload: PayloadSource<'a>,
        ) -> BoxFuture<'a, Result<ArtifactId, Error>> {
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
            if self.fail_exists.load(Ordering::SeqCst) {
                return Box::pin(async { Err(StoreError::Backend("exists failed".into()).into()) });
            }
            if self.hidden.lock().unwrap().contains(id) {
                return Box::pin(async { Ok(false) });
            }
            self.inner.exists(id)
        }

        fn put_meta<'a>(
            &'a self,
            key: &'a RelPath,
            data: &'a [u8],
        ) -> BoxFuture<'a, Result<(), Error>> {
            if self.fail_put_meta.load(Ordering::SeqCst) {
                return Box::pin(async {
                    Err(StoreError::Backend("put_meta failed".into()).into())
                });
            }
            self.inner.put_meta(key, data)
        }

        fn get_meta<'a>(
            &'a self,
            key: &'a RelPath,
        ) -> BoxFuture<'a, Result<Option<Vec<u8>>, Error>> {
            if self.fail_get_meta.load(Ordering::SeqCst) {
                return Box::pin(async {
                    Err(StoreError::Backend("get_meta failed".into()).into())
                });
            }
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

    #[test]
    fn cache_config_hash_is_canonical() {
        let a = serde_json::json!({
            "z": [true, false, null, -42, 3.5, "line\n\t\"\\\x01"],
            "a": { "b": 2, "a": 1 }
        });
        let b = serde_json::json!({
            "a": { "a": 1, "b": 2 },
            "z": [true, false, null, -42, 3.5, "line\n\t\"\\\x01"]
        });

        assert_eq!(config_hash(&a), config_hash(&b));
        assert_ne!(
            config_hash(&a),
            config_hash(&serde_json::json!({ "a": { "a": 2, "b": 2 } }))
        );
        // Domain-separated from plain BLAKE3.
        assert_ne!(config_hash(&a), Digest::hash(&canonical_json(&a)));
    }

    #[test]
    fn cache_config_hash_with_secret_ref_is_stable_across_secret_rotation() {
        let with = serde_json::json!({
            "endpoint": "https://api.example.test",
            "token": "${secret:API_TOKEN}"
        });

        let ctx_v1 = Context::new(
            RunId::new(),
            std::env::temp_dir(),
            Arc::new(MemoryStore::new()),
            with.clone(),
            CancelSource::new().token(),
            Arc::new(MemorySecretProvider::default().with_secret("API_TOKEN", "secret-value-v1")),
        );
        let ctx_v2 = Context::new(
            RunId::new(),
            std::env::temp_dir(),
            Arc::new(MemoryStore::new()),
            with,
            CancelSource::new().token(),
            Arc::new(MemorySecretProvider::default().with_secret("API_TOKEN", "secret-value-v2")),
        );

        assert_eq!(
            ctx_v1.secret("API_TOKEN").unwrap().expose(),
            "secret-value-v1"
        );
        assert_eq!(
            ctx_v2.secret("API_TOKEN").unwrap().expose(),
            "secret-value-v2"
        );
        assert_eq!(config_hash(ctx_v1.config()), config_hash(ctx_v2.config()));

        let canonical = String::from_utf8(canonical_json(ctx_v1.config())).unwrap();
        assert!(!canonical.contains("secret-value-v1"));
        assert!(canonical.contains("${secret:API_TOKEN}"));

        // Changing the secret reference name itself does change the hash.
        let other_ref = serde_json::json!({
            "endpoint": "https://api.example.test",
            "token": "${secret:OTHER_TOKEN}"
        });
        assert_ne!(config_hash(ctx_v1.config()), config_hash(&other_ref));
    }

    #[test]
    fn cache_key_changes_for_every_field_and_prevents_boundary_collisions() {
        let id_a = ArtifactId::from_bytes([0x11; 32]);
        let id_b = ArtifactId::from_bytes([0x22; 32]);
        let inputs = vec![("data".to_owned(), id_a)];
        let det = CachePolicy::Deterministic { extra_hash: None };
        let key = |n: &ValidatedNode, i: &[(String, ArtifactId)], p: &CachePolicy| {
            cache_key(n, i, p).unwrap()
        };
        let base = key(&make_node(), &inputs, &det);

        let mut variants = vec![
            key(&with_spec("std/other", "0.1.0"), &inputs, &det),
            key(&with_spec("std/step", "0.2.0"), &inputs, &det),
            key(&make_node(), &[("data".to_owned(), id_b)], &det),
            key(&make_node(), &[("other".to_owned(), id_a)], &det),
            key(&make_node(), &[], &det),
        ];
        let mut node = make_node();
        node.config.with = serde_json::json!({ "k": 2 });
        variants.push(key(&node, &inputs, &det));
        // `seed`: None, Some(0) and Some(1) all differ.
        for seed in [0, 1] {
            let mut node = make_node();
            node.config.seed = Some(seed);
            variants.push(key(&node, &inputs, &det));
        }
        // `extra_hash`: None, Some(h1) and Some(h2) all differ.
        for extra in [b"script-v1", b"script-v2"] {
            let policy = CachePolicy::Deterministic {
                extra_hash: Some(Digest::hash(extra)),
            };
            variants.push(key(&make_node(), &inputs, &policy));
        }
        variants.push(base);
        let distinct: BTreeSet<_> = variants.iter().collect();
        assert_eq!(distinct.len(), variants.len());

        // Length-prefix framing prevents boundary collisions between adjacent strings.
        assert_ne!(
            key(&with_spec("std/ab", "c"), &[], &det),
            key(&with_spec("std/a", "bc"), &[], &det)
        );
    }

    #[test]
    fn cache_key_is_independent_of_input_order() {
        let node = make_node();
        let id = |b: u8| ArtifactId::from_bytes([b; 32]);
        let policy = CachePolicy::Deterministic { extra_hash: None };
        let order_1 = vec![
            ("a".to_owned(), id(0xaa)),
            ("b".to_owned(), id(0xbb)),
            ("c".to_owned(), id(0xcc)),
        ];
        let order_2 = vec![order_1[2].clone(), order_1[0].clone(), order_1[1].clone()];
        assert_eq!(
            cache_key(&node, &order_1, &policy),
            cache_key(&node, &order_2, &policy)
        );
    }

    #[test]
    fn cache_key_ignores_node_id_and_use_ref() {
        let node_a = make_node();
        let mut node_b = make_node();
        node_b.id = "renamed_node".to_owned();
        node_b.config.r#use = "std/alias_b".parse().unwrap();
        let inputs = vec![("train".to_owned(), ArtifactId::from_bytes([0x42; 32]))];
        let policy = CachePolicy::Deterministic { extra_hash: None };
        assert_eq!(
            cache_key(&node_a, &inputs, &policy),
            cache_key(&node_b, &inputs, &policy)
        );
    }

    #[test]
    fn cache_key_uncacheable_returns_none() {
        assert!(cache_key(&make_node(), &[], &CachePolicy::Uncacheable).is_none());
    }

    async fn assert_roundtrip_on_store(store: &dyn ArtifactStore) {
        let key = Digest::hash(b"cache-key-1");

        // Absent entry is a miss.
        assert_eq!(lookup(store, &key).await.unwrap(), None);

        let id_model = store
            .put(
                Manifest::new(ArtifactKind::Model, "onnx"),
                PayloadSource::Bytes(b"model-bytes"),
            )
            .await
            .unwrap();
        let id_metrics = store
            .put(
                Manifest::new(ArtifactKind::Metrics, "json"),
                PayloadSource::Bytes(b"{\"acc\":0.95}"),
            )
            .await
            .unwrap();

        let outputs = BTreeMap::from([
            ("model".to_owned(), id_model),
            ("metrics".to_owned(), id_metrics),
        ]);

        record(store, &key, &outputs).await.unwrap();

        // Check stored metadata path and JSON format.
        let meta_path = RelPath::parse(&format!("cache/{key}.json")).unwrap();
        let raw = store.get_meta(&meta_path).await.unwrap().unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        assert_eq!(
            parsed,
            serde_json::json!({
                "v": 1,
                "outputs": {
                    "metrics": id_metrics.to_hex(),
                    "model": id_model.to_hex(),
                }
            })
        );

        assert_eq!(lookup(store, &key).await.unwrap(), Some(outputs));

        // Empty outputs map also roundtrips cleanly.
        let empty_key = Digest::hash(b"cache-key-empty");
        let empty = BTreeMap::new();
        record(store, &empty_key, &empty).await.unwrap();
        assert_eq!(lookup(store, &empty_key).await.unwrap(), Some(empty));
    }

    #[tokio::test]
    async fn cache_record_and_lookup_roundtrip_memory_and_local_store() {
        let mem = MemoryStore::new();
        assert_roundtrip_on_store(&mem).await;

        let dir = tempfile::tempdir().unwrap();
        let local = LocalStore::new(dir.path()).unwrap();
        assert_roundtrip_on_store(&local).await;
    }

    #[tokio::test]
    async fn cache_lookup_corrupted_or_wrong_version_entry_is_miss() {
        let store = MemoryStore::new();
        let id = store
            .put(
                Manifest::new(ArtifactKind::Blob, "bin"),
                PayloadSource::Bytes(b"payload"),
            )
            .await
            .unwrap();
        let key = Digest::hash(b"corrupt-key");
        let meta_path = RelPath::parse(&format!("cache/{key}.json")).unwrap();

        let bad_payloads: Vec<Vec<u8>> = vec![
            b"{not valid json".to_vec(),
            vec![0xff, 0xfe, 0xfd],
            b"{}".to_vec(),
            b"{\"outputs\":{}}".to_vec(),
            b"{\"v\":1}".to_vec(),
            format!("{{\"v\":2,\"outputs\":{{\"out\":\"{id}\"}}}}").into_bytes(),
            format!("{{\"v\":0,\"outputs\":{{\"out\":\"{id}\"}}}}").into_bytes(),
            b"{\"v\":1,\"outputs\":{\"out\":\"not-a-valid-artifact-id\"}}".to_vec(),
        ];

        for payload in bad_payloads {
            store.put_meta(&meta_path, &payload).await.unwrap();
            assert_eq!(lookup(&store, &key).await.unwrap(), None);
        }
    }

    #[tokio::test]
    async fn cache_lookup_missing_artifact_is_miss() {
        let store = FaultyStore::new();
        let id_a = store
            .put(
                Manifest::new(ArtifactKind::Blob, "bin"),
                PayloadSource::Bytes(b"a"),
            )
            .await
            .unwrap();
        let id_b = store
            .put(
                Manifest::new(ArtifactKind::Blob, "bin"),
                PayloadSource::Bytes(b"b"),
            )
            .await
            .unwrap();
        let key = Digest::hash(b"missing-artifact-key");
        let outputs = BTreeMap::from([("a".to_owned(), id_a), ("b".to_owned(), id_b)]);

        record(&store, &key, &outputs).await.unwrap();
        assert_eq!(lookup(&store, &key).await.unwrap(), Some(outputs));

        // Hiding one referenced output turns lookup into a cache miss.
        store.hide(id_b);
        assert_eq!(lookup(&store, &key).await.unwrap(), None);
    }

    #[tokio::test]
    async fn cache_lookup_and_record_propagate_store_errors() {
        let store = FaultyStore::new();
        let id = store
            .put(
                Manifest::new(ArtifactKind::Blob, "bin"),
                PayloadSource::Bytes(b"x"),
            )
            .await
            .unwrap();
        let key = Digest::hash(b"store-error-key");
        let outputs = BTreeMap::from([("out".to_owned(), id)]);

        store.fail_put_meta.store(true, Ordering::SeqCst);
        assert!(matches!(
            record(&store, &key, &outputs).await,
            Err(Error::Store(StoreError::Backend(_)))
        ));
        store.fail_put_meta.store(false, Ordering::SeqCst);

        record(&store, &key, &outputs).await.unwrap();

        store.fail_get_meta.store(true, Ordering::SeqCst);
        assert!(matches!(
            lookup(&store, &key).await,
            Err(Error::Store(StoreError::Backend(_)))
        ));
        store.fail_get_meta.store(false, Ordering::SeqCst);

        store.fail_exists.store(true, Ordering::SeqCst);
        assert!(matches!(
            lookup(&store, &key).await,
            Err(Error::Store(StoreError::Backend(_)))
        ));
    }
}
