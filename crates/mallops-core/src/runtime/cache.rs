//! Content-hash cache of node results (SPEC NFR-2, ARCHITECTURE §4, §6.1).
//!
//! A node's cache key covers everything that determines its outputs; the entry
//! `cache/<key-hex>.json` in the store metadata maps output ports to committed artifact ids.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::{ArtifactId, ArtifactStore, CachePolicy, Digest, Error, RelPath, ValidatedNode};

/// Domain-separation context for [`config_hash`].
const CONFIG_HASH_V1_CONTEXT: &str = "mallops config v1";

/// Domain-separation context for [`cache_key`].
const CACHE_KEY_V1_CONTEXT: &str = "mallops cache v1";

/// Format version stored in `cache/<key-hex>.json`.
const CACHE_ENTRY_VERSION: u32 = 1;

const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";

fn write_json_string(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000C}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let code = c as u8;
                out.push_str("\\u00");
                out.push(HEX_DIGITS[(code >> 4) as usize] as char);
                out.push(HEX_DIGITS[(code & 0x0f) as usize] as char);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Writes `value` as compact canonical JSON (object keys sorted by UTF-8 byte order at every
/// nesting level, matching `serde_json::to_vec` without `preserve_order`, even if a downstream
/// crate enables `serde_json/preserve_order` via Cargo feature unification).
fn write_canonical_json(out: &mut String, value: &serde_json::Value) {
    match value {
        serde_json::Value::Null => out.push_str("null"),
        serde_json::Value::Bool(true) => out.push_str("true"),
        serde_json::Value::Bool(false) => out.push_str("false"),
        serde_json::Value::Number(n) => {
            let _ = write!(out, "{n}");
        }
        serde_json::Value::String(s) => write_json_string(out, s),
        serde_json::Value::Array(items) => {
            out.push('[');
            for (idx, item) in items.iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                write_canonical_json(out, item);
            }
            out.push(']');
        }
        serde_json::Value::Object(map) => {
            let mut entries: Vec<(&String, &serde_json::Value)> = map.iter().collect();
            entries.sort_unstable_by_key(|(k, _)| *k);
            out.push('{');
            for (idx, (key, val)) in entries.into_iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                write_json_string(out, key);
                out.push(':');
                write_canonical_json(out, val);
            }
            out.push('}');
        }
    }
}

fn canonical_json(value: &serde_json::Value) -> Vec<u8> {
    let mut out = String::new();
    write_canonical_json(&mut out, value);
    out.into_bytes()
}

fn update_len_prefixed(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn entry_path(key: &Digest) -> Result<RelPath, Error> {
    RelPath::parse(&format!("cache/{key}.json"))
}

#[derive(Debug, Deserialize)]
struct CacheEntry {
    v: u32,
    outputs: BTreeMap<String, ArtifactId>,
}

#[derive(Serialize)]
struct CacheEntryRef<'a> {
    v: u32,
    outputs: &'a BTreeMap<String, ArtifactId>,
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
    let entry = CacheEntryRef {
        v: CACHE_ENTRY_VERSION,
        outputs,
    };
    let bytes = serde_json::to_vec(&entry)
        .map_err(|e| Error::Runtime(format!("cannot serialize cache entry: {e}")))?;
    store.put_meta(&path, &bytes).await
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::path::{Path, PathBuf};
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

    fn make_node(
        node_id: &str,
        use_ref: &str,
        spec_id: &'static str,
        spec_version: &'static str,
        with: serde_json::Value,
        seed: Option<u64>,
    ) -> ValidatedNode {
        let spec = BlockSpec::new(spec_id, spec_version, Stage::Custom);
        let ports = ResolvedPorts::from(&spec);
        ValidatedNode {
            id: node_id.to_owned(),
            block: Arc::new(DummyBlock(spec)),
            config: BlockConfig {
                r#use: use_ref.parse().unwrap(),
                r#in: BTreeMap::new(),
                with,
                cache: None,
                seed,
            },
            ports,
        }
    }

    /// Store wrapper that can hide selected artifacts from `exists` or inject backend errors.
    struct FaultyStore {
        inner: MemoryStore,
        hidden: Mutex<BTreeSet<ArtifactId>>,
        fail_get_meta: Mutex<bool>,
        fail_exists: Mutex<bool>,
        fail_put_meta: Mutex<bool>,
    }

    impl FaultyStore {
        fn new() -> Self {
            Self {
                inner: MemoryStore::new(),
                hidden: Mutex::new(BTreeSet::new()),
                fail_get_meta: Mutex::new(false),
                fail_exists: Mutex::new(false),
                fail_put_meta: Mutex::new(false),
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
            if *self.fail_exists.lock().unwrap() {
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
            if *self.fail_put_meta.lock().unwrap() {
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
            if *self.fail_get_meta.lock().unwrap() {
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
    fn cache_config_hash_is_canonical_and_matches_serde_json() {
        let a = serde_json::json!({
            "z": [true, false, null, -42, 3.5, "line\n\t\"\\\x01"],
            "a": { "b": 2, "a": 1 }
        });
        let b = serde_json::json!({
            "a": { "a": 1, "b": 2 },
            "z": [true, false, null, -42, 3.5, "line\n\t\"\\\x01"]
        });

        assert_eq!(canonical_json(&a), serde_json::to_vec(&a).unwrap());
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

    #[cfg(feature = "yaml")]
    #[test]
    fn cache_config_hash_yaml_secret_ref_stable_across_env_changes() {
        use crate::config::PipelineConfig;

        let src = r#"
version: 1
name: p
blocks:
  fetch:
    use: std/ingest.http
    with:
      token: "${secret:API_TOKEN}"
      env_val: "${env:REGION}"
"#;
        let c1 = PipelineConfig::from_yaml_str(src, "p.yaml", &|k| match k {
            "REGION" => Some("eu".into()),
            "API_TOKEN" => Some("leaked-v1".into()),
            _ => None,
        })
        .unwrap();
        let c2 = PipelineConfig::from_yaml_str(src, "p.yaml", &|k| match k {
            "REGION" => Some("eu".into()),
            "API_TOKEN" => Some("leaked-v2".into()),
            _ => None,
        })
        .unwrap();

        assert_eq!(
            config_hash(&c1.blocks["fetch"].with),
            config_hash(&c2.blocks["fetch"].with)
        );
    }

    #[test]
    fn cache_key_changes_for_every_field_and_prevents_boundary_collisions() {
        let id_a = ArtifactId::from_bytes([0x11; 32]);
        let id_b = ArtifactId::from_bytes([0x22; 32]);
        let base_inputs = vec![("data".to_owned(), id_a)];
        let det = CachePolicy::Deterministic { extra_hash: None };

        let base_node = make_node(
            "n",
            "std/step",
            "std/step",
            "0.1.0",
            serde_json::json!({ "k": 1 }),
            None,
        );
        let base_key = cache_key(&base_node, &base_inputs, &det).unwrap();

        // BlockSpec.id changes key.
        let diff_block = make_node(
            "n",
            "std/step",
            "std/other",
            "0.1.0",
            serde_json::json!({ "k": 1 }),
            None,
        );
        assert_ne!(
            cache_key(&diff_block, &base_inputs, &det).unwrap(),
            base_key
        );

        // BlockSpec.version changes key.
        let diff_version = make_node(
            "n",
            "std/step",
            "std/step",
            "0.2.0",
            serde_json::json!({ "k": 1 }),
            None,
        );
        assert_ne!(
            cache_key(&diff_version, &base_inputs, &det).unwrap(),
            base_key
        );

        // `with` config changes key.
        let diff_with = make_node(
            "n",
            "std/step",
            "std/step",
            "0.1.0",
            serde_json::json!({ "k": 2 }),
            None,
        );
        assert_ne!(cache_key(&diff_with, &base_inputs, &det).unwrap(), base_key);

        // `seed`: None vs Some(0) vs Some(1) all differ.
        let seed_zero = make_node(
            "n",
            "std/step",
            "std/step",
            "0.1.0",
            serde_json::json!({ "k": 1 }),
            Some(0),
        );
        let seed_one = make_node(
            "n",
            "std/step",
            "std/step",
            "0.1.0",
            serde_json::json!({ "k": 1 }),
            Some(1),
        );
        let k_seed_zero = cache_key(&seed_zero, &base_inputs, &det).unwrap();
        let k_seed_one = cache_key(&seed_one, &base_inputs, &det).unwrap();
        assert_ne!(k_seed_zero, base_key);
        assert_ne!(k_seed_one, base_key);
        assert_ne!(k_seed_zero, k_seed_one);

        // Input artifact id changes key.
        let diff_input_id = vec![("data".to_owned(), id_b)];
        assert_ne!(
            cache_key(&base_node, &diff_input_id, &det).unwrap(),
            base_key
        );

        // Input port name changes key.
        let diff_input_port = vec![("other".to_owned(), id_a)];
        assert_ne!(
            cache_key(&base_node, &diff_input_port, &det).unwrap(),
            base_key
        );

        // Empty vs non-empty inputs differ.
        assert_ne!(cache_key(&base_node, &[], &det).unwrap(), base_key);

        // `extra_hash`: None vs Some(h1) vs Some(h2) all differ.
        let extra_a = CachePolicy::Deterministic {
            extra_hash: Some(Digest::hash(b"script-v1")),
        };
        let extra_b = CachePolicy::Deterministic {
            extra_hash: Some(Digest::hash(b"script-v2")),
        };
        let k_extra_a = cache_key(&base_node, &base_inputs, &extra_a).unwrap();
        let k_extra_b = cache_key(&base_node, &base_inputs, &extra_b).unwrap();
        assert_ne!(k_extra_a, base_key);
        assert_ne!(k_extra_b, base_key);
        assert_ne!(k_extra_a, k_extra_b);

        // Length-prefix framing prevents boundary collisions between adjacent strings.
        let split_a = make_node(
            "n",
            "std/step",
            "std/ab",
            "c",
            serde_json::Value::Null,
            None,
        );
        let split_b = make_node(
            "n",
            "std/step",
            "std/a",
            "bc",
            serde_json::Value::Null,
            None,
        );
        assert_ne!(
            cache_key(&split_a, &[], &det).unwrap(),
            cache_key(&split_b, &[], &det).unwrap()
        );
    }

    #[test]
    fn cache_key_is_independent_of_input_order() {
        let node = make_node(
            "n",
            "std/join",
            "std/join",
            "1.0.0",
            serde_json::json!({}),
            Some(42),
        );
        let id_a = ArtifactId::from_bytes([0xaa; 32]);
        let id_b = ArtifactId::from_bytes([0xbb; 32]);
        let id_c = ArtifactId::from_bytes([0xcc; 32]);
        let policy = CachePolicy::Deterministic { extra_hash: None };

        let order_1 = vec![
            ("a".to_owned(), id_a),
            ("b".to_owned(), id_b),
            ("c".to_owned(), id_c),
        ];
        let order_2 = vec![
            ("c".to_owned(), id_c),
            ("a".to_owned(), id_a),
            ("b".to_owned(), id_b),
        ];

        assert_eq!(
            cache_key(&node, &order_1, &policy),
            cache_key(&node, &order_2, &policy)
        );
    }

    #[test]
    fn cache_key_ignores_node_id_and_use_ref() {
        let node_a = make_node(
            "first_name",
            "std/alias_a",
            "std/train",
            "1.0.0",
            serde_json::json!({ "lr": 0.01 }),
            Some(7),
        );
        let node_b = make_node(
            "renamed_node",
            "std/alias_b",
            "std/train",
            "1.0.0",
            serde_json::json!({ "lr": 0.01 }),
            Some(7),
        );
        let inputs = vec![("train".to_owned(), ArtifactId::from_bytes([0x42; 32]))];
        let policy = CachePolicy::Deterministic { extra_hash: None };

        assert_eq!(
            cache_key(&node_a, &inputs, &policy),
            cache_key(&node_b, &inputs, &policy)
        );
    }

    #[test]
    fn cache_key_uncacheable_returns_none() {
        let node = make_node(
            "n",
            "std/ingest.mqtt",
            "std/ingest.mqtt",
            "1.0.0",
            serde_json::Value::Null,
            None,
        );
        assert!(cache_key(&node, &[], &CachePolicy::Uncacheable).is_none());
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

        *store.fail_put_meta.lock().unwrap() = true;
        assert!(matches!(
            record(&store, &key, &outputs).await,
            Err(Error::Store(StoreError::Backend(_)))
        ));
        *store.fail_put_meta.lock().unwrap() = false;

        record(&store, &key, &outputs).await.unwrap();

        *store.fail_get_meta.lock().unwrap() = true;
        assert!(matches!(
            lookup(&store, &key).await,
            Err(Error::Store(StoreError::Backend(_)))
        ));
        *store.fail_get_meta.lock().unwrap() = false;

        *store.fail_exists.lock().unwrap() = true;
        assert!(matches!(
            lookup(&store, &key).await,
            Err(Error::Store(StoreError::Backend(_)))
        ));
    }
}
