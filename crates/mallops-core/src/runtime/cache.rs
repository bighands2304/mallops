//! Content-hash cache of node results (SPEC NFR-2, ARCHITECTURE §4, §6.1).
//!
//! A node's cache key covers everything that determines its outputs; the entry
//! `cache/<key-hex>.json` in the store metadata maps output ports to committed artifact ids.

use std::collections::BTreeMap;

use crate::{ArtifactId, ArtifactStore, CachePolicy, Digest, Error, ValidatedNode};

/// Domain-separated hash of a node's `with:` config, in canonical JSON (sorted keys).
///
/// `${secret:NAME}` references stay literal in the config, so secret values never enter it.
pub(crate) fn config_hash(_with: &serde_json::Value) -> Digest {
    todo!("8.1")
}

/// Cache key of `node` run with `inputs` (`(port, artifact id)` pairs) under `policy`.
///
/// `None` when the policy is [`CachePolicy::Uncacheable`].
pub(crate) fn cache_key(
    _node: &ValidatedNode,
    _inputs: &[(String, ArtifactId)],
    _policy: &CachePolicy,
) -> Option<Digest> {
    todo!("8.1")
}

/// Outputs recorded under `key`, or `None` on a miss (no entry, unreadable entry,
/// or a referenced artifact no longer in the store).
pub(crate) async fn lookup(
    _store: &dyn ArtifactStore,
    _key: &Digest,
) -> Result<Option<BTreeMap<String, ArtifactId>>, Error> {
    todo!("8.1")
}

/// Records the committed `outputs` of a succeeded node under `key`.
pub(crate) async fn record(
    _store: &dyn ArtifactStore,
    _key: &Digest,
    _outputs: &BTreeMap<String, ArtifactId>,
) -> Result<(), Error> {
    todo!("8.1")
}
