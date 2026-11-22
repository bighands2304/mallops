//! Explicit, instance-local block registry and block references (ADR-0007 D8).

use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use crate::{Block, BlockSpec, Error};

/// Validated reference to a block: `namespace/stage[.name]` (e.g. `std/ingest.csv`).
///
/// Segments are non-empty and made of `[a-z0-9_-]`. The `wasm:`/`grpc:` prefixes and the
/// `plugin/` namespace are reserved and rejected in v0.1.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub struct BlockRef(String);

fn segment_ok(s: &str) -> bool {
    !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
}

impl BlockRef {
    /// The namespace (the part before `/`).
    #[must_use]
    pub fn namespace(&self) -> &str {
        self.0.split_once('/').map_or("", |(ns, _)| ns)
    }

    /// The name (the part after `/`, i.e. `stage[.name]`).
    #[must_use]
    pub fn name(&self) -> &str {
        self.0.split_once('/').map_or("", |(_, n)| n)
    }
}

impl FromStr for BlockRef {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Error> {
        if s.starts_with("wasm:") || s.starts_with("grpc:") || s.starts_with("plugin/") {
            return Err(Error::Config(format!(
                "block reference `{s}`: wasm:, grpc: and plugin/ blocks are not supported in v0.1"
            )));
        }
        let malformed = || Error::Config(format!("malformed block reference `{s}`"));
        let (ns, name) = s.split_once('/').ok_or_else(malformed)?;
        let mut parts = name.split('.');
        let (stage, sub) = (parts.next().unwrap_or(""), parts.next());
        if !segment_ok(ns)
            || !segment_ok(stage)
            || sub.is_some_and(|n| !segment_ok(n))
            || parts.next().is_some()
        {
            return Err(malformed());
        }
        Ok(Self(s.to_owned()))
    }
}

impl fmt::Display for BlockRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Instance-local registry of blocks, keyed by block id. No global state.
#[derive(Default)]
pub struct BlockRegistry(BTreeMap<String, Arc<dyn Block>>);

impl BlockRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a block under its `spec().id`.
    ///
    /// # Errors
    ///
    /// [`Error::Config`] if the id is not a valid [`BlockRef`]; [`Error::Validation`] if a
    /// block with the same id is already registered (the original is kept).
    pub fn register(&mut self, block: Arc<dyn Block>) -> Result<(), Error> {
        let id = block.spec().id.to_string();
        id.parse::<BlockRef>()?;
        if self.0.contains_key(&id) {
            return Err(Error::Validation(format!(
                "block `{id}` is already registered"
            )));
        }
        self.0.insert(id, block);
        Ok(())
    }

    /// Returns the block registered under `r`, if any.
    #[must_use]
    pub fn get(&self, r: &BlockRef) -> Option<Arc<dyn Block>> {
        self.0.get(&r.0).cloned()
    }

    /// Whether a block is registered under `r`.
    #[must_use]
    pub fn contains(&self, r: &BlockRef) -> bool {
        self.0.contains_key(&r.0)
    }

    /// Specs of all registered blocks, sorted by id.
    #[must_use]
    pub fn list(&self) -> Vec<&BlockSpec> {
        self.0.values().map(|b| b.spec()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BlockOutcome, BoxFuture, Context, Inputs, OutputDrafts, Stage};

    struct Dummy(BlockSpec);
    impl Block for Dummy {
        fn spec(&self) -> &BlockSpec {
            &self.0
        }
        fn run<'a>(
            &'a self,
            _: &'a Context,
            _: Inputs,
        ) -> BoxFuture<'a, Result<BlockOutcome, Error>> {
            Box::pin(async { Ok(BlockOutcome::Produced(OutputDrafts::new())) })
        }
    }

    fn dummy(id: &'static str, version: &'static str) -> Arc<dyn Block> {
        Arc::new(Dummy(BlockSpec::new(id, version, Stage::Custom)))
    }

    fn r(s: &str) -> BlockRef {
        s.parse().unwrap()
    }

    #[test]
    fn register_and_get() {
        let mut reg = BlockRegistry::new();
        reg.register(dummy("std/ingest.csv", "1")).unwrap();
        reg.register(dummy("std/package", "1")).unwrap();
        for id in ["std/ingest.csv", "std/package"] {
            assert!(reg.contains(&r(id)));
            assert_eq!(reg.get(&r(id)).unwrap().spec().id, id);
        }
        assert!(reg.get(&r("std/train")).is_none());
        assert!(!reg.contains(&r("std/train")));
    }

    #[test]
    fn duplicate_keeps_original() {
        let mut reg = BlockRegistry::new();
        reg.register(dummy("std/package", "1")).unwrap();
        let err = reg.register(dummy("std/package", "2")).unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
        assert_eq!(reg.get(&r("std/package")).unwrap().spec().version, "1");
    }

    #[test]
    fn invalid_spec_id_rejected() {
        let mut reg = BlockRegistry::new();
        assert!(matches!(
            reg.register(dummy("Bad Id", "1")),
            Err(Error::Config(_))
        ));
    }

    #[test]
    fn list_sorted_by_id() {
        let mut reg = BlockRegistry::new();
        for id in ["std/train", "core/noop", "std/ingest.csv"] {
            reg.register(dummy(id, "1")).unwrap();
        }
        let ids: Vec<_> = reg.list().iter().map(|s| s.id.to_string()).collect();
        assert_eq!(ids, ["core/noop", "std/ingest.csv", "std/train"]);
    }

    #[test]
    fn instances_isolated_concurrently() {
        let handles: Vec<_> = (0..8)
            .map(|i| {
                std::thread::spawn(move || {
                    let mut reg = BlockRegistry::new();
                    assert!(reg.list().is_empty());
                    reg.register(dummy("std/package", "1")).unwrap();
                    assert_eq!(reg.list().len(), 1, "thread {i}");
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn parse_valid() {
        let b = r("std/ingest.csv");
        assert_eq!((b.namespace(), b.name()), ("std", "ingest.csv"));
        assert_eq!(b.to_string(), "std/ingest.csv");
        assert_eq!(r("std/package").name(), "package");
        assert_eq!(r("core/noop").namespace(), "core");
    }

    #[test]
    fn parse_malformed() {
        for s in [
            "",
            "std",
            "std/",
            "/package",
            "Std/package",
            "std/ingest.",
            "std/.csv",
            "std/a.b.c",
            "std/pack age",
            "std//x",
            "a/b/c",
        ] {
            match s.parse::<BlockRef>() {
                Err(Error::Config(m)) => {
                    assert!(m.contains("malformed block reference"), "{s}: {m}")
                }
                other => panic!("{s}: {other:?}"),
            }
        }
    }

    #[test]
    fn parse_unsupported_plugins() {
        for s in ["wasm:mod.wasm", "grpc:127.0.0.1:50051", "plugin/custom"] {
            match s.parse::<BlockRef>() {
                Err(Error::Config(m)) => assert!(m.contains("not supported in v0.1"), "{s}: {m}"),
                other => panic!("{s}: {other:?}"),
            }
        }
    }
}
