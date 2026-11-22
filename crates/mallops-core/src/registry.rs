//! Explicit, instance-local block registry and block references (ADR-0007 D8).

use std::collections::{BTreeMap, btree_map::Entry};
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use crate::{Block, BlockSpec, Error};

/// Validated reference to a block: `namespace/stage[.variant]` (e.g. `std/ingest.csv`).
///
/// Segments start with `[a-z0-9]` and then contain only `[a-z0-9_-]`. The `wasm:`/`grpc:` prefixes and the
/// `plugin/` namespace are reserved and rejected in v0.1.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub struct BlockRef(String);

fn segment_ok(s: &str) -> bool {
    !s.is_empty()
        && s.bytes().enumerate().all(|(i, b)| {
            b.is_ascii_lowercase() || b.is_ascii_digit() || (i > 0 && (b == b'_' || b == b'-'))
        })
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
        let (stage, sub) = name
            .split_once('.')
            .map_or((name, None), |(a, b)| (a, Some(b)));
        if !segment_ok(ns) || !segment_ok(stage) || sub.is_some_and(|n| !segment_ok(n)) {
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
pub struct BlockRegistry(BTreeMap<BlockRef, Arc<dyn Block>>);

impl fmt::Debug for BlockRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.0.keys()).finish()
    }
}

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
        let id: BlockRef = block.spec().id.parse()?;
        match self.0.entry(id) {
            Entry::Occupied(e) => Err(Error::Validation(format!(
                "block `{}` is already registered",
                e.key()
            ))),
            Entry::Vacant(e) => {
                e.insert(block);
                Ok(())
            }
        }
    }

    /// Returns the block registered under `r`, if any.
    #[must_use]
    pub fn get(&self, r: &BlockRef) -> Option<Arc<dyn Block>> {
        self.0.get(r).cloned()
    }

    /// Whether a block is registered under `r`.
    #[must_use]
    pub fn contains(&self, r: &BlockRef) -> bool {
        self.0.contains_key(r)
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
                    let id: &'static str = Box::leak(format!("std/custom.t{i}").into_boxed_str());
                    let mut reg = BlockRegistry::new();
                    reg.register(dummy(id, "1")).unwrap();
                    let ids: Vec<_> = reg.list().iter().map(|s| s.id.to_string()).collect();
                    assert_eq!(ids, [id]);
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn parse_valid() {
        for s in [
            "std/ingest.csv",
            "std/package",
            "core/noop",
            "std/custom.t1",
        ] {
            assert_eq!(r(s).to_string(), s);
        }
    }

    fn err_msg(s: &str) -> String {
        match s.parse::<BlockRef>() {
            Err(Error::Config(m)) => m,
            other => panic!("{s}: {other:?}"),
        }
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
            "-/package",
            "std/-ingest",
        ] {
            assert!(err_msg(s).contains("malformed block reference"), "{s}");
        }
    }

    #[test]
    fn parse_unsupported_plugins() {
        for s in ["wasm:mod.wasm", "grpc:127.0.0.1:50051", "plugin/custom"] {
            assert!(err_msg(s).contains("not supported in v0.1"), "{s}");
        }
    }
}
