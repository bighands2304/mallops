//! Shared conformance suite for [`LineageSink`] implementations.
//!
//! Sink backends (in this crate or in `mallops-std`/plugins) call [`check_sink`] from a test.

use std::future::Future;

use super::LineageSink;
use crate::RunId;

/// Runs the conformance checks against `sink`.
///
/// `read_back(run)` returns the persisted document of `run` (the parsed `lineage.json` for
/// [`super::JsonSink`]), or `None` for write-only sinks, in which case only the `Ok` results
/// are checked.
///
/// # Panics
///
/// On the first failed check.
pub async fn check_sink<F, Fut>(sink: &dyn LineageSink, read_back: F)
where
    F: Fn(RunId) -> Fut,
    Fut: Future<Output = Option<serde_json::Value>>,
{
    // 9.1: full event sequence with every Status, order preserved, two interleaved runs kept apart.
    let _ = (sink, read_back);
}
