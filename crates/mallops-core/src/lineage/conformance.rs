//! Shared conformance suite for [`LineageSink`] implementations.
//!
//! Sink backends (in this crate or in `mallops-std`/plugins) call [`check_sink`] from a test.

use std::future::Future;
use std::time::{Duration, UNIX_EPOCH};

use super::{HostInfo, LineageEvent, LineageSink, Timestamp};
use crate::{ArtifactId, BlockRef, Digest, RunId, Status};

fn ts(offset_ms: u64) -> Timestamp {
    // Base: 2026-11-23T10:00:00.000Z (1_795_428_000 s).
    Timestamp::new(
        UNIX_EPOCH + Duration::from_secs(1_795_428_000) + Duration::from_millis(offset_ms),
    )
}

fn parse_block(s: &str) -> BlockRef {
    match s.parse() {
        Ok(b) => b,
        Err(err) => panic!("valid test block ref `{s}` failed to parse: {err}"),
    }
}

async fn record_ok(sink: &dyn LineageSink, event: &LineageEvent, label: &str) {
    if let Err(err) = sink.record(event).await {
        panic!("sink.record({label}) failed: {err}");
    }
}

fn assert_doc_header(doc: &serde_json::Value, run: RunId) -> &[serde_json::Value] {
    assert_eq!(
        doc["v"], 1,
        "expected lineage document version 1, got {doc}"
    );
    assert_eq!(
        doc["run"],
        run.to_string(),
        "expected lineage document run {run}, got {doc}"
    );
    match doc["events"].as_array() {
        Some(events) => events.as_slice(),
        None => panic!("expected `events` array in lineage document, got {doc}"),
    }
}

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
    check_full_sequence(sink, &read_back).await;
    check_interleaved_runs(sink, &read_back).await;
}

async fn check_full_sequence<F, Fut>(sink: &dyn LineageSink, read_back: &F)
where
    F: Fn(RunId) -> Fut,
    Fut: Future<Output = Option<serde_json::Value>>,
{
    let run = RunId::new();
    let pipeline_hash = Digest::from_bytes([0x11; 32]);
    let config_hash = Digest::from_bytes([0x22; 32]);
    let in_art = ArtifactId::from_bytes([0x33; 32]);
    let out_art = ArtifactId::from_bytes([0x44; 32]);
    let host = HostInfo {
        os: "linux".into(),
        arch: "x86_64".into(),
        hostname: "conformance-host".into(),
        mallops_version: env!("CARGO_PKG_VERSION").into(),
    };

    let events = [
        LineageEvent::RunStarted {
            run,
            timestamp: ts(100),
            pipeline_hash,
            git_commit: Some("deadbeef".into()),
            host: host.clone(),
        },
        LineageEvent::BlockStarted {
            run,
            timestamp: ts(110),
            node: "n_ok".into(),
            block: parse_block("std/train.external"),
            block_version: "1.2.3".into(),
            config_hash,
            seed: Some(42),
            inputs: vec![in_art],
        },
        LineageEvent::ArtifactProduced {
            run,
            timestamp: ts(120),
            node: "n_ok".into(),
            artifact: out_art,
        },
        LineageEvent::BlockFinished {
            run,
            timestamp: ts(130),
            node: "n_ok".into(),
            status: Status::Succeeded,
            duration: Duration::from_millis(20),
            cached: false,
        },
        LineageEvent::BlockFinished {
            run,
            timestamp: ts(140),
            node: "n_skip".into(),
            status: Status::Skipped,
            duration: Duration::ZERO,
            cached: false,
        },
        LineageEvent::BlockFinished {
            run,
            timestamp: ts(150),
            node: "n_fail".into(),
            status: Status::Failed,
            duration: Duration::from_millis(7),
            cached: false,
        },
        LineageEvent::BlockFinished {
            run,
            timestamp: ts(160),
            node: "n_cancel".into(),
            status: Status::Cancelled,
            duration: Duration::ZERO,
            cached: false,
        },
        LineageEvent::RunFinished {
            run,
            timestamp: ts(170),
            status: Status::Failed,
        },
    ];

    for (idx, event) in events.iter().enumerate() {
        record_ok(sink, event, &format!("full_sequence[{idx}]")).await;
    }

    let Some(doc) = read_back(run).await else {
        return;
    };

    let persisted = assert_doc_header(&doc, run);
    assert_eq!(
        persisted.len(),
        events.len(),
        "expected {} events, got {}",
        events.len(),
        persisted.len()
    );

    let run_str = run.to_string();

    // 0: RunStarted
    let ev0 = &persisted[0];
    assert_eq!(ev0["event"], "run_started");
    assert_eq!(ev0["run"], run_str);
    assert_eq!(ev0["timestamp"], "2026-11-23T10:00:00.100Z");
    assert_eq!(ev0["pipeline_hash"], pipeline_hash.to_hex());
    assert_eq!(ev0["git_commit"], "deadbeef");
    assert_eq!(ev0["host"]["os"], host.os);
    assert_eq!(ev0["host"]["arch"], host.arch);
    assert_eq!(ev0["host"]["hostname"], host.hostname);
    assert_eq!(ev0["host"]["mallops_version"], host.mallops_version);

    // 1: BlockStarted
    let ev1 = &persisted[1];
    assert_eq!(ev1["event"], "block_started");
    assert_eq!(ev1["run"], run_str);
    assert_eq!(ev1["timestamp"], "2026-11-23T10:00:00.110Z");
    assert_eq!(ev1["node"], "n_ok");
    assert_eq!(ev1["block"], "std/train.external");
    assert_eq!(ev1["block_version"], "1.2.3");
    assert_eq!(ev1["config_hash"], config_hash.to_hex());
    assert_eq!(ev1["seed"], 42);
    assert_eq!(ev1["inputs"], serde_json::json!([in_art.to_hex()]));

    // 2: ArtifactProduced
    let ev2 = &persisted[2];
    assert_eq!(ev2["event"], "artifact_produced");
    assert_eq!(ev2["run"], run_str);
    assert_eq!(ev2["timestamp"], "2026-11-23T10:00:00.120Z");
    assert_eq!(ev2["node"], "n_ok");
    assert_eq!(ev2["artifact"], out_art.to_hex());

    // 3..=6: BlockFinished for all 4 Status variants in order
    let expected_finishes = [
        ("n_ok", "succeeded", 20u64, "2026-11-23T10:00:00.130Z"),
        ("n_skip", "skipped", 0u64, "2026-11-23T10:00:00.140Z"),
        ("n_fail", "failed", 7u64, "2026-11-23T10:00:00.150Z"),
        ("n_cancel", "cancelled", 0u64, "2026-11-23T10:00:00.160Z"),
    ];
    for (offset, (node, status, duration_ms, expected_ts)) in
        expected_finishes.into_iter().enumerate()
    {
        let ev = &persisted[3 + offset];
        assert_eq!(ev["event"], "block_finished");
        assert_eq!(ev["run"], run_str);
        assert_eq!(ev["timestamp"], expected_ts);
        assert_eq!(ev["node"], node);
        assert_eq!(ev["status"], status);
        assert_eq!(ev["duration_ms"], duration_ms);
        assert_eq!(ev["cached"], false);
    }

    // 7: RunFinished
    let ev7 = &persisted[7];
    assert_eq!(ev7["event"], "run_finished");
    assert_eq!(ev7["run"], run_str);
    assert_eq!(ev7["timestamp"], "2026-11-23T10:00:00.170Z");
    assert_eq!(ev7["status"], "failed");
}

async fn check_interleaved_runs<F, Fut>(sink: &dyn LineageSink, read_back: &F)
where
    F: Fn(RunId) -> Fut,
    Fut: Future<Output = Option<serde_json::Value>>,
{
    let run_a = RunId::new();
    let run_b = RunId::new();
    assert_ne!(run_a, run_b);

    let hash_a = Digest::from_bytes([0xa1; 32]);
    let hash_b = Digest::from_bytes([0xb1; 32]);
    let cfg_a = Digest::from_bytes([0xa2; 32]);
    let cfg_b = Digest::from_bytes([0xb2; 32]);
    let art_a = ArtifactId::from_bytes([0xa3; 32]);
    let host = HostInfo::current();

    let seq_a = [
        LineageEvent::RunStarted {
            run: run_a,
            timestamp: ts(200),
            pipeline_hash: hash_a,
            git_commit: None,
            host: host.clone(),
        },
        LineageEvent::BlockStarted {
            run: run_a,
            timestamp: ts(210),
            node: "alpha".into(),
            block: parse_block("std/ingest.csv"),
            block_version: "0.1.0".into(),
            config_hash: cfg_a,
            seed: None,
            inputs: vec![],
        },
        LineageEvent::ArtifactProduced {
            run: run_a,
            timestamp: ts(220),
            node: "alpha".into(),
            artifact: art_a,
        },
        LineageEvent::BlockFinished {
            run: run_a,
            timestamp: ts(230),
            node: "alpha".into(),
            status: Status::Succeeded,
            duration: Duration::from_millis(5),
            cached: true,
        },
        LineageEvent::RunFinished {
            run: run_a,
            timestamp: ts(240),
            status: Status::Succeeded,
        },
    ];

    let seq_b = [
        LineageEvent::RunStarted {
            run: run_b,
            timestamp: ts(205),
            pipeline_hash: hash_b,
            git_commit: None,
            host,
        },
        LineageEvent::BlockStarted {
            run: run_b,
            timestamp: ts(215),
            node: "beta".into(),
            block: parse_block("std/package"),
            block_version: "0.2.0".into(),
            config_hash: cfg_b,
            seed: Some(9),
            inputs: vec![],
        },
        LineageEvent::BlockFinished {
            run: run_b,
            timestamp: ts(225),
            node: "beta".into(),
            status: Status::Skipped,
            duration: Duration::ZERO,
            cached: false,
        },
        LineageEvent::RunFinished {
            run: run_b,
            timestamp: ts(235),
            status: Status::Succeeded,
        },
    ];

    // Interleave sequentially (core has no tokio `macros` feature for `tokio::join!`).
    // `run_b` finishes before `run_a` to verify cleanup of one run never disturbs the other.
    let max_len = seq_a.len().max(seq_b.len());
    for idx in 0..max_len {
        if let Some(ev) = seq_a.get(idx) {
            record_ok(sink, ev, &format!("run_a[{idx}]")).await;
        }
        if let Some(ev) = seq_b.get(idx) {
            record_ok(sink, ev, &format!("run_b[{idx}]")).await;
        }
    }

    if let Some(doc_a) = read_back(run_a).await {
        let evs = assert_doc_header(&doc_a, run_a);
        assert_eq!(evs.len(), seq_a.len());
        let expected_tags = [
            "run_started",
            "block_started",
            "artifact_produced",
            "block_finished",
            "run_finished",
        ];
        for (ev, tag) in evs.iter().zip(expected_tags) {
            assert_eq!(ev["event"], tag);
            assert_eq!(ev["run"], run_a.to_string());
        }
        assert_eq!(evs[1]["node"], "alpha");
        assert_eq!(evs[1]["config_hash"], cfg_a.to_hex());
        assert_eq!(evs[3]["cached"], true);
    }

    if let Some(doc_b) = read_back(run_b).await {
        let evs = assert_doc_header(&doc_b, run_b);
        assert_eq!(evs.len(), seq_b.len());
        let expected_tags = [
            "run_started",
            "block_started",
            "block_finished",
            "run_finished",
        ];
        for (ev, tag) in evs.iter().zip(expected_tags) {
            assert_eq!(ev["event"], tag);
            assert_eq!(ev["run"], run_b.to_string());
        }
        assert_eq!(evs[1]["node"], "beta");
        assert_eq!(evs[1]["seed"], 9);
        assert_eq!(evs[1]["config_hash"], cfg_b.to_hex());
        assert_eq!(evs[2]["status"], "skipped");
    }
}
