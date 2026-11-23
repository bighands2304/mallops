//! Run lineage: events emitted by the runtime and the sinks that persist them (ADR-0006).
//!
//! Blocks never see lineage: the runtime is the only producer.

pub mod conformance;

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Serialize, Serializer};

use crate::{
    ArtifactId, ArtifactStore, BlockRef, BoxFuture, Digest, Error, RelPath, RunId, Status,
};

/// Format version stored in `runs/<run-id>/lineage.json`.
const LINEAGE_DOC_VERSION: u32 = 1;

/// Wall-clock instant of a lineage event.
///
/// Serializes as an RFC 3339 UTC timestamp with millisecond precision
/// (e.g. `2026-11-23T10:00:00.123Z`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Timestamp(SystemTime);

impl Timestamp {
    /// Wraps a [`SystemTime`].
    #[must_use]
    pub fn new(time: SystemTime) -> Self {
        Self(time)
    }

    /// The wrapped [`SystemTime`].
    #[must_use]
    pub fn system_time(&self) -> SystemTime {
        self.0
    }
}

/// Converts days since 1970-01-01 (proleptic Gregorian calendar) to `(year, month, day)`.
///
/// Howard Hinnant's `civil_from_days` algorithm.
fn civil_from_days(z: i128) -> (i128, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097) as u64; // [0, 146_096]
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = i128::from(yoe) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let y = y + i128::from(m <= 2);
    (y, m as u32, d as u32)
}

fn format_rfc3339_millis(time: SystemTime) -> Result<String, String> {
    let total_ms: i128 = match time.duration_since(UNIX_EPOCH) {
        Ok(dur) => i128::from(dur.as_secs()) * 1_000 + i128::from(dur.subsec_millis()),
        Err(err) => {
            let dur = err.duration();
            let total_ns =
                i128::from(dur.as_secs()) * 1_000_000_000 + i128::from(dur.subsec_nanos());
            (-total_ns).div_euclid(1_000_000)
        }
    };
    let total_secs = total_ms.div_euclid(1_000);
    let millis = total_ms.rem_euclid(1_000) as u32;
    let days = total_secs.div_euclid(86_400);
    let secs_of_day = total_secs.rem_euclid(86_400) as u32;

    let (year, month, day) = civil_from_days(days);
    if !(0..=9999).contains(&year) {
        return Err(format!(
            "timestamp year {year} is outside RFC 3339 range 0000..=9999"
        ));
    }

    let hour = secs_of_day / 3_600;
    let minute = (secs_of_day % 3_600) / 60;
    let second = secs_of_day % 60;

    Ok(format!(
        "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}Z"
    ))
}

impl Serialize for Timestamp {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let formatted = format_rfc3339_millis(self.0).map_err(serde::ser::Error::custom)?;
        serializer.serialize_str(&formatted)
    }
}

fn serialize_display<T: fmt::Display, S: Serializer>(
    value: &T,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.collect_str(value)
}

fn serialize_duration_ms<S: Serializer>(
    duration: &Duration,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let ms = u64::try_from(duration.as_millis()).unwrap_or(u64::MAX);
    serializer.serialize_u64(ms)
}

/// Host a run executed on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[non_exhaustive]
pub struct HostInfo {
    /// Operating system ([`std::env::consts::OS`]).
    pub os: String,
    /// CPU architecture ([`std::env::consts::ARCH`]).
    pub arch: String,
    /// Host name, `"unknown"` when not available.
    pub hostname: String,
    /// Version of `mallops-core`.
    pub mallops_version: String,
}

impl HostInfo {
    /// Describes the current host.
    #[must_use]
    pub fn current() -> Self {
        // ponytail: hostname from env only (no unsafe/libc); "unknown" when unset (common on macOS).
        let hostname = std::env::var("HOSTNAME")
            .ok()
            .filter(|s| !s.is_empty())
            .or_else(|| std::env::var("COMPUTERNAME").ok().filter(|s| !s.is_empty()))
            .unwrap_or_else(|| "unknown".into());
        Self {
            os: std::env::consts::OS.into(),
            arch: std::env::consts::ARCH.into(),
            hostname,
            mallops_version: env!("CARGO_PKG_VERSION").into(),
        }
    }
}

/// One lineage event of a run.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
#[non_exhaustive]
pub enum LineageEvent {
    /// The run started.
    #[non_exhaustive]
    RunStarted {
        /// Run id.
        run: RunId,
        /// When.
        timestamp: Timestamp,
        /// Hash of the validated pipeline.
        pipeline_hash: Digest,
        /// Git commit of the pipeline sources, when known.
        git_commit: Option<String>,
        /// Host executing the run.
        host: HostInfo,
    },
    /// A node started (after acquiring its job slot).
    #[non_exhaustive]
    BlockStarted {
        /// Run id.
        run: RunId,
        /// When.
        timestamp: Timestamp,
        /// Node id.
        node: String,
        /// Block used by the node.
        #[serde(serialize_with = "serialize_display")]
        block: BlockRef,
        /// `BlockSpec.version`.
        block_version: String,
        /// Hash of the node's `with:` config.
        config_hash: Digest,
        /// Node seed.
        seed: Option<u64>,
        /// Input artifact ids.
        inputs: Vec<ArtifactId>,
    },
    /// A node committed (or got from the cache) an output.
    #[non_exhaustive]
    ArtifactProduced {
        /// Run id.
        run: RunId,
        /// When.
        timestamp: Timestamp,
        /// Node id.
        node: String,
        /// Output artifact.
        artifact: ArtifactId,
    },
    /// A node reached its final status.
    #[non_exhaustive]
    BlockFinished {
        /// Run id.
        run: RunId,
        /// When.
        timestamp: Timestamp,
        /// Node id.
        node: String,
        /// Final status.
        status: Status,
        /// Time spent on the node (zero if it never started).
        #[serde(rename = "duration_ms", serialize_with = "serialize_duration_ms")]
        duration: Duration,
        /// `true` when served from the cache.
        cached: bool,
    },
    /// The run ended.
    #[non_exhaustive]
    RunFinished {
        /// Run id.
        run: RunId,
        /// When.
        timestamp: Timestamp,
        /// Overall status.
        status: Status,
    },
}

fn event_run(event: &LineageEvent) -> RunId {
    match event {
        LineageEvent::RunStarted { run, .. }
        | LineageEvent::BlockStarted { run, .. }
        | LineageEvent::ArtifactProduced { run, .. }
        | LineageEvent::BlockFinished { run, .. }
        | LineageEvent::RunFinished { run, .. } => *run,
    }
}

/// Destination of lineage events. A failing sink never fails the run.
pub trait LineageSink: Send + Sync {
    /// Records one event.
    ///
    /// # Errors
    ///
    /// When the event cannot be persisted; the runtime logs it and continues.
    fn record<'a>(&'a self, event: &'a LineageEvent) -> BoxFuture<'a, Result<(), Error>>;
}

#[derive(Serialize)]
struct LineageDocument<'a> {
    v: u32,
    run: RunId,
    events: &'a [serde_json::Value],
}

/// Core sink: writes `runs/<run-id>/lineage.json` through [`ArtifactStore::put_meta`].
pub struct JsonSink {
    store: Arc<dyn ArtifactStore>,
    runs: Mutex<HashMap<RunId, Vec<serde_json::Value>>>,
}

impl JsonSink {
    /// Creates a sink writing into `store`.
    #[must_use]
    pub fn new(store: Arc<dyn ArtifactStore>) -> Self {
        Self {
            store,
            runs: Mutex::new(HashMap::new()),
        }
    }
}

impl fmt::Debug for JsonSink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JsonSink").finish_non_exhaustive()
    }
}

impl LineageSink for JsonSink {
    fn record<'a>(&'a self, event: &'a LineageEvent) -> BoxFuture<'a, Result<(), Error>> {
        Box::pin(async move {
            let run = event_run(event);
            let is_finished = matches!(event, LineageEvent::RunFinished { .. });
            let path = RelPath::parse(&format!("runs/{run}/lineage.json"))?;
            let value = serde_json::to_value(event)
                .map_err(|e| Error::Runtime(format!("cannot serialize lineage event: {e}")))?;

            // ponytail: O(n^2) full rewrite on every event is accepted for local runs so
            // runs/<run-id>/lineage.json stays valid JSON even after a crash.
            let bytes = {
                let mut runs = self
                    .runs
                    .lock()
                    .map_err(|_| Error::Runtime("lineage sink mutex poisoned".into()))?;
                let serialized = if is_finished {
                    let mut events = runs.remove(&run).unwrap_or_default();
                    events.push(value);
                    serde_json::to_vec(&LineageDocument {
                        v: LINEAGE_DOC_VERSION,
                        run,
                        events: &events,
                    })
                } else {
                    let events = runs.entry(run).or_default();
                    events.push(value);
                    serde_json::to_vec(&LineageDocument {
                        v: LINEAGE_DOC_VERSION,
                        run,
                        events,
                    })
                };
                serialized.map_err(|e| {
                    Error::Runtime(format!("cannot serialize lineage document: {e}"))
                })?
            };

            self.store.put_meta(&path, &bytes).await
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LocalStore, MemoryStore};
    use serde_json::json;

    async fn read_lineage_doc(store: &dyn ArtifactStore, run: RunId) -> Option<serde_json::Value> {
        let path = RelPath::parse(&format!("runs/{run}/lineage.json")).unwrap();
        let bytes = store.get_meta(&path).await.unwrap()?;
        Some(serde_json::from_slice(&bytes).unwrap())
    }

    #[tokio::test]
    async fn lineage_json_sink_conformance_memory() {
        let store: Arc<dyn ArtifactStore> = Arc::new(MemoryStore::new());
        let sink = JsonSink::new(Arc::clone(&store));
        conformance::check_sink(&sink, |run| {
            let store = Arc::clone(&store);
            async move { read_lineage_doc(store.as_ref(), run).await }
        })
        .await;
    }

    #[tokio::test]
    async fn lineage_json_sink_conformance_local() {
        let dir = tempfile::tempdir().unwrap();
        let store: Arc<dyn ArtifactStore> = Arc::new(LocalStore::new(dir.path()).unwrap());
        let sink = JsonSink::new(Arc::clone(&store));
        conformance::check_sink(&sink, |run| {
            let store = Arc::clone(&store);
            async move { read_lineage_doc(store.as_ref(), run).await }
        })
        .await;
    }

    #[tokio::test]
    async fn lineage_conformance_accepts_write_only_sink() {
        let store: Arc<dyn ArtifactStore> = Arc::new(MemoryStore::new());
        let sink = JsonSink::new(store);
        conformance::check_sink(&sink, |_| async { None }).await;
    }

    #[test]
    fn lineage_timestamp_format_cases() {
        let ts = |secs: u64, nanos: u32| Timestamp::new(UNIX_EPOCH + Duration::new(secs, nanos));

        // UNIX epoch.
        assert_eq!(
            serde_json::to_value(Timestamp::new(UNIX_EPOCH)).unwrap(),
            json!("1970-01-01T00:00:00.000Z")
        );

        // Date with milliseconds (and sub-millisecond truncation).
        // 2026-11-23T10:00:00Z = 20_780 days * 86_400 + 10 * 3_600 = 1_795_428_000.
        assert_eq!(
            serde_json::to_value(ts(1_795_428_000, 123_999_999)).unwrap(),
            json!("2026-11-23T10:00:00.123Z")
        );

        // Leap day: 2024-02-29T23:59:59.999Z -> +1ms -> 2024-03-01T00:00:00.000Z.
        // 19_782 days * 86_400 + 86_399 = 1_709_251_199.
        assert_eq!(
            serde_json::to_value(ts(1_709_251_199, 999_000_000)).unwrap(),
            json!("2024-02-29T23:59:59.999Z")
        );
        assert_eq!(
            serde_json::to_value(ts(1_709_251_200, 0)).unwrap(),
            json!("2024-03-01T00:00:00.000Z")
        );

        // Century leap day (divisible by 400): 2000-02-29T12:34:56.007Z.
        // 11_016 days * 86_400 + 12 * 3_600 + 34 * 60 + 56 = 951_827_696.
        assert_eq!(
            serde_json::to_value(ts(951_827_696, 7_000_000)).unwrap(),
            json!("2000-02-29T12:34:56.007Z")
        );

        // Century non-leap transition (2100): 2100-02-28T23:59:59.000Z -> 2100-03-01T00:00:00.000Z.
        // Days 1970-01-01..2100-01-01 = 130 * 365 + 32 = 47_482; + 31 (Jan) + 27 (Feb 28) = 47_540.
        let feb_28_2100_end = 47_540 * 86_400 + 86_399;
        assert_eq!(
            serde_json::to_value(ts(feb_28_2100_end, 0)).unwrap(),
            json!("2100-02-28T23:59:59.000Z")
        );
        assert_eq!(
            serde_json::to_value(ts(feb_28_2100_end + 1, 0)).unwrap(),
            json!("2100-03-01T00:00:00.000Z")
        );

        // Pre-epoch instants.
        assert_eq!(
            serde_json::to_value(Timestamp::new(UNIX_EPOCH - Duration::from_millis(1))).unwrap(),
            json!("1969-12-31T23:59:59.999Z")
        );
        assert_eq!(
            serde_json::to_value(Timestamp::new(UNIX_EPOCH - Duration::from_secs(1))).unwrap(),
            json!("1969-12-31T23:59:59.000Z")
        );

        // Year > 9999 fails serialization instead of emitting invalid RFC 3339.
        let far_future = Timestamp::new(UNIX_EPOCH + Duration::from_secs(400_000_000_000));
        assert!(serde_json::to_value(far_future).is_err());
    }

    #[test]
    fn lineage_event_exact_serialization() {
        let run = RunId::new();
        let timestamp = Timestamp::new(UNIX_EPOCH + Duration::new(1_795_428_000, 123_000_000));
        let pipeline_hash = Digest::from_bytes([0x11; 32]);
        let config_hash = Digest::from_bytes([0x22; 32]);
        let in_id = ArtifactId::from_bytes([0xaa; 32]);
        let out_id = ArtifactId::from_bytes([0xbb; 32]);
        let block: BlockRef = "std/train.external".parse().unwrap();

        let started = LineageEvent::RunStarted {
            run,
            timestamp,
            pipeline_hash,
            git_commit: Some("abc1234".into()),
            host: HostInfo {
                os: "linux".into(),
                arch: "x86_64".into(),
                hostname: "worker-1".into(),
                mallops_version: "0.1.0".into(),
            },
        };
        assert_eq!(
            serde_json::to_value(&started).unwrap(),
            json!({
                "event": "run_started",
                "run": run.to_string(),
                "timestamp": "2026-11-23T10:00:00.123Z",
                "pipeline_hash": pipeline_hash.to_hex(),
                "git_commit": "abc1234",
                "host": {
                    "os": "linux",
                    "arch": "x86_64",
                    "hostname": "worker-1",
                    "mallops_version": "0.1.0"
                }
            })
        );

        let block_started = LineageEvent::BlockStarted {
            run,
            timestamp,
            node: "train".into(),
            block,
            block_version: "1.0.0".into(),
            config_hash,
            seed: Some(42),
            inputs: vec![in_id],
        };
        assert_eq!(
            serde_json::to_value(&block_started).unwrap(),
            json!({
                "event": "block_started",
                "run": run.to_string(),
                "timestamp": "2026-11-23T10:00:00.123Z",
                "node": "train",
                "block": "std/train.external",
                "block_version": "1.0.0",
                "config_hash": config_hash.to_hex(),
                "seed": 42,
                "inputs": [in_id.to_hex()]
            })
        );

        let produced = LineageEvent::ArtifactProduced {
            run,
            timestamp,
            node: "train".into(),
            artifact: out_id,
        };
        assert_eq!(
            serde_json::to_value(&produced).unwrap(),
            json!({
                "event": "artifact_produced",
                "run": run.to_string(),
                "timestamp": "2026-11-23T10:00:00.123Z",
                "node": "train",
                "artifact": out_id.to_hex()
            })
        );

        let block_finished = LineageEvent::BlockFinished {
            run,
            timestamp,
            node: "train".into(),
            status: Status::Succeeded,
            duration: Duration::from_millis(350),
            cached: true,
        };
        assert_eq!(
            serde_json::to_value(&block_finished).unwrap(),
            json!({
                "event": "block_finished",
                "run": run.to_string(),
                "timestamp": "2026-11-23T10:00:00.123Z",
                "node": "train",
                "status": "succeeded",
                "duration_ms": 350,
                "cached": true
            })
        );

        let run_finished = LineageEvent::RunFinished {
            run,
            timestamp,
            status: Status::Cancelled,
        };
        assert_eq!(
            serde_json::to_value(&run_finished).unwrap(),
            json!({
                "event": "run_finished",
                "run": run.to_string(),
                "timestamp": "2026-11-23T10:00:00.123Z",
                "status": "cancelled"
            })
        );
    }

    #[tokio::test]
    async fn lineage_json_sink_writes_incrementally_and_frees_buffer() {
        let store: Arc<dyn ArtifactStore> = Arc::new(MemoryStore::new());
        let sink = JsonSink::new(Arc::clone(&store));
        let run = RunId::new();
        let ts = Timestamp::new(UNIX_EPOCH + Duration::from_secs(1_795_428_000));

        sink.record(&LineageEvent::RunStarted {
            run,
            timestamp: ts,
            pipeline_hash: Digest::from_bytes([1; 32]),
            git_commit: None,
            host: HostInfo::current(),
        })
        .await
        .unwrap();

        let doc = read_lineage_doc(store.as_ref(), run).await.unwrap();
        assert_eq!(doc["v"], 1);
        assert_eq!(doc["run"], run.to_string());
        assert_eq!(doc["events"].as_array().unwrap().len(), 1);
        assert!(sink.runs.lock().unwrap().contains_key(&run));

        sink.record(&LineageEvent::RunFinished {
            run,
            timestamp: ts,
            status: Status::Succeeded,
        })
        .await
        .unwrap();

        let doc = read_lineage_doc(store.as_ref(), run).await.unwrap();
        assert_eq!(doc["events"].as_array().unwrap().len(), 2);
        assert!(!sink.runs.lock().unwrap().contains_key(&run));
    }

    #[tokio::test]
    async fn lineage_json_sink_poisoned_mutex_returns_runtime_error() {
        let store: Arc<dyn ArtifactStore> = Arc::new(MemoryStore::new());
        let sink = Arc::new(JsonSink::new(store));

        let poison_sink = Arc::clone(&sink);
        let _ = std::thread::spawn(move || {
            let _guard = poison_sink.runs.lock().unwrap();
            panic!("poison mutex");
        })
        .join();

        let err = sink
            .record(&LineageEvent::RunFinished {
                run: RunId::new(),
                timestamp: Timestamp::new(UNIX_EPOCH),
                status: Status::Failed,
            })
            .await
            .unwrap_err();
        assert!(
            matches!(err, Error::Runtime(_)),
            "expected Runtime, got {err:?}"
        );
    }

    #[test]
    fn lineage_host_info_current_populates_fields() {
        let host = HostInfo::current();
        assert_eq!(host.os, std::env::consts::OS);
        assert_eq!(host.arch, std::env::consts::ARCH);
        assert_eq!(host.mallops_version, env!("CARGO_PKG_VERSION"));
        assert!(!host.hostname.is_empty());
    }
}
