//! Run lineage: events emitted by the runtime and the sinks that persist them (ADR-0006).
//!
//! Blocks never see lineage: the runtime is the only producer.

pub mod conformance;

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Serialize, Serializer};
use tokio::sync::Mutex;

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

fn format_rfc3339_from_ms(total_ms: i128) -> Result<String, String> {
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
    format_rfc3339_from_ms(total_ms)
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
    events: &'a [LineageEvent],
}

/// Core sink: writes `runs/<run-id>/lineage.json` through [`ArtifactStore::put_meta`].
pub struct JsonSink {
    store: Arc<dyn ArtifactStore>,
    // ponytail: single sink-wide lock serializes writes across different runs too;
    // split into per-run locks if multi-run throughput ever requires it.
    runs: Mutex<HashMap<RunId, Vec<LineageEvent>>>,
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

            // ponytail: O(n^2) full rewrite on every event is accepted for local runs so
            // runs/<run-id>/lineage.json stays valid JSON even after a crash.
            let mut runs = self.runs.lock().await;
            let events = runs.entry(run).or_default();
            events.push(event.clone());
            let serialized = serde_json::to_vec(&LineageDocument {
                v: LINEAGE_DOC_VERSION,
                run,
                events,
            });
            let bytes = match serialized {
                Ok(bytes) => bytes,
                Err(e) => {
                    if is_finished {
                        runs.remove(&run);
                    } else {
                        events.pop();
                    }
                    return Err(Error::Runtime(format!(
                        "cannot serialize lineage document: {e}"
                    )));
                }
            };

            let result = self.store.put_meta(&path, &bytes).await;
            if is_finished {
                runs.remove(&run);
            }
            result
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::{Artifact, LocalStore, Manifest, MemoryStore, PayloadSource};
    use serde_json::json;

    async fn read_lineage_doc(store: &dyn ArtifactStore, run: RunId) -> Option<serde_json::Value> {
        let path = RelPath::parse(&format!("runs/{run}/lineage.json")).unwrap();
        let bytes = store.get_meta(&path).await.unwrap()?;
        Some(serde_json::from_slice(&bytes).unwrap())
    }

    async fn check_json_sink(store: Arc<dyn ArtifactStore>) {
        let sink = JsonSink::new(Arc::clone(&store));
        conformance::check_sink(&sink, |run| {
            let store = Arc::clone(&store);
            async move { read_lineage_doc(store.as_ref(), run).await }
        })
        .await;
    }

    #[tokio::test]
    async fn lineage_json_sink_conformance_memory() {
        check_json_sink(Arc::new(MemoryStore::new())).await;
    }

    #[tokio::test]
    async fn lineage_json_sink_conformance_local() {
        let dir = tempfile::tempdir().unwrap();
        check_json_sink(Arc::new(LocalStore::new(dir.path()).unwrap())).await;
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

        // Pre-epoch instants (including sub-millisecond flooring).
        assert_eq!(
            serde_json::to_value(Timestamp::new(UNIX_EPOCH - Duration::from_millis(1))).unwrap(),
            json!("1969-12-31T23:59:59.999Z")
        );
        assert_eq!(
            serde_json::to_value(Timestamp::new(UNIX_EPOCH - Duration::from_secs(1))).unwrap(),
            json!("1969-12-31T23:59:59.000Z")
        );
        // 100 ns is representable on Windows FILETIME too; Unix also checks the 1 ns edge.
        assert_eq!(
            serde_json::to_value(Timestamp::new(UNIX_EPOCH - Duration::from_nanos(100))).unwrap(),
            json!("1969-12-31T23:59:59.999Z")
        );
        assert_eq!(
            serde_json::to_value(Timestamp::new(UNIX_EPOCH - Duration::new(0, 1_000_100))).unwrap(),
            json!("1969-12-31T23:59:59.998Z")
        );
        #[cfg(not(windows))]
        {
            assert_eq!(
                serde_json::to_value(Timestamp::new(UNIX_EPOCH - Duration::from_nanos(1))).unwrap(),
                json!("1969-12-31T23:59:59.999Z")
            );
            assert_eq!(
                serde_json::to_value(Timestamp::new(UNIX_EPOCH - Duration::new(0, 1_000_001)))
                    .unwrap(),
                json!("1969-12-31T23:59:59.998Z")
            );
        }

        // Exact RFC 3339 lower bound: 0000-01-01T00:00:00.000Z = -719_528 days = -62_167_219_200 s.
        const YEAR_0000_SECS: u64 = 62_167_219_200;
        assert_eq!(
            format_rfc3339_from_ms(-i128::from(YEAR_0000_SECS) * 1_000).unwrap(),
            "0000-01-01T00:00:00.000Z"
        );
        assert!(format_rfc3339_from_ms(-i128::from(YEAR_0000_SECS) * 1_000 - 1).is_err());
        if let Some(y0) = UNIX_EPOCH.checked_sub(Duration::from_secs(YEAR_0000_SECS)) {
            assert_eq!(
                serde_json::to_value(Timestamp::new(y0)).unwrap(),
                json!("0000-01-01T00:00:00.000Z")
            );
        }
        if let Some(before_y0) = UNIX_EPOCH.checked_sub(Duration::new(YEAR_0000_SECS, 1_000_000)) {
            assert!(serde_json::to_value(Timestamp::new(before_y0)).is_err());
        }

        // Exact RFC 3339 upper bound: 9999-12-31T23:59:59.999Z, and year 10000 -> error.
        // Days 1970-01-01..10000-01-01 = 2_932_897 -> 253_402_300_800 s.
        const YEAR_10000_SECS: u64 = 253_402_300_800;
        assert_eq!(
            serde_json::to_value(ts(YEAR_10000_SECS - 1, 999_000_000)).unwrap(),
            json!("9999-12-31T23:59:59.999Z")
        );
        assert!(serde_json::to_value(ts(YEAR_10000_SECS, 0)).is_err());
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
        assert!(sink.runs.lock().await.contains_key(&run));

        sink.record(&LineageEvent::RunFinished {
            run,
            timestamp: ts,
            status: Status::Succeeded,
        })
        .await
        .unwrap();

        let doc = read_lineage_doc(store.as_ref(), run).await.unwrap();
        assert_eq!(doc["events"].as_array().unwrap().len(), 2);
        assert!(!sink.runs.lock().await.contains_key(&run));
    }

    /// Store wrapper that delays the first `put_meta` and records every written snapshot length.
    struct SlowFirstPutStore {
        inner: Arc<dyn ArtifactStore>,
        calls: AtomicUsize,
        written_lens: std::sync::Mutex<Vec<usize>>,
    }

    impl ArtifactStore for SlowFirstPutStore {
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
            self.inner.exists(id)
        }

        fn put_meta<'a>(
            &'a self,
            key: &'a RelPath,
            data: &'a [u8],
        ) -> BoxFuture<'a, Result<(), Error>> {
            Box::pin(async move {
                if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    // Delay the first write so concurrent calls would race ahead and be
                    // overwritten if `JsonSink` released its lock before `put_meta`.
                    tokio::time::sleep(Duration::from_millis(30)).await;
                }
                self.inner.put_meta(key, data).await?;
                let doc: serde_json::Value = serde_json::from_slice(data).unwrap();
                let len = doc["events"].as_array().unwrap().len();
                self.written_lens.lock().unwrap().push(len);
                Ok(())
            })
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn lineage_json_sink_concurrent_records_never_regress() {
        const N: usize = 16;
        let dir = tempfile::tempdir().unwrap();
        let inner: Arc<dyn ArtifactStore> = Arc::new(LocalStore::new(dir.path()).unwrap());
        let store = Arc::new(SlowFirstPutStore {
            inner,
            calls: AtomicUsize::new(0),
            written_lens: std::sync::Mutex::new(Vec::new()),
        });
        let sink = Arc::new(JsonSink::new(Arc::clone(&store) as Arc<dyn ArtifactStore>));
        let run = RunId::new();
        let ts = Timestamp::new(UNIX_EPOCH + Duration::from_secs(1_795_428_000));
        let barrier = Arc::new(tokio::sync::Barrier::new(N));

        let mut tasks = tokio::task::JoinSet::new();
        for i in 0..N {
            let sink = Arc::clone(&sink);
            let barrier = Arc::clone(&barrier);
            tasks.spawn(async move {
                barrier.wait().await;
                let event = LineageEvent::ArtifactProduced {
                    run,
                    timestamp: ts,
                    node: format!("n{i:02}"),
                    artifact: ArtifactId::from_bytes([u8::try_from(i).unwrap(); 32]),
                };
                sink.record(&event).await.unwrap();
            });
        }

        tokio::time::timeout(Duration::from_secs(10), async {
            while let Some(res) = tasks.join_next().await {
                res.unwrap();
            }
        })
        .await
        .expect("concurrent record tasks timed out");

        let lens = store.written_lens.lock().unwrap().clone();
        assert_eq!(lens.len(), N, "expected {N} put_meta writes, got {lens:?}");
        for pair in lens.windows(2) {
            assert!(
                pair[1] >= pair[0],
                "lineage.json snapshot regressed: {lens:?}"
            );
        }

        let doc = read_lineage_doc(store.as_ref(), run).await.unwrap();
        let events = doc["events"].as_array().unwrap();
        assert_eq!(events.len(), N);
        let mut nodes: Vec<String> = events
            .iter()
            .map(|e| e["node"].as_str().unwrap().to_owned())
            .collect();
        nodes.sort();
        let expected: Vec<String> = (0..N).map(|i| format!("n{i:02}")).collect();
        assert_eq!(nodes, expected);

        sink.record(&LineageEvent::RunFinished {
            run,
            timestamp: ts,
            status: Status::Succeeded,
        })
        .await
        .unwrap();
        let final_doc = read_lineage_doc(store.as_ref(), run).await.unwrap();
        assert_eq!(final_doc["events"].as_array().unwrap().len(), N + 1);
        assert!(!sink.runs.lock().await.contains_key(&run));
    }
}
