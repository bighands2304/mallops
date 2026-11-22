//! Block execution context, run identifiers, cooperative cancellation, and secret isolation.
//!
//! See `docs/SPEC.md` §2 and §4.1, `docs/ARCHITECTURE.md` §3, and
//! `docs/adr/0007-core-contracts-revision.md` (D4, D7).

pub(crate) mod secret;

use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tokio::sync::watch;

use crate::{ArtifactStore, Error};

pub use secret::{EnvSecretProvider, MemorySecretProvider, Secret, SecretProvider};

/// Unique identifier for a single pipeline run, backed by a ULID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RunId(ulid::Ulid);

impl RunId {
    /// Generates a new ULID-backed [`RunId`], sortable by time with millisecond granularity.
    pub fn new() -> Self {
        Self(ulid::Ulid::new())
    }
}

impl Default for RunId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for RunId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for RunId {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        ulid::Ulid::from_string(s)
            .map(Self)
            .map_err(|err| Error::Validation(format!("invalid run id `{s}`: {err}")))
    }
}

impl Serialize for RunId {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0.to_string())
    }
}

impl<'de> Deserialize<'de> for RunId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        ulid::Ulid::from_string(&raw)
            .map(Self)
            .map_err(serde::de::Error::custom)
    }
}

/// Producer half of cooperative cancellation for a pipeline run.
///
/// Built on [`tokio::sync::watch`] without requiring `tokio-util`.
#[derive(Debug)]
pub struct CancelSource {
    tx: watch::Sender<bool>,
}

impl CancelSource {
    /// Creates a new uncancelled [`CancelSource`].
    pub fn new() -> Self {
        let (tx, _rx) = watch::channel(false);
        Self { tx }
    }

    /// Creates a [`CancelToken`] observing this source.
    pub fn token(&self) -> CancelToken {
        CancelToken {
            rx: self.tx.subscribe(),
        }
    }

    /// Signals cancellation to all derived [`CancelToken`]s.
    ///
    /// Calling `cancel` multiple times is idempotent.
    pub fn cancel(&self) {
        // send_replace stores the value even when no token is alive (send would drop it).
        self.tx.send_replace(true);
    }
}

impl Default for CancelSource {
    fn default() -> Self {
        Self::new()
    }
}

/// Cooperative cancellation handle passed to blocks via [`Context`].
///
/// Cloning a `CancelToken` is cheap and shares the same cancellation signal.
#[derive(Debug, Clone)]
pub struct CancelToken {
    rx: watch::Receiver<bool>,
}

impl CancelToken {
    /// Returns `true` if cancellation has been signalled.
    pub fn is_cancelled(&self) -> bool {
        *self.rx.borrow()
    }

    /// Resolves once cancellation is signalled.
    ///
    /// Returns immediately if already cancelled. If the corresponding
    /// [`CancelSource`] is dropped without calling [`CancelSource::cancel`],
    /// this future remains pending so dropping a source never triggers a
    /// spurious cancellation.
    pub async fn cancelled(&self) {
        let mut rx = self.rx.clone();
        if rx.wait_for(|c| *c).await.is_err() {
            std::future::pending::<()>().await;
        }
    }
}

/// Execution context handed to each block during a pipeline run.
///
/// The runtime instruments block execution with a `tracing` span externally;
/// `Context` intentionally carries no `tracing::Span` field (ADR-0007 D4).
/// Blocks never call `store.put()` for their outputs — the runtime commits
/// returned output drafts and records lineage.
#[derive(Clone)]
#[non_exhaustive]
pub struct Context {
    run_id: RunId,
    workdir: PathBuf,
    store: Arc<dyn ArtifactStore>,
    config: serde_json::Value,
    cancel: CancelToken,
    secrets: Arc<dyn SecretProvider>,
}

impl Context {
    /// Creates a new block execution [`Context`].
    // ponytail: 6 argomenti posizionali, passare a un builder quando arriva il 7°
    pub fn new(
        run_id: RunId,
        workdir: PathBuf,
        store: Arc<dyn ArtifactStore>,
        config: serde_json::Value,
        cancel: CancelToken,
        secrets: Arc<dyn SecretProvider>,
    ) -> Self {
        Self {
            run_id,
            workdir,
            store,
            config,
            cancel,
            secrets,
        }
    }

    /// Returns the identifier of the current pipeline run.
    pub fn run_id(&self) -> RunId {
        self.run_id
    }

    /// Returns the scratch working directory allocated for this block invocation.
    pub fn workdir(&self) -> &Path {
        &self.workdir
    }

    /// Returns a handle to the run's [`ArtifactStore`].
    pub fn store(&self) -> &Arc<dyn ArtifactStore> {
        &self.store
    }

    /// Returns the block's resolved configuration (`with:`).
    ///
    /// This value never contains secrets; use [`secret`](Self::secret) to
    /// resolve `${secret:NAME}` references on demand.
    pub fn config(&self) -> &serde_json::Value {
        &self.config
    }

    /// Returns the cooperative cancellation token for this run.
    pub fn cancel(&self) -> &CancelToken {
        &self.cancel
    }

    /// Resolves a named secret through the context's [`SecretProvider`].
    pub fn secret(&self, name: &str) -> Result<Secret, Error> {
        self.secrets.get(name)
    }

    /// Deserializes the block's `config` into a strongly typed value `T`.
    ///
    /// Returns [`Error::Config`] if deserialization fails.
    pub fn config_as<T: DeserializeOwned>(&self) -> Result<T, Error> {
        T::deserialize(&self.config).map_err(|err| Error::Config(err.to_string()))
    }
}

impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Context")
            .field("run_id", &self.run_id)
            .field("workdir", &self.workdir)
            .field("store", &"<ArtifactStore>")
            .field("config", &self.config)
            .field("cancel", &self.cancel)
            .field("secrets", &"<SecretProvider>")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use serde::Deserialize;
    use serde_json::json;

    use super::*;
    use crate::MemoryStore;

    fn test_context(
        config: serde_json::Value,
        cancel: CancelToken,
        secrets: Arc<dyn SecretProvider>,
    ) -> Context {
        Context::new(
            RunId::new(),
            PathBuf::from("/tmp/mallops-test-workdir"),
            Arc::new(MemoryStore::new()),
            config,
            cancel,
            secrets,
        )
    }

    #[test]
    fn run_id_display_from_str_and_serde_roundtrip() {
        let id = RunId::new();
        let text = id.to_string();
        assert_eq!(text.len(), 26);

        let parsed: RunId = text.parse().expect("valid ULID string");
        assert_eq!(parsed, id);

        let encoded = serde_json::to_string(&id).expect("serialize RunId");
        let decoded: RunId = serde_json::from_str(&encoded).expect("deserialize RunId");
        assert_eq!(decoded, id);

        let err = "not-a-ulid".parse::<RunId>().expect_err("invalid RunId");
        assert!(matches!(err, Error::Validation(_)));
    }

    #[tokio::test]
    async fn cancel_token_signals_active_and_late_waiters() {
        let source = CancelSource::new();
        let token1 = source.token();
        let token2 = token1.clone();

        assert!(!token1.is_cancelled());
        assert!(!token2.is_cancelled());

        let waiter = tokio::spawn(async move {
            token1.cancelled().await;
            token1.is_cancelled()
        });

        tokio::task::yield_now().await;
        source.cancel();
        // Idempotent second call.
        source.cancel();

        assert!(token2.is_cancelled());
        assert!(waiter.await.expect("waiter task"));

        // Already-cancelled token resolves immediately.
        token2.cancelled().await;
    }

    #[test]
    fn cancel_before_token_is_observed() {
        let source = CancelSource::new();
        source.cancel();
        assert!(source.token().is_cancelled());
    }

    #[tokio::test]
    async fn cancel_token_remains_pending_when_source_dropped_without_cancel() {
        let source = CancelSource::new();
        let token = source.token();
        drop(source);

        assert!(!token.is_cancelled());
        let resolved = tokio::select! {
            biased;
            _ = token.cancelled() => true,
            _ = tokio::task::yield_now() => false,
        };
        assert!(!resolved);
    }

    #[test]
    fn context_config_as_deserializes_typed_config_and_reports_config_error() {
        #[derive(Debug, Deserialize, PartialEq)]
        struct BlockCfg {
            path: String,
            retries: u32,
        }

        let source = CancelSource::new();
        let ctx = test_context(
            json!({ "path": "data/customers.csv", "retries": 3 }),
            source.token(),
            Arc::new(MemorySecretProvider::default()),
        );

        let cfg: BlockCfg = ctx.config_as().expect("valid config");
        assert_eq!(
            cfg,
            BlockCfg {
                path: "data/customers.csv".to_string(),
                retries: 3,
            }
        );

        let bad_ctx = test_context(
            json!({ "path": "data/customers.csv", "retries": "not-a-number" }),
            source.token(),
            Arc::new(MemorySecretProvider::default()),
        );
        let err = bad_ctx
            .config_as::<BlockCfg>()
            .expect_err("type mismatch must fail");
        assert!(matches!(err, Error::Config(_)));
    }

    #[test]
    fn context_accessors_and_secret_resolution() {
        let run_id = RunId::new();
        let workdir = PathBuf::from("/tmp/workdir-accessor-test");
        let store: Arc<dyn ArtifactStore> = Arc::new(MemoryStore::new());
        let config = json!({ "key": "value" });
        let source = CancelSource::new();
        let secrets = Arc::new(MemorySecretProvider::default().with_secret("TOKEN", "top-secret"));

        let ctx = Context::new(
            run_id,
            workdir.clone(),
            Arc::clone(&store),
            config.clone(),
            source.token(),
            secrets,
        );

        assert_eq!(ctx.run_id(), run_id);
        assert_eq!(ctx.workdir(), workdir.as_path());
        assert_eq!(ctx.config(), &config);
        assert!(!ctx.cancel().is_cancelled());
        assert!(Arc::ptr_eq(ctx.store(), &store));

        let secret = ctx.secret("TOKEN").expect("secret should resolve");
        assert_eq!(secret.expose(), "top-secret");
        assert_eq!(format!("{secret}"), "***");

        let debug = format!("{ctx:?}");
        assert!(!debug.contains("top-secret"));
    }
}
