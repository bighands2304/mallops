//! HTTP/JSON inference server for a `package` artifact (`mallops serve`, outside the DAG,
//! ADR-0007 D5).
//!
//! [`Server::load`] materializes the package in a temporary directory owned by the server, picks
//! the predictor by flavor through [`PredictorRegistry::get`] (code-executing flavors need
//! [`ServeOptions::trust_unsafe_flavors`], ADR-0007 D7), loads the optional fitted transform and
//! the model. [`Server::run`] serves until the shutdown future resolves.
//!
//! Endpoints:
//! - `GET /health` → `200 {"status":"ok","package":ID,"flavor":F}`; no auth, timeout or
//!   concurrency limit, so a liveness probe never gets `503` from a busy server.
//! - `POST /predict`: a JSON object is one row and returns one object; an array of objects is a
//!   batch and returns an array in the same order. With a transform the rows go through
//!   [`FittedTransform::apply_json`] and the model's `io_schema.input` columns are sent to the
//!   predictor. Errors are `{"error": msg}`: `400` bad JSON or transform input, `401` missing or
//!   wrong `Authorization: Bearer` token, `413` body over [`ServeOptions::max_body_bytes`], `503`
//!   concurrency limit reached, `504` timeout, `500` predictor failure (generic message).
//!
//! Request logs (`tracing` INFO) carry only method, path, status, latency and byte sizes, never
//! bodies; the predictor error of a `500` is logged at WARN and may quote the worker's stderr.
//!
//! Hot swap: [`ServeHandle::swap`] builds the new state completely, then swaps it atomically;
//! requests in flight finish on the old state, which is freed with its last reference.

use std::future::Future;
use std::sync::{Arc, PoisonError, RwLock};
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::{Bytes, HttpBody};
use axum::extract::rejection::BytesRejection;
use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::middleware::{Next, from_fn_with_state};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
#[cfg(feature = "preprocess-tabular")]
use mallops_core::Manifest;
use mallops_core::{ArtifactId, ArtifactStore, Error, Secret};
use serde_json::{Value, json};
use tokio::sync::{Mutex, Semaphore};

#[cfg(feature = "preprocess-tabular")]
use crate::FittedTransform;
use crate::Package;
use crate::predictor::{LoadedPredictor, PredictInput, PredictorRegistry};

/// Options of [`Server::load`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ServeOptions {
    /// Address to bind; [`Server`] does not use it (the caller binds the listener).
    pub host: String,
    /// Port to bind; [`Server`] does not use it (the caller binds the listener).
    pub port: u16,
    /// Largest accepted `/predict` body; larger ones get `413`.
    pub max_body_bytes: usize,
    /// Per-request timeout of `/predict` (`504`).
    pub timeout: Duration,
    /// Concurrent `/predict` requests; more get `503`.
    pub max_concurrency: usize,
    /// Allow flavors that execute code from the artifact (ADR-0007 D7).
    pub trust_unsafe_flavors: bool,
    /// Bearer token required by `/predict`; `None` disables auth.
    pub auth_token: Option<Secret>,
}

impl Default for ServeOptions {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 8080,
            max_body_bytes: 10 * 1024 * 1024,
            timeout: Duration::from_secs(30),
            max_concurrency: 64,
            trust_unsafe_flavors: false,
            auth_token: None,
        }
    }
}

/// One loaded package. Fields drop in declaration order: the predictor (an external worker runs
/// in the package dir) goes before the directory.
struct LoadedState {
    id: ArtifactId,
    flavor: String,
    io_schema: Option<Value>,
    #[cfg(feature = "preprocess-tabular")]
    model: Manifest,
    #[cfg(feature = "preprocess-tabular")]
    transform: Option<FittedTransform>,
    loaded: Box<dyn LoadedPredictor>,
    _dir: tempfile::TempDir,
}

impl LoadedState {
    async fn load(
        store: &dyn ArtifactStore,
        id: &ArtifactId,
        predictors: &PredictorRegistry,
        trust: bool,
    ) -> Result<Self, Error> {
        let dir = tempfile::tempdir()
            .map_err(|e| Error::Runtime(format!("cannot create a temporary directory: {e}")))?;
        let pkg = Package::load(store, id, &dir.path().join("package")).await?;
        let predictor = predictors.get(pkg.flavor(), trust)?;
        #[cfg(feature = "preprocess-tabular")]
        let transform = pkg
            .transform()
            .map(|(_, d)| FittedTransform::from_dir(&d))
            .transpose()?;
        #[cfg(not(feature = "preprocess-tabular"))]
        if pkg.transform().is_some() {
            return Err(Error::Config(format!(
                "package {id} has a transform: build mallops-std with the `preprocess-tabular` feature"
            )));
        }
        let loaded = predictor.load(pkg.model(), &pkg.model_dir()).await?;
        Ok(Self {
            id: *id,
            flavor: pkg.flavor().to_owned(),
            io_schema: pkg.io_schema().cloned(),
            #[cfg(feature = "preprocess-tabular")]
            model: pkg.model().manifest().clone(),
            #[cfg(feature = "preprocess-tabular")]
            transform,
            loaded,
            _dir: dir,
        })
    }

    /// The predictor inputs for `rows` (JSON objects): through the transform when there is one.
    fn inputs(&self, rows: Vec<Value>) -> Result<Vec<PredictInput>, Fail> {
        #[cfg(feature = "preprocess-tabular")]
        let rows = match &self.transform {
            None => rows,
            Some(t) => {
                let batch = t
                    .apply_json(&Value::Array(rows))
                    .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
                crate::rows::feature_names(&self.model, &batch, t.target())
                    .and_then(|names| crate::rows::rows(&batch, &names))
                    .map_err(internal)?
            }
        };
        Ok(rows.into_iter().map(PredictInput::Json).collect())
    }
}

/// State shared by the server and its [`ServeHandle`]s.
struct Shared {
    current: RwLock<Arc<LoadedState>>,
    /// Serializes swaps (and the id comparison).
    swapping: Mutex<()>,
    trust: bool,
}

/// Swaps the served package while the server runs (cheap to clone).
#[derive(Clone)]
pub struct ServeHandle(Arc<Shared>);

impl ServeHandle {
    fn current(&self) -> Arc<LoadedState> {
        // Clone and release: the lock is never held across an await.
        Arc::clone(
            &self
                .0
                .current
                .read()
                .unwrap_or_else(PoisonError::into_inner),
        )
    }

    /// Loads package `id` with the server's trust policy and swaps it in atomically; requests in
    /// flight finish on the old package. Returns `false` (and does nothing) if `id` is already
    /// served. Swaps are serialized.
    ///
    /// # Errors
    ///
    /// Any [`Error`] of [`Server::load`]; the current package stays in service and the error is
    /// also logged at WARN.
    pub async fn swap(
        &self,
        store: &dyn ArtifactStore,
        id: &ArtifactId,
        predictors: &PredictorRegistry,
    ) -> Result<bool, Error> {
        let _serial = self.0.swapping.lock().await;
        let old = self.current();
        if old.id == *id {
            return Ok(false);
        }
        let new = match LoadedState::load(store, id, predictors, self.0.trust).await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(package = %id, error = %e, "swap failed, still serving {}", old.id);
                return Err(e);
            }
        };
        if new.io_schema != old.io_schema {
            tracing::warn!(old = %old.id, new = %id, "the new package has a different io_schema");
        }
        *self
            .0
            .current
            .write()
            .unwrap_or_else(PoisonError::into_inner) = Arc::new(new);
        tracing::info!(old = %old.id, new = %id, "package swapped");
        // ponytail: the last Arc removes the old package dir synchronously on this thread
        // (large models = a short stall); move it to spawn_blocking if that matters.
        Ok(true)
    }
}

/// Limits of `/predict`, outside the swapped state.
struct Guard {
    timeout: Duration,
    permits: Semaphore,
    /// BLAKE3 of the token: `Hash` equality is constant-time.
    token: Option<blake3::Hash>,
}

/// An inference server for one package (swappable through [`Server::handle`]).
pub struct Server {
    handle: ServeHandle,
    guard: Arc<Guard>,
    max_body_bytes: usize,
}

impl Server {
    /// Loads package `id` from `store` into a temporary directory owned by the server, gates its
    /// flavor, and loads its transform and model.
    ///
    /// # Errors
    ///
    /// [`Error::Config`] if `id` is not a package, the flavor has no predictor or needs trust,
    /// the auth token is empty, `max_concurrency` is 0, or the package has a transform without
    /// the `preprocess-tabular` feature; any other [`Error`] of the load.
    pub async fn load(
        store: &dyn ArtifactStore,
        id: &ArtifactId,
        predictors: &PredictorRegistry,
        opts: ServeOptions,
    ) -> Result<Self, Error> {
        let token = match &opts.auth_token {
            Some(t) if t.expose().trim().is_empty() => {
                return Err(Error::Config("the auth token is empty".into()));
            }
            Some(t) => Some(blake3::hash(t.expose().as_bytes())),
            None => None,
        };
        if opts.max_concurrency == 0 {
            return Err(Error::Config("max_concurrency must be at least 1".into()));
        }
        let state = LoadedState::load(store, id, predictors, opts.trust_unsafe_flavors).await?;
        Ok(Self {
            handle: ServeHandle(Arc::new(Shared {
                current: RwLock::new(Arc::new(state)),
                swapping: Mutex::new(()),
                trust: opts.trust_unsafe_flavors,
            })),
            guard: Arc::new(Guard {
                timeout: opts.timeout,
                permits: Semaphore::new(opts.max_concurrency),
                token,
            }),
            max_body_bytes: opts.max_body_bytes,
        })
    }

    /// A handle to swap the served package.
    #[must_use]
    pub fn handle(&self) -> ServeHandle {
        self.handle.clone()
    }

    /// Serves on `listener` until `shutdown` resolves, then finishes the requests in flight.
    /// The package (predictor and directory) is freed when the last [`ServeHandle`] drops.
    ///
    /// # Errors
    ///
    /// [`Error::Runtime`] if the listener fails.
    pub async fn run(
        self,
        listener: tokio::net::TcpListener,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> Result<(), Error> {
        let predict = post(predict)
            .layer(DefaultBodyLimit::max(self.max_body_bytes))
            .layer(from_fn_with_state(self.guard, guard));
        let app = Router::new()
            .route("/health", get(health))
            .route("/predict", predict)
            .with_state(self.handle);
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown)
            .await
            .map_err(|e| Error::Runtime(format!("serve: {e}")))
    }
}

fn error(status: StatusCode, msg: &str) -> Response {
    (status, axum::Json(json!({ "error": msg }))).into_response()
}

/// A failed `/predict`: status and client message.
type Fail = (StatusCode, String);

/// `500` with a generic message; the cause is logged at WARN.
fn internal(e: Error) -> Fail {
    tracing::warn!(error = %e, "prediction failed");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        "prediction failed".into(),
    )
}

/// Request body size, passed from the handler to the log in [`guard`].
#[derive(Clone, Copy)]
struct ReqBytes(usize);

/// `/predict` middleware: log → timeout → auth → concurrency permit.
async fn guard(State(g): State<Arc<Guard>>, req: Request, next: Next) -> Response {
    let start = Instant::now();
    let (method, path) = (req.method().clone(), req.uri().path().to_owned());
    // ponytail: the timeout is not preemptive for predictors that compute synchronously inside
    // their future (smartcore); spawn_blocking with the permit moved in if CPU-heavy models need it.
    let resp = match tokio::time::timeout(g.timeout, admit(&g, req, next)).await {
        Ok(r) => r,
        Err(_) => error(StatusCode::GATEWAY_TIMEOUT, "request timed out"),
    };
    tracing::info!(
        %method,
        path,
        status = resp.status().as_u16(),
        latency_ms = start.elapsed().as_millis() as u64,
        req_bytes = resp.extensions().get::<ReqBytes>().map_or(0, |b| b.0),
        resp_bytes = resp.body().size_hint().exact().unwrap_or(0),
        "request"
    );
    resp
}

async fn admit(g: &Guard, req: Request, next: Next) -> Response {
    if let Some(want) = &g.token
        && !authorized(req.headers(), want)
    {
        let mut r = error(StatusCode::UNAUTHORIZED, "missing or invalid bearer token");
        r.headers_mut()
            .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        return r;
    }
    let Ok(_permit) = g.permits.try_acquire() else {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "too many concurrent requests",
        );
    };
    next.run(req).await
}

fn authorized(headers: &HeaderMap, want: &blake3::Hash) -> bool {
    headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .is_some_and(|t| blake3::hash(t.as_bytes()) == *want)
}

async fn health(State(h): State<ServeHandle>) -> Response {
    let s = h.current();
    axum::Json(json!({ "status": "ok", "package": s.id, "flavor": s.flavor })).into_response()
}

async fn predict(State(h): State<ServeHandle>, body: Result<Bytes, BytesRejection>) -> Response {
    let body = match body {
        Ok(b) => b,
        Err(r) => return error(r.status(), &r.body_text()),
    };
    let mut resp = match answer(&h, &body).await {
        Ok(v) => axum::Json(v).into_response(),
        Err((status, msg)) => error(status, &msg),
    };
    resp.extensions_mut().insert(ReqBytes(body.len()));
    resp
}

async fn answer(h: &ServeHandle, body: &[u8]) -> Result<Value, Fail> {
    let value: Value = serde_json::from_slice(body)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid JSON: {e}")))?;
    let (rows, single) = match value {
        Value::Object(_) => (vec![value], true),
        Value::Array(items) if items.iter().all(Value::is_object) => (items, false),
        _ => {
            let msg = "expected a JSON object or an array of objects";
            return Err((StatusCode::BAD_REQUEST, msg.into()));
        }
    };
    if rows.is_empty() {
        return Ok(json!([]));
    }
    let state = h.current();
    let inputs = state.inputs(rows)?;
    let n = inputs.len();
    let outputs = state.loaded.predict_batch(inputs).await.map_err(internal)?;
    if outputs.len() != n {
        return Err(internal(Error::Runtime(format!(
            "predictor returned {} outputs for {n} rows",
            outputs.len()
        ))));
    }
    let mut values = Vec::with_capacity(n);
    for o in outputs {
        let PredictInput::Json(v) = o else {
            return Err(internal(Error::Runtime("non-JSON prediction".into())));
        };
        values.push(v);
    }
    Ok(if single {
        values.swap_remove(0)
    } else {
        Value::Array(values)
    })
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicBool, Ordering};

    use mallops_core::{
        Artifact, ArtifactKind, Block, BlockOutcome, BoxFuture, CancelSource, Context, Inputs,
        Manifest, MemorySecretProvider, MemoryStore, PayloadSource, RunId,
    };
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};
    use tokio::sync::oneshot;
    use tokio::task::JoinHandle;

    use super::*;
    use crate::package::PackageBlock;
    use crate::predictor::Predictor;

    const MOCK: &str = "mock";
    const UNSAFE: &str = "mock-unsafe";

    struct Mock(&'static [&'static str], bool);

    impl Predictor for Mock {
        fn flavors(&self) -> &[&str] {
            self.0
        }
        fn executes_code(&self) -> bool {
            self.1
        }
        fn load<'a>(
            &'a self,
            _: &'a Artifact,
            dir: &'a Path,
        ) -> BoxFuture<'a, Result<Box<dyn LoadedPredictor>, Error>> {
            let file = dir.join("data");
            Box::pin(async move { Ok(Box::new(MockLoaded(file)) as Box<dyn LoadedPredictor>) })
        }
    }

    /// Answers `{"answer": <model payload>}`, read from disk at predict time (after sleeping the
    /// row's `sleep_ms`), so a removed package dir fails the request; `fail` makes it error.
    struct MockLoaded(PathBuf);

    impl LoadedPredictor for MockLoaded {
        fn predict(&self, input: PredictInput) -> BoxFuture<'_, Result<PredictInput, Error>> {
            Box::pin(async move {
                let PredictInput::Json(row) = input else {
                    return Err(Error::Runtime("json only".into()));
                };
                if let Some(ms) = row.get("sleep_ms").and_then(Value::as_u64) {
                    tokio::time::sleep(Duration::from_millis(ms)).await;
                }
                if row.get("fail").is_some() {
                    return Err(Error::Runtime("mock failure".into()));
                }
                let answer = std::fs::read_to_string(&self.0)
                    .map_err(|e| Error::Runtime(format!("{}: {e}", self.0.display())))?;
                Ok(PredictInput::Json(json!({ "answer": answer })))
            })
        }
        fn predict_batch(
            &self,
            inputs: Vec<PredictInput>,
        ) -> BoxFuture<'_, Result<Vec<PredictInput>, Error>> {
            Box::pin(async move {
                let mut out = Vec::new();
                for i in inputs {
                    out.push(self.predict(i).await?);
                }
                Ok(out)
            })
        }
    }

    fn predictors() -> PredictorRegistry {
        let mut r = PredictorRegistry::new();
        r.register(Arc::new(Mock(&[MOCK], false))).unwrap();
        r.register(Arc::new(Mock(&[UNSAFE], true))).unwrap();
        r
    }

    fn new_store() -> Arc<dyn ArtifactStore> {
        Arc::new(MemoryStore::new())
    }

    async fn put(store: &Arc<dyn ArtifactStore>, m: Manifest, src: PayloadSource<'_>) -> Artifact {
        let id = store.put(m.clone(), src).await.unwrap();
        Artifact::new(id, m)
    }

    /// Runs `std/package` and commits its output like the runtime does.
    async fn package(
        store: &Arc<dyn ArtifactStore>,
        model: &Artifact,
        transform: Option<&Artifact>,
    ) -> ArtifactId {
        let wd = tempfile::tempdir().unwrap();
        let mut inputs = Inputs::new().with("model", model.clone());
        if let Some(t) = transform {
            inputs = inputs.with("preprocess", t.clone());
        }
        let ctx = Context::new(
            RunId::new(),
            wd.path().to_path_buf(),
            Arc::clone(store),
            Value::Null,
            CancelSource::new().token(),
            Arc::new(MemorySecretProvider::default()),
        );
        let BlockOutcome::Produced(d) = PackageBlock::new().run(&ctx, inputs).await.unwrap() else {
            panic!("skipped");
        };
        let draft = d.get("package").unwrap();
        store
            .put(draft.manifest.clone(), draft.payload.as_source())
            .await
            .unwrap()
    }

    /// A package of a mock model whose payload is `answer`.
    async fn mock_package(
        store: &Arc<dyn ArtifactStore>,
        flavor: &str,
        answer: &str,
    ) -> ArtifactId {
        let m = Manifest::new(ArtifactKind::Model, "txt").with_flavor(flavor);
        let model = put(store, m, PayloadSource::Bytes(answer.as_bytes())).await;
        package(store, &model, None).await
    }

    struct Running {
        addr: SocketAddr,
        handle: ServeHandle,
        stop: oneshot::Sender<()>,
        task: JoinHandle<Result<(), Error>>,
    }

    /// A global INFO subscriber that discards everything, installed before any server runs: a
    /// callsite first hit by a thread without a subscriber could otherwise cache `Interest::never`
    /// and hide events from the thread-local subscriber of `serve_http_hardening`.
    fn global_sink() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            let sink = tracing_subscriber::fmt()
                .with_writer(std::io::sink)
                .with_max_level(tracing::Level::INFO)
                .finish();
            tracing::subscriber::set_global_default(sink).unwrap();
        });
    }

    async fn start(server: Server) -> Running {
        global_sink();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = server.handle();
        let (stop, rx) = oneshot::channel();
        let task = tokio::spawn(server.run(listener, async {
            let _ = rx.await;
        }));
        Running {
            addr,
            handle,
            stop,
            task,
        }
    }

    async fn serve(store: &Arc<dyn ArtifactStore>, id: &ArtifactId, opts: ServeOptions) -> Running {
        start(
            Server::load(store.as_ref(), id, &predictors(), opts)
                .await
                .unwrap(),
        )
        .await
    }

    /// Minimal HTTP/1.1 client: returns status, raw head and body.
    async fn call(
        addr: SocketAddr,
        method: &str,
        path: &str,
        headers: &[(&str, &str)],
        body: &str,
    ) -> (u16, String, String) {
        let mut s = TcpStream::connect(addr).await.unwrap();
        let mut req = format!(
            "{method} {path} HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\n",
            body.len()
        );
        for (k, v) in headers {
            req.push_str(&format!("{k}: {v}\r\n"));
        }
        req.push_str("\r\n");
        req.push_str(body);
        s.write_all(req.as_bytes()).await.unwrap();
        let mut buf = Vec::new();
        // A server that rejects early may reset after replying: keep what was read.
        let _ = s.read_to_end(&mut buf).await;
        let text = String::from_utf8(buf).unwrap();
        let (head, body) = text.split_once("\r\n\r\n").unwrap();
        (
            head[9..12].parse().unwrap(),
            head.to_owned(),
            body.to_owned(),
        )
    }

    async fn post(addr: SocketAddr, headers: &[(&str, &str)], body: &str) -> (u16, Value) {
        let (status, _, body) = call(addr, "POST", "/predict", headers, body).await;
        (status, serde_json::from_str(&body).unwrap())
    }

    async fn health(addr: SocketAddr) -> Value {
        let (status, _, body) = call(addr, "GET", "/health", &[], "").await;
        assert_eq!(status, 200);
        serde_json::from_str(&body).unwrap()
    }

    async fn answer(addr: SocketAddr, body: &str) -> String {
        let (status, v) = post(addr, &[], body).await;
        assert_eq!(status, 200, "{v}");
        v["answer"].as_str().unwrap().to_owned()
    }

    fn pkg_dir(h: &ServeHandle) -> PathBuf {
        h.current()._dir.path().to_path_buf()
    }

    #[cfg(all(feature = "predict-smartcore", feature = "preprocess-tabular"))]
    #[tokio::test]
    async fn serve_http_predict_and_transform() {
        use mallops_core::RelPath;

        use crate::ingest::{Column, ColumnType};
        use crate::predictor::SmartcorePredictor;
        use crate::preprocess::transform::{Affine, FittedStep};

        let store = new_store();
        let col = |name: &str, ty| Column {
            name: name.into(),
            ty,
        };
        let t = FittedTransform::new(
            "y".into(),
            vec![
                col("x1", ColumnType::Float64),
                col("flag", ColumnType::Bool),
                col("y", ColumnType::Int64),
            ],
            vec![FittedStep::Scale {
                params: [(
                    "x1".to_owned(),
                    Affine {
                        offset: 1.0,
                        scale: 2.0,
                    },
                )]
                .into(),
            }],
        );
        let t_bytes = t.to_json().unwrap();
        let files = |name: &str, bytes| [(RelPath::parse(name).unwrap(), bytes)].into();
        let transform = put(
            &store,
            Manifest::new(ArtifactKind::Transform, "json"),
            PayloadSource::Files(files(crate::TRANSFORM_FILE, t_bytes.as_slice())),
        )
        .await;
        let model_json = serde_json::to_vec(&json!({
            "v": 1, "algorithm": "logistic_regression", "features": ["x1", "flag"],
            "target": "y", "intercept": -0.5, "coefficients": [1.0, 0.5],
            "classes": [0, 1], "threshold": 0.5
        }))
        .unwrap();
        let model_manifest = Manifest::new(ArtifactKind::Model, "json")
            .with_flavor("smartcore-logistic")
            .with_io_schema(json!({
                "input": [{"name": "x1", "type": "float64"}, {"name": "flag", "type": "bool"}],
                "output": [{"name": "y", "type": "int64"}],
            }));
        let model = put(
            &store,
            model_manifest.clone(),
            PayloadSource::Files(files("model.json", model_json.as_slice())),
        )
        .await;
        let id = package(&store, &model, Some(&transform)).await;

        let mut predictors = PredictorRegistry::new();
        crate::predictor::register_std_predictors(&mut predictors).unwrap();
        let server = Server::load(store.as_ref(), &id, &predictors, ServeOptions::default())
            .await
            .unwrap();
        let s = start(server).await;

        let h = health(s.addr).await;
        assert_eq!(
            h,
            json!({"status": "ok", "package": id, "flavor": "smartcore-logistic"})
        );

        // offline: transform + predict
        let rows = json!([
            {"x1": 3.0, "flag": true},
            {"x1": -2.0, "flag": false},
            {"x1": 0.5, "flag": true},
        ]);
        let batch = t.apply_json(&rows).unwrap();
        let names = crate::rows::feature_names(&model_manifest, &batch, "y").unwrap();
        let inputs = crate::rows::rows(&batch, &names).unwrap();
        let mdir = tempfile::tempdir().unwrap();
        std::fs::write(mdir.path().join("model.json"), &model_json).unwrap();
        let loaded = SmartcorePredictor::new()
            .load(&model, mdir.path())
            .await
            .unwrap();
        let offline: Vec<Value> = loaded
            .predict_batch(inputs.into_iter().map(PredictInput::Json).collect())
            .await
            .unwrap()
            .into_iter()
            .map(|o| match o {
                PredictInput::Json(v) => v,
                other => panic!("{other:?}"),
            })
            .collect();
        // the transform matters: unscaled x1 = 3.0 would give a different probability
        assert_ne!(
            offline[0]["probability"],
            json!(1.0 / (1.0 + (-4.0f64).exp()))
        );

        // JSON text round trip of f64 may differ in the last ulp (no `float_roundtrip`)
        let same = |got: &Value, want: &Value| {
            assert_eq!(got["prediction"], want["prediction"]);
            let (g, w) = (got["probability"].as_f64(), want["probability"].as_f64());
            assert!((g.unwrap() - w.unwrap()).abs() < 1e-12, "{got} vs {want}");
        };
        let (status, batch_out) = post(s.addr, &[], &rows.to_string()).await;
        assert_eq!(status, 200);
        let batch_out = batch_out.as_array().unwrap();
        assert_eq!(batch_out.len(), offline.len());
        batch_out.iter().zip(&offline).for_each(|(g, w)| same(g, w));
        let (status, single) = post(s.addr, &[], &rows[0].to_string()).await;
        assert_eq!(status, 200);
        same(&single, &offline[0]);
        let (status, empty) = post(s.addr, &[], "[]").await;
        assert_eq!((status, empty), (200, json!([])));
        // wrong input type for the transform
        let (status, e) = post(s.addr, &[], r#"{"x1": "a", "flag": true}"#).await;
        assert_eq!(status, 400, "{e}");
        assert!(e["error"].is_string());

        s.stop.send(()).unwrap();
        s.task.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn serve_http_unsafe_flavor_gate() {
        let store = new_store();
        let id = mock_package(&store, UNSAFE, "U").await;
        let opts = ServeOptions::default();
        let err = Server::load(store.as_ref(), &id, &predictors(), opts.clone())
            .await
            .err()
            .unwrap();
        let Error::Config(msg) = err else {
            panic!("{err:?}")
        };
        assert!(msg.contains("--trust-unsafe-flavors"), "{msg}");

        let mut trusted = opts;
        trusted.trust_unsafe_flavors = true;
        let s = serve(&store, &id, trusted).await;
        assert_eq!(answer(s.addr, "{}").await, "U");
    }

    #[derive(Clone, Default)]
    struct Logs(Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for Logs {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    // current_thread runtime: the thread-local subscriber also sees the server's tasks.
    #[tokio::test]
    async fn serve_http_hardening() {
        global_sink();
        let logs = Logs::default();
        let w = logs.clone();
        let sub = tracing_subscriber::fmt()
            .with_writer(move || w.clone())
            .with_ansi(false) // `ansi` may be unified in by mallops-cli
            .with_max_level(tracing::Level::INFO)
            .finish();
        let _guard = tracing::subscriber::set_default(sub);

        let d = ServeOptions::default();
        assert_eq!((d.host.as_str(), d.port), ("127.0.0.1", 8080));
        assert_eq!(d.max_body_bytes, 10 * 1024 * 1024);
        assert!(d.auth_token.is_none() && !d.trust_unsafe_flavors);

        let store = new_store();
        let id = mock_package(&store, MOCK, "A").await;
        let mut opts = ServeOptions {
            max_body_bytes: 256,
            timeout: Duration::from_millis(300),
            max_concurrency: 1,
            auth_token: Some(Secret::new(" ")),
            ..ServeOptions::default()
        };
        let err = Server::load(store.as_ref(), &id, &predictors(), opts.clone()).await;
        assert!(matches!(err, Err(Error::Config(_))), "empty token");
        opts.auth_token = Some(Secret::new("s3cret"));
        let s = serve(&store, &id, opts).await;
        let ok = [("Authorization", "Bearer s3cret")];

        // health needs no token
        assert_eq!(health(s.addr).await["package"], json!(id));

        // auth
        for headers in [
            &[][..],
            &[("Authorization", "Bearer wrong")][..],
            &[("Authorization", "Basic s3cret")][..],
            &[("Authorization", "Bearer ")][..],
        ] {
            let (status, head, body) = call(s.addr, "POST", "/predict", headers, "{}").await;
            assert_eq!(status, 401, "{headers:?}");
            assert!(
                head.to_lowercase().contains("www-authenticate: bearer"),
                "{head}"
            );
            assert!(body.contains("\"error\""), "{body}");
        }

        // valid request: the body (with a marker) is never logged
        let marker = "SECRET-PAYLOAD-MARKER";
        let body = json!({ "note": marker }).to_string();
        let (status, v) = post(s.addr, &ok, &body).await;
        assert_eq!((status, v), (200, json!({"answer": "A"})));
        // 400s carry the payload in serde/transform errors: still not logged
        let (status, v) = post(s.addr, &ok, &format!("{{\"{marker}\": ")).await;
        assert_eq!(status, 400);
        assert!(v["error"].is_string());
        let (status, _) = post(s.addr, &ok, &json!([marker]).to_string()).await;
        assert_eq!(status, 400);

        // body limit
        let big = json!({ "pad": "x".repeat(1000) }).to_string();
        let (status, v) = post(s.addr, &ok, &big).await;
        assert_eq!(status, 413);
        assert!(v["error"].is_string());

        // timeout
        let (status, v) = post(s.addr, &ok, r#"{"sleep_ms": 2000}"#).await;
        assert_eq!(status, 504, "{v}");

        // concurrency: one slow request holds the only permit
        let addr = s.addr;
        let slow = tokio::spawn(async move { post(addr, &ok, r#"{"sleep_ms": 200}"#).await.0 });
        tokio::time::sleep(Duration::from_millis(50)).await;
        let (status, v) = post(s.addr, &ok, "{}").await;
        assert_eq!(status, 503, "{v}");
        assert_eq!(slow.await.unwrap(), 200);

        // predictor failure: generic message
        let (status, v) = post(s.addr, &ok, r#"{"fail": true}"#).await;
        assert_eq!((status, v), (500, json!({"error": "prediction failed"})));

        s.stop.send(()).unwrap();
        s.task.await.unwrap().unwrap();

        let text = String::from_utf8(logs.0.lock().unwrap().clone()).unwrap();
        for want in [
            "status=200",
            "status=400",
            "status=401",
            "status=413",
            "status=503",
            "status=504",
            "status=500",
            "latency_ms=",
            "resp_bytes=",
            &format!("req_bytes={}", body.len()),
        ] {
            assert!(text.contains(want), "missing {want} in logs:\n{text}");
        }
        assert!(!text.contains(marker), "payload logged:\n{text}");
    }

    #[tokio::test]
    async fn serve_http_hot_swap() {
        let store = new_store();
        let a = mock_package(&store, MOCK, "A").await;
        let b = mock_package(&store, MOCK, "B").await;
        let s = serve(&store, &a, ServeOptions::default()).await;
        let a_dir = pkg_dir(&s.handle);

        let stop = Arc::new(AtomicBool::new(false));
        let clients: Vec<_> = (0..4)
            .map(|_| {
                let (addr, stop) = (s.addr, Arc::clone(&stop));
                tokio::spawn(async move {
                    let mut seen = Vec::new();
                    while !stop.load(Ordering::Relaxed) {
                        seen.push(answer(addr, "{}").await);
                    }
                    seen
                })
            })
            .collect();
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(
            s.handle
                .swap(store.as_ref(), &b, &predictors())
                .await
                .unwrap()
        );
        assert!(
            !s.handle
                .swap(store.as_ref(), &b, &predictors())
                .await
                .unwrap()
        );
        tokio::time::sleep(Duration::from_millis(150)).await;
        stop.store(true, Ordering::Relaxed);
        for c in clients {
            let seen = c.await.unwrap();
            // per client: old answers, then new ones only
            let first_b = seen.iter().position(|x| x == "B").unwrap();
            assert!(first_b > 0, "{seen:?}");
            assert!(seen[first_b..].iter().all(|x| x == "B"), "{seen:?}");
        }
        assert_eq!(health(s.addr).await["package"], json!(b));
        assert!(!a_dir.exists(), "old package freed");

        // after run and the last handle, the package dir is gone
        let b_dir = pkg_dir(&s.handle);
        s.stop.send(()).unwrap();
        s.task.await.unwrap().unwrap();
        assert!(b_dir.exists(), "still owned by the handle");
        drop(s.handle);
        assert!(!b_dir.exists());
    }

    #[tokio::test]
    async fn serve_http_swap_inflight() {
        let store = new_store();
        let a = mock_package(&store, MOCK, "A").await;
        let b = mock_package(&store, MOCK, "B").await;
        let s = serve(&store, &a, ServeOptions::default()).await;
        let a_dir = pkg_dir(&s.handle);

        let addr = s.addr;
        // the mock reads its payload file after the sleep: the old dir must still exist
        let slow = tokio::spawn(async move { answer(addr, r#"{"sleep_ms": 300}"#).await });
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            s.handle
                .swap(store.as_ref(), &b, &predictors())
                .await
                .unwrap()
        );
        assert!(a_dir.exists(), "in use by the slow request");
        assert_eq!(answer(s.addr, "{}").await, "B");
        assert_eq!(slow.await.unwrap(), "A");
        assert!(!a_dir.exists(), "freed with the last request");
    }

    #[tokio::test]
    async fn serve_http_swap_failure_keeps_old() {
        let store = new_store();
        let a = mock_package(&store, MOCK, "A").await;
        let s = serve(&store, &a, ServeOptions::default()).await;
        let reg = predictors();

        // absent id
        let missing = ArtifactId::from_bytes([7; 32]);
        let r = s.handle.swap(store.as_ref(), &missing, &reg).await;
        assert!(matches!(r, Err(Error::Config(_))), "{r:?}");
        // trust gate (D7) holds on swap
        let unsafe_pkg = mock_package(&store, UNSAFE, "U").await;
        let r = s.handle.swap(store.as_ref(), &unsafe_pkg, &reg).await;
        assert!(
            matches!(&r, Err(Error::Config(m)) if m.contains("--trust-unsafe-flavors")),
            "{r:?}"
        );
        // a transform that cannot be loaded (or no `preprocess-tabular`)
        let m = Manifest::new(ArtifactKind::Model, "txt").with_flavor(MOCK);
        let model = put(&store, m, PayloadSource::Bytes(b"T")).await;
        let t = Manifest::new(ArtifactKind::Transform, "json");
        let t = put(&store, t, PayloadSource::Bytes(b"{}")).await;
        let with_t = package(&store, &model, Some(&t)).await;
        let r = s.handle.swap(store.as_ref(), &with_t, &reg).await;
        if cfg!(feature = "preprocess-tabular") {
            assert!(r.is_err(), "{r:?}");
        } else {
            assert!(matches!(r, Err(Error::Config(_))), "{r:?}");
        }

        assert_eq!(health(s.addr).await["package"], json!(a));
        assert_eq!(answer(s.addr, "{}").await, "A");
    }
}
