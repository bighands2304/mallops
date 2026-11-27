//! `std/evaluate.classification` and `std/evaluate.regression`: score a model on a held-out
//! `features` artifact and emit a `metrics` artifact (flat JSON `metric → number`), with an
//! optional quality gate.
//!
//! The model is loaded through the [`PredictorRegistry`] by its `flavor`, so any model format with
//! a registered predictor can be evaluated. In `run` the pipeline is trusted, so code-executing
//! flavors (pickle, …) are allowed (ADR-0007 D7). Without a `predict-*` feature the blocks are
//! registered but fail at run time with "no predictor for flavor".
//!
//! Limits (v0.1): the whole test set is read into memory and predicted in one `predict_batch`;
//! cancellation is checked before and after it, not during it (a slow predictor is bounded by its
//! own request timeout).

pub(crate) mod metrics;

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{Array, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray};
use mallops_core::{
    ArtifactKind, Block, BlockOutcome, BlockSpec, BoxFuture, Context, DraftPayload, Error, Inputs,
    Manifest, OutputDraft, OutputDrafts, Port, ResolvedPorts, Stage,
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::has_secret_ref;
use crate::ingest::{DATA_FILE, blocking, parse_config, read_parquet_batch};
use crate::predictor::{PredictInput, PredictorRegistry};
use metrics::{CLASSIFICATION_METRICS, Label, REGRESSION_METRICS};

const MODEL: &str = "model";
const TEST: &str = "test";
const METRICS: &str = "metrics";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Task {
    Classification,
    Regression,
}

impl Task {
    fn id(self) -> &'static str {
        match self {
            Self::Classification => "std/evaluate.classification",
            Self::Regression => "std/evaluate.regression",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Classification => "classification",
            Self::Regression => "regression",
        }
    }

    fn metrics(self) -> &'static [&'static str] {
        match self {
            Self::Classification => CLASSIFICATION_METRICS,
            Self::Regression => REGRESSION_METRICS,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum OnFail {
    #[default]
    Skip,
    Fail,
}

impl OnFail {
    fn as_str(self) -> &'static str {
        match self {
            Self::Skip => "skip",
            Self::Fail => "fail",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gate {
    metric: String,
    min: Option<f64>,
    max: Option<f64>,
    #[serde(default)]
    on_fail: OnFail,
}

impl Gate {
    /// `Ok(())` when `value` is inside `[min, max]`; the failure reason otherwise. Written so a
    /// NaN fails the gate instead of slipping through.
    fn check(&self, value: f64) -> Result<(), String> {
        let passed = self.min.is_none_or(|m| value >= m) && self.max.is_none_or(|m| value <= m);
        if passed {
            return Ok(());
        }
        let bound = match (self.min, self.max) {
            (Some(m), _) if value < m => format!("< min {m}"),
            (_, Some(m)) if value > m => format!("> max {m}"),
            _ => "out of range".to_owned(),
        };
        Err(format!("gate failed: {} = {value} {bound}", self.metric))
    }

    fn to_json(&self, passed: bool) -> Value {
        let mut g =
            json!({ "metric": self.metric, "on_fail": self.on_fail.as_str(), "passed": passed });
        if let Some(m) = self.min {
            g["min"] = json!(m);
        }
        if let Some(m) = self.max {
            g["max"] = json!(m);
        }
        g
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawConfig {
    target: Option<String>,
    positive: Option<Value>,
    gate: Option<Gate>,
}

/// Validated `with:` configuration.
#[derive(Debug)]
struct Config {
    target: Option<String>,
    positive: Option<Label>,
    gate: Option<Gate>,
}

impl Config {
    fn parse(task: Task, config: &Value) -> Result<Self, Error> {
        if has_secret_ref(config) {
            return Err(Error::Config(format!(
                "`${{secret:...}}` is not supported in `{}` config",
                task.id()
            )));
        }
        let raw: RawConfig = parse_config(config)?;
        if raw.target.as_deref().is_some_and(str::is_empty) {
            return Err(Error::Config("target must not be empty".into()));
        }
        let positive = match raw.positive {
            None => None,
            Some(_) if task == Task::Regression => {
                return Err(Error::Config(
                    "`positive` is only valid for classification".into(),
                ));
            }
            Some(v) => Some(Label::from_json(&v).ok_or_else(|| {
                Error::Config("`positive` must be an integer, boolean or string".into())
            })?),
        };
        if let Some(g) = &raw.gate {
            if !task.metrics().contains(&g.metric.as_str()) {
                return Err(Error::Config(format!(
                    "gate metric `{}` is not a {} metric (one of: {})",
                    g.metric,
                    task.name(),
                    task.metrics().join(", ")
                )));
            }
            if g.min.is_none() && g.max.is_none() {
                return Err(Error::Config("gate needs `min` and/or `max`".into()));
            }
            if g.min.into_iter().chain(g.max).any(|v| !v.is_finite()) {
                return Err(Error::Config("gate bounds must be finite".into()));
            }
            if g.min.zip(g.max).is_some_and(|(lo, hi)| lo > hi) {
                return Err(Error::Config("gate `min` must not exceed `max`".into()));
            }
        }
        Ok(Self {
            target: raw.target,
            positive,
            gate: raw.gate,
        })
    }
}

/// The `std/evaluate.classification` / `std/evaluate.regression` block.
pub(crate) struct Evaluate {
    spec: BlockSpec,
    task: Task,
    predictors: Arc<PredictorRegistry>,
}

impl Evaluate {
    /// Creates `std/evaluate.classification`.
    ///
    /// Note: bump `BlockSpec.version` when the semantics of a metric change.
    pub(crate) fn classification(predictors: Arc<PredictorRegistry>) -> Self {
        Self::new(Task::Classification, predictors)
    }

    /// Creates `std/evaluate.regression`.
    ///
    /// Note: bump `BlockSpec.version` when the semantics of a metric change.
    pub(crate) fn regression(predictors: Arc<PredictorRegistry>) -> Self {
        Self::new(Task::Regression, predictors)
    }

    fn new(task: Task, predictors: Arc<PredictorRegistry>) -> Self {
        let spec = BlockSpec::new(task.id(), "0.1.0", Stage::Evaluate)
            .with_input(Port::new(MODEL, ArtifactKind::Model))
            .with_input(Port::new(TEST, ArtifactKind::Features).with_format("parquet"))
            .with_output(Port::new(METRICS, ArtifactKind::Metrics).with_format("json"));
        Self {
            spec,
            task,
            predictors,
        }
    }
}

/// A typed view over one Arrow column of the test set.
enum Col<'a> {
    I(&'a Int64Array),
    F(&'a Float64Array),
    B(&'a BooleanArray),
    S(&'a StringArray),
}

impl<'a> Col<'a> {
    fn new(batch: &'a RecordBatch, name: &str) -> Result<Self, Error> {
        let array = batch
            .column_by_name(name)
            .ok_or_else(|| Error::Runtime(format!("column `{name}` not found in the test set")))?;
        let any = array.as_any();
        any.downcast_ref::<Int64Array>()
            .map(Self::I)
            .or_else(|| any.downcast_ref::<Float64Array>().map(Self::F))
            .or_else(|| any.downcast_ref::<BooleanArray>().map(Self::B))
            .or_else(|| any.downcast_ref::<StringArray>().map(Self::S))
            .ok_or_else(|| {
                Error::Runtime(format!(
                    "column `{name}` has unsupported type {} (use int64, float64, bool or utf8)",
                    array.data_type()
                ))
            })
    }

    fn is_null(&self, i: usize) -> bool {
        match self {
            Self::I(a) => a.is_null(i),
            Self::F(a) => a.is_null(i),
            Self::B(a) => a.is_null(i),
            Self::S(a) => a.is_null(i),
        }
    }

    /// Feature value for the predictor; null and non-finite floats become JSON `null`.
    fn json(&self, i: usize) -> Value {
        if self.is_null(i) {
            return Value::Null;
        }
        match self {
            Self::I(a) => a.value(i).into(),
            Self::F(a) => {
                serde_json::Number::from_f64(a.value(i)).map_or(Value::Null, Value::Number)
            }
            Self::B(a) => a.value(i).into(),
            Self::S(a) => a.value(i).into(),
        }
    }

    fn labels(&self, name: &str) -> Result<Vec<Label>, Error> {
        let n = match self {
            Self::I(a) => a.len(),
            Self::F(_) => {
                return Err(Error::Runtime(format!(
                    "classification target `{name}` must be int64, bool or string, not float64"
                )));
            }
            Self::B(a) => a.len(),
            Self::S(a) => a.len(),
        };
        (0..n)
            .map(|i| {
                if self.is_null(i) {
                    return Err(Error::Runtime(format!(
                        "target `{name}` is null at row {i}"
                    )));
                }
                Ok(match self {
                    Self::I(a) => Label::I64(a.value(i)),
                    Self::B(a) => Label::Bool(a.value(i)),
                    Self::S(a) => Label::Str(a.value(i).to_owned()),
                    Self::F(_) => unreachable!("rejected above"),
                })
            })
            .collect()
    }

    fn numbers(&self, name: &str) -> Result<Vec<f64>, Error> {
        let n = match self {
            Self::I(a) => a.len(),
            Self::F(a) => a.len(),
            _ => {
                return Err(Error::Runtime(format!(
                    "regression target `{name}` must be int64 or float64"
                )));
            }
        };
        (0..n)
            .map(|i| {
                let v = match self {
                    Self::I(a) if !a.is_null(i) => a.value(i) as f64,
                    Self::F(a) if !a.is_null(i) => a.value(i),
                    _ => {
                        return Err(Error::Runtime(format!(
                            "target `{name}` is null at row {i}"
                        )));
                    }
                };
                if v.is_finite() {
                    Ok(v)
                } else {
                    Err(Error::Runtime(format!(
                        "target `{name}` is not finite at row {i}"
                    )))
                }
            })
            .collect()
    }
}

enum Truth {
    Labels(Vec<Label>),
    Numbers(Vec<f64>),
}

struct TestSet {
    rows: Vec<Value>,
    truth: Truth,
}

fn resolve_target(cfg: &Config, manifest: &Manifest) -> Result<String, Error> {
    let meta = manifest
        .metadata
        .get("target")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty());
    match (cfg.target.as_deref(), meta) {
        (Some(c), Some(m)) if c != m => Err(Error::Runtime(format!(
            "with.target `{c}` differs from the test set's metadata target `{m}`"
        ))),
        (Some(t), _) | (None, Some(t)) => Ok(t.to_owned()),
        (None, None) => Err(Error::Runtime(
            "target column not specified in `with.target` or test set manifest metadata".into(),
        )),
    }
}

/// Names of the columns to send to the predictor: the model's `io_schema.input` (strings, or
/// objects with a `name`) when declared, else every column but the target.
fn feature_names(model: &Manifest, batch: &RecordBatch, target: &str) -> Vec<String> {
    let declared = model
        .io_schema
        .as_ref()
        .and_then(|s| s.get("input"))
        .and_then(Value::as_array)
        .and_then(|cols| {
            cols.iter()
                .map(|c| {
                    c.as_str()
                        .or_else(|| c.get("name")?.as_str())
                        .map(str::to_owned)
                })
                .collect::<Option<Vec<_>>>()
        });
    declared.unwrap_or_else(|| {
        batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .filter(|n| n != target)
            .collect()
    })
}

fn load_test_set(
    task: Task,
    path: &std::path::Path,
    model: &Manifest,
    target: &str,
    cancel: &mallops_core::CancelToken,
) -> Result<TestSet, Error> {
    let batch = read_parquet_batch(path, cancel)?;
    if batch.num_rows() == 0 {
        return Err(Error::Runtime("the test set has no rows".into()));
    }
    let truth_col = Col::new(&batch, target)?;
    let truth = match task {
        Task::Classification => Truth::Labels(truth_col.labels(target)?),
        Task::Regression => Truth::Numbers(truth_col.numbers(target)?),
    };
    let names = feature_names(model, &batch, target);
    let cols = names
        .iter()
        .map(|n| Col::new(&batch, n).map(|c| (n, c)))
        .collect::<Result<Vec<_>, _>>()?;
    let rows = (0..batch.num_rows())
        .map(|i| {
            Value::Object(
                cols.iter()
                    .map(|(n, c)| ((*n).clone(), c.json(i)))
                    .collect(),
            )
        })
        .collect();
    Ok(TestSet { rows, truth })
}

/// Predictions parsed from the predictor's outputs.
struct Predictions {
    values: Vec<Value>,
    /// P(larger class) per row, present in every row or in none.
    proba: Option<Vec<f64>>,
}

fn parse_outputs(outputs: Vec<PredictInput>, rows: usize) -> Result<Predictions, Error> {
    if outputs.len() != rows {
        return Err(Error::Runtime(format!(
            "predictor returned {} outputs for {rows} rows",
            outputs.len()
        )));
    }
    let mut values = Vec::with_capacity(rows);
    let mut proba = Vec::new();
    for (i, out) in outputs.into_iter().enumerate() {
        let PredictInput::Json(Value::Object(mut obj)) = out else {
            return Err(Error::Runtime(format!(
                "predictor output {i} is not a JSON object"
            )));
        };
        values.push(
            obj.remove("prediction").ok_or_else(|| {
                Error::Runtime(format!("predictor output {i} has no `prediction`"))
            })?,
        );
        if let Some(p) = obj.remove("probability") {
            let p = p
                .as_f64()
                .filter(|p| (0.0..=1.0).contains(p))
                .ok_or_else(|| {
                    Error::Runtime(format!(
                        "predictor output {i}: `probability` must be in [0, 1]"
                    ))
                })?;
            proba.push(p);
        }
    }
    if !proba.is_empty() && proba.len() != rows {
        return Err(Error::Runtime(
            "`probability` is present in only some predictor outputs".into(),
        ));
    }
    Ok(Predictions {
        values,
        proba: (!proba.is_empty()).then_some(proba),
    })
}

struct Scored {
    values: BTreeMap<String, f64>,
    positive: Option<Label>,
}

fn score(task: Task, cfg: &Config, truth: &Truth, pred: &Predictions) -> Result<Scored, Error> {
    match (task, truth) {
        (Task::Classification, Truth::Labels(y_true)) => {
            let y_pred = pred
                .values
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    Label::from_json(v).ok_or_else(|| {
                        Error::Runtime(format!(
                            "prediction {i} is not an integer, boolean or string: {v}"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let m = metrics::classification(
                y_true,
                &y_pred,
                pred.proba.as_deref(),
                cfg.positive.as_ref(),
            )?;
            Ok(Scored {
                values: m.values,
                positive: m.positive,
            })
        }
        (Task::Regression, Truth::Numbers(y_true)) => {
            let y_pred = pred
                .values
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| {
                        Error::Runtime(format!("prediction {i} is not a finite number: {v}"))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Scored {
                values: metrics::regression(y_true, &y_pred)?,
                positive: None,
            })
        }
        _ => Err(Error::Runtime(
            "internal error: task/target mismatch".into(),
        )),
    }
}

impl Block for Evaluate {
    fn spec(&self) -> &BlockSpec {
        &self.spec
    }

    fn validate_config(&self, config: &Value) -> Result<(), Error> {
        Config::parse(self.task, config).map(|_| ())
    }

    fn resolve_ports(&self, config: &Value) -> Result<ResolvedPorts, Error> {
        Config::parse(self.task, config)?;
        Ok(ResolvedPorts::from(&self.spec))
    }

    fn run<'a>(
        &'a self,
        ctx: &'a Context,
        inputs: Inputs,
    ) -> BoxFuture<'a, Result<BlockOutcome, Error>> {
        Box::pin(async move {
            if ctx.cancel().is_cancelled() {
                return Err(Error::Cancelled);
            }
            let cfg = Config::parse(self.task, ctx.config())?;
            let model = inputs.get(MODEL)?;
            let test = inputs.get(TEST)?;
            let model_manifest = model.manifest().clone();
            let flavor = model_manifest
                .flavor
                .clone()
                .ok_or_else(|| Error::Runtime("model artifact has no `flavor`".into()))?;
            let target = resolve_target(&cfg, test.manifest())?;

            let model_dir = ctx.workdir().join(MODEL);
            let test_dir = ctx.workdir().join(TEST);
            let model_dir = ctx.store().fetch(&model.id(), &model_dir).await?;
            let test_dir = ctx.store().fetch(&test.id(), &test_dir).await?;
            if ctx.cancel().is_cancelled() {
                return Err(Error::Cancelled);
            }

            // The pipeline is trusted at run time (ADR-0007 D7): code-executing flavors allowed.
            let predictor = self.predictors.get(&flavor, true)?;
            let loaded = predictor.load(model, &model_dir).await?;

            let (task, cancel) = (self.task, ctx.cancel().clone());
            let (file, tgt, mm) = (test_dir.join(DATA_FILE), target.clone(), model_manifest);
            let set = blocking(move || load_test_set(task, &file, &mm, &tgt, &cancel)).await?;
            let rows = set.rows.len();
            let inputs = set.rows.into_iter().map(PredictInput::Json).collect();
            let outputs = loaded.predict_batch(inputs).await?;
            if ctx.cancel().is_cancelled() {
                return Err(Error::Cancelled);
            }

            let pred = parse_outputs(outputs, rows)?;
            let scored = score(self.task, &cfg, &set.truth, &pred)?;

            let verdict = match &cfg.gate {
                None => None,
                Some(g) => {
                    let v = *scored.values.get(&g.metric).ok_or_else(|| {
                        Error::Config(format!(
                            "gate metric `{}` is not available for this run (e.g. `roc_auc` needs `probability` and both classes)",
                            g.metric
                        ))
                    })?;
                    Some((g, g.check(v)))
                }
            };

            let payload = serde_json::to_vec(&scored.values)
                .map_err(|e| Error::Runtime(format!("failed to serialize metrics: {e}")))?;
            let mut manifest = Manifest::new(ArtifactKind::Metrics, "json")
                .with_metadata_entry("task", json!(self.task.name()))
                .with_metadata_entry("target", json!(target))
                .with_metadata_entry("rows", json!(rows));
            if let Some(p) = &scored.positive {
                manifest = manifest.with_metadata_entry("positive", p.to_json());
            }
            if let Some((g, res)) = &verdict {
                manifest = manifest.with_metadata_entry("gate", g.to_json(res.is_ok()));
            }
            let drafts = OutputDrafts::new().with(
                METRICS,
                OutputDraft::new(manifest, DraftPayload::Bytes(payload)),
            );
            match verdict {
                Some((g, Err(reason))) => match g.on_fail {
                    OnFail::Skip => Ok(BlockOutcome::Skipped {
                        reason,
                        outputs: drafts,
                    }),
                    OnFail::Fail => Err(Error::Block {
                        id: self.spec.id.to_string(),
                        source: reason.into(),
                    }),
                },
                _ => Ok(BlockOutcome::Produced(drafts)),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use arrow_array::{ArrayRef, RecordBatchIterator};
    use mallops_core::{
        Artifact, ArtifactStore, BlockRegistry, CancelSource, MemorySecretProvider, MemoryStore,
        PayloadSource, PipelineBuilder, RelPath, RunId, Runtime, RuntimeConfig, Status,
    };

    use super::*;
    use crate::ingest::write_parquet;
    use crate::predictor::{LoadedPredictor, Predictor};

    const FLAVOR: &str = "mock";

    /// Echoes column `p` as the prediction and column `q` (when non-null) as the probability, so
    /// fixtures are analytic. `drop_last` makes it return one output too few.
    struct Mock {
        drop_last: bool,
    }

    struct MockLoaded {
        drop_last: bool,
    }

    impl Predictor for Mock {
        fn flavors(&self) -> &[&str] {
            &[FLAVOR]
        }
        fn executes_code(&self) -> bool {
            true // proves `run` passes trust to the registry
        }
        fn load<'a>(
            &'a self,
            _: &'a Artifact,
            _: &'a Path,
        ) -> BoxFuture<'a, Result<Box<dyn LoadedPredictor>, Error>> {
            let drop_last = self.drop_last;
            Box::pin(
                async move { Ok(Box::new(MockLoaded { drop_last }) as Box<dyn LoadedPredictor>) },
            )
        }
    }

    impl LoadedPredictor for MockLoaded {
        fn predict(&self, input: PredictInput) -> BoxFuture<'_, Result<PredictInput, Error>> {
            Box::pin(async move {
                let PredictInput::Json(row) = input else {
                    return Err(Error::Runtime("json only".into()));
                };
                let mut out = json!({ "prediction": row["p"] });
                if !row["q"].is_null() {
                    out["probability"] = row["q"].clone();
                }
                Ok(PredictInput::Json(out))
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
                if self.drop_last {
                    out.pop();
                }
                Ok(out)
            })
        }
    }

    fn registry(drop_last: bool) -> Arc<PredictorRegistry> {
        let mut r = PredictorRegistry::new();
        r.register(Arc::new(Mock { drop_last })).unwrap();
        Arc::new(r)
    }

    fn model_manifest() -> Manifest {
        Manifest::new(ArtifactKind::Model, "mock").with_flavor(FLAVOR)
    }

    fn write_test_parquet(batch: &RecordBatch, path: &Path) {
        let cancel = CancelSource::new().token();
        let iter = RecordBatchIterator::new([Ok(batch.clone())], batch.schema());
        write_parquet(iter, path, path, &cancel).unwrap();
    }

    fn features_manifest(target: Option<&str>) -> Manifest {
        let m = Manifest::new(ArtifactKind::Features, "parquet");
        target.map_or(m.clone(), |t| m.with_metadata_entry("target", json!(t)))
    }

    /// Stores a mock model and a test set; returns the block inputs.
    async fn stage(store: &Arc<MemoryStore>, batch: &RecordBatch, target: Option<&str>) -> Inputs {
        let model = model_manifest();
        let mid = store
            .put(model.clone(), PayloadSource::Bytes(b"m"))
            .await
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(DATA_FILE);
        write_test_parquet(batch, &path);
        let tm = features_manifest(target);
        let tid = store
            .put(tm.clone(), PayloadSource::File(&path))
            .await
            .unwrap();
        Inputs::new()
            .with(MODEL, Artifact::new(mid, model))
            .with(TEST, Artifact::new(tid, tm))
    }

    fn ctx(store: &Arc<MemoryStore>, workdir: PathBuf, config: Value) -> (Context, CancelSource) {
        let cancel = CancelSource::new();
        let ctx = Context::new(
            RunId::new(),
            workdir,
            Arc::clone(store) as Arc<dyn ArtifactStore>,
            config,
            cancel.token(),
            Arc::new(MemorySecretProvider::default()),
        );
        (ctx, cancel)
    }

    async fn run_with(
        block: &Evaluate,
        batch: &RecordBatch,
        meta_target: Option<&str>,
        config: Value,
    ) -> Result<BlockOutcome, Error> {
        let store = Arc::new(MemoryStore::new());
        let inputs = stage(&store, batch, meta_target).await;
        let wd = tempfile::tempdir().unwrap();
        let (ctx, _c) = ctx(&store, wd.path().to_path_buf(), config);
        block.run(&ctx, inputs).await
    }

    fn drafts(o: BlockOutcome) -> OutputDrafts {
        match o {
            BlockOutcome::Produced(d) => d,
            BlockOutcome::Skipped { reason, .. } => panic!("skipped: {reason}"),
            _ => panic!("unexpected outcome"),
        }
    }

    fn metrics_of(d: &OutputDrafts) -> BTreeMap<String, f64> {
        let DraftPayload::Bytes(b) = &d.get(METRICS).unwrap().payload else {
            panic!("expected bytes");
        };
        serde_json::from_slice(b).unwrap()
    }

    fn close(m: &BTreeMap<String, f64>, k: &str, want: f64) {
        assert!((m[k] - want).abs() < 1e-12, "{k}: {} != {want}", m[k]);
    }

    fn cls_batch() -> RecordBatch {
        RecordBatch::try_from_iter([
            (
                "x",
                Arc::new(Int64Array::from(vec![1, 2, 3, 4])) as ArrayRef,
            ),
            ("p", Arc::new(Int64Array::from(vec![0, 1, 1, 1])) as _),
            (
                "q",
                Arc::new(Float64Array::from(vec![0.1, 0.6, 0.4, 0.9])) as _,
            ),
            ("y", Arc::new(Int64Array::from(vec![0, 0, 1, 1])) as _),
        ])
        .unwrap()
    }

    fn reg_batch(p: Vec<f64>) -> RecordBatch {
        RecordBatch::try_from_iter([
            ("p", Arc::new(Float64Array::from(p)) as ArrayRef),
            (
                "y",
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])) as _,
            ),
        ])
        .unwrap()
    }

    #[tokio::test]
    async fn evaluate_classification_metrics() {
        let b = Evaluate::classification(registry(false));
        let out = run_with(&b, &cls_batch(), Some("y"), json!({}))
            .await
            .unwrap();
        let d = drafts(out);
        let m = metrics_of(&d);
        assert_eq!(
            m.keys().map(String::as_str).collect::<Vec<_>>(),
            ["accuracy", "f1", "precision", "recall", "roc_auc"]
        );
        close(&m, "accuracy", 0.75);
        close(&m, "precision", 2.0 / 3.0);
        close(&m, "recall", 1.0);
        close(&m, "f1", 0.8);
        close(&m, "roc_auc", 0.75);
        let man = &d.get(METRICS).unwrap().manifest;
        assert_eq!(man.kind, ArtifactKind::Metrics);
        assert_eq!(man.format, "json");
        assert_eq!(man.metadata["task"], "classification");
        assert_eq!(man.metadata["target"], "y");
        assert_eq!(man.metadata["rows"], 4);
        assert_eq!(man.metadata["positive"], 1);
        assert!(!man.metadata.contains_key("gate"));
    }

    #[tokio::test]
    async fn evaluate_classification_metrics_positive_and_target_override() {
        let b = Evaluate::classification(registry(false));
        // no metadata target: with.target is required and with.positive flips the roles
        let cfg = json!({ "target": "y", "positive": 0 });
        let m = metrics_of(&drafts(
            run_with(&b, &cls_batch(), None, cfg).await.unwrap(),
        ));
        close(&m, "recall", 0.5);
        close(&m, "roc_auc", 0.75);
        // a different with.target is rejected
        let e = run_with(&b, &cls_batch(), Some("y"), json!({ "target": "x" })).await;
        assert!(matches!(e, Err(Error::Runtime(_))));
        // neither source names the target
        let e = run_with(&b, &cls_batch(), None, json!({})).await;
        assert!(matches!(e, Err(Error::Runtime(_))));
    }

    #[tokio::test]
    async fn evaluate_classification_metrics_rejects_bad_outputs() {
        let short = Evaluate::classification(registry(true));
        let e = run_with(&short, &cls_batch(), Some("y"), json!({})).await;
        assert!(matches!(e, Err(Error::Runtime(m)) if m.contains("outputs for")));

        // probability on some rows only
        let batch = RecordBatch::try_from_iter([
            ("p", Arc::new(Int64Array::from(vec![0, 1])) as ArrayRef),
            (
                "q",
                Arc::new(Float64Array::from(vec![Some(0.5), None])) as _,
            ),
            ("y", Arc::new(Int64Array::from(vec![0, 1])) as _),
        ])
        .unwrap();
        let b = Evaluate::classification(registry(false));
        let e = run_with(&b, &batch, Some("y"), json!({})).await;
        assert!(matches!(e, Err(Error::Runtime(m)) if m.contains("only some")));

        // float target is not a class label
        let e = run_with(&b, &reg_batch(vec![1.0; 4]), Some("y"), json!({})).await;
        assert!(matches!(e, Err(Error::Runtime(_))));
    }

    #[tokio::test]
    async fn evaluate_regression_metrics() {
        let b = Evaluate::regression(registry(false));
        let batch = reg_batch(vec![1.0, 2.0, 3.0, 6.0]);
        let d = drafts(run_with(&b, &batch, Some("y"), json!({})).await.unwrap());
        let m = metrics_of(&d);
        assert_eq!(m.len(), 3);
        close(&m, "mae", 0.5);
        close(&m, "rmse", 1.0);
        close(&m, "r2", 0.2);
        let man = &d.get(METRICS).unwrap().manifest;
        assert_eq!(man.metadata["task"], "regression");
        assert!(!man.metadata.contains_key("positive"));

        // overflowing squared error must fail, not produce NaN/inf metrics
        let huge = reg_batch(vec![1e200, -1e200, 1e200, -1e200]);
        let e = run_with(
            &b,
            &huge,
            Some("y"),
            json!({ "gate": { "metric": "rmse", "max": 1.0 } }),
        )
        .await;
        assert!(matches!(e, Err(Error::Runtime(_))), "{e:?}");
    }

    #[test]
    fn evaluate_config_validation() {
        let b = Evaluate::classification(registry(false));
        let r = Evaluate::regression(registry(false));
        let ok = |blk: &Evaluate, v: Value| blk.validate_config(&v).is_ok();
        assert!(ok(&b, json!({})));
        assert!(ok(
            &b,
            json!({ "positive": "yes", "gate": { "metric": "f1", "min": 0.5, "max": 1.0 } })
        ));
        assert!(!ok(&b, json!({ "bogus": 1 })));
        assert!(!ok(&b, json!({ "target": "" })));
        assert!(!ok(&b, json!({ "positive": 1.5 })));
        assert!(!ok(&r, json!({ "positive": 1 })));
        assert!(!ok(&b, json!({ "target": "${secret:T}" })));
        assert!(!ok(&b, json!({ "gate": { "metric": "rmse", "min": 1 } })));
        assert!(!ok(&r, json!({ "gate": { "metric": "f1", "min": 1 } })));
        assert!(!ok(&b, json!({ "gate": { "metric": "f1" } })));
        assert!(!ok(
            &b,
            json!({ "gate": { "metric": "f1", "min": 0.9, "max": 0.1 } })
        ));
        assert!(!ok(
            &b,
            json!({ "gate": { "metric": "f1", "min": 0.1, "on_fail": "nope" } })
        ));
        assert!(ok(
            &r,
            json!({ "gate": { "metric": "rmse", "max": 2, "on_fail": "fail" } })
        ));
    }

    #[tokio::test]
    async fn evaluate_unavailable_gate_metric_is_config_error() {
        // no probability from the predictor -> roc_auc omitted -> gate cannot be evaluated
        let batch = RecordBatch::try_from_iter([
            ("p", Arc::new(Int64Array::from(vec![0, 1])) as ArrayRef),
            ("y", Arc::new(Int64Array::from(vec![0, 1])) as _),
        ])
        .unwrap();
        let b = Evaluate::classification(registry(false));
        for on_fail in ["skip", "fail"] {
            let cfg = json!({ "gate": { "metric": "roc_auc", "min": 0.5, "on_fail": on_fail } });
            let e = run_with(&b, &batch, Some("y"), cfg).await;
            assert!(matches!(e, Err(Error::Config(_))), "{e:?}");
        }
    }

    // ---- AC3: the gate inside the runtime ----

    struct Src {
        spec: BlockSpec,
        batch: Option<RecordBatch>,
    }

    impl Block for Src {
        fn spec(&self) -> &BlockSpec {
            &self.spec
        }
        fn run<'a>(
            &'a self,
            ctx: &'a Context,
            _: Inputs,
        ) -> BoxFuture<'a, Result<BlockOutcome, Error>> {
            Box::pin(async move {
                let draft = match &self.batch {
                    None => OutputDraft::new(model_manifest(), DraftPayload::Bytes(b"m".to_vec())),
                    Some(b) => {
                        let path = ctx.workdir().join(DATA_FILE);
                        write_test_parquet(b, &path);
                        OutputDraft::new(features_manifest(Some("y")), DraftPayload::File(path))
                    }
                };
                Ok(BlockOutcome::Produced(
                    OutputDrafts::new().with("out", draft),
                ))
            })
        }
    }

    struct Down(BlockSpec);

    impl Block for Down {
        fn spec(&self) -> &BlockSpec {
            &self.0
        }
        fn run<'a>(
            &'a self,
            _: &'a Context,
            _: Inputs,
        ) -> BoxFuture<'a, Result<BlockOutcome, Error>> {
            Box::pin(async {
                let d = OutputDraft::new(
                    Manifest::new(ArtifactKind::Blob, "text"),
                    DraftPayload::Bytes(b"x".to_vec()),
                );
                Ok(BlockOutcome::Produced(OutputDrafts::new().with("out", d)))
            })
        }
    }

    async fn gate_run(gate: Value) -> (mallops_core::RunSummary, Arc<MemoryStore>) {
        let mut reg = BlockRegistry::new();
        let src = |id: &'static str, kind, fmt: Option<&'static str>, batch| {
            let mut port = Port::new("out", kind);
            if let Some(f) = fmt {
                port = port.with_format(f);
            }
            Arc::new(Src {
                spec: BlockSpec::new(id, "1", Stage::Preprocess).with_output(port),
                batch,
            })
        };
        reg.register(src("t/model", ArtifactKind::Model, None, None))
            .unwrap();
        reg.register(src(
            "t/test",
            ArtifactKind::Features,
            Some("parquet"),
            Some(cls_batch()),
        ))
        .unwrap();
        reg.register(Arc::new(Evaluate::classification(registry(false))))
            .unwrap();
        reg.register(Arc::new(Down(
            BlockSpec::new("t/down", "1", Stage::Package)
                .with_input(Port::new("metrics", ArtifactKind::Metrics))
                .with_output(Port::new("out", ArtifactKind::Blob)),
        )))
        .unwrap();
        let mut b = PipelineBuilder::new("gate");
        b.block("src_model", "t/model".parse().unwrap());
        b.block("src_test", "t/test".parse().unwrap());
        b.block("eval", "std/evaluate.classification".parse().unwrap())
            .input("model", "src_model.out")
            .input("test", "src_test.out")
            .with(json!({ "gate": gate }));
        b.block("down", "t/down".parse().unwrap())
            .input("metrics", "eval.metrics");
        let p = b.build(&reg).unwrap();
        let store = Arc::new(MemoryStore::new());
        let rt = Runtime::new(RuntimeConfig::new(
            Arc::clone(&store) as Arc<dyn ArtifactStore>
        ));
        let s = rt.run(&p).await.unwrap();
        (s, store)
    }

    async fn lineage_status(
        s: &mallops_core::RunSummary,
        store: &MemoryStore,
        node: &str,
    ) -> String {
        let key = RelPath::parse(&format!("runs/{}/lineage.json", s.run_id)).unwrap();
        let doc: Value =
            serde_json::from_slice(&store.get_meta(&key).await.unwrap().unwrap()).unwrap();
        doc["events"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["event"] == "block_finished" && e["node"] == node)
            .unwrap()["status"]
            .as_str()
            .unwrap()
            .to_owned()
    }

    #[tokio::test]
    async fn evaluate_gate_skip_and_fail() {
        // f1 = 0.8: passes
        let (s, store) = gate_run(json!({ "metric": "f1", "min": 0.5 })).await;
        assert!(s.is_success());
        assert_eq!(s.nodes["eval"].status, Status::Succeeded);
        assert_eq!(s.nodes["down"].status, Status::Succeeded);
        let a = store
            .get(&s.nodes["eval"].outputs["metrics"])
            .await
            .unwrap()
            .unwrap();
        assert_eq!(a.manifest().metadata["gate"]["passed"], true);

        // skip (default on_fail): metrics kept, required consumer skipped, run still succeeds
        let (s, store) = gate_run(json!({ "metric": "f1", "min": 0.9 })).await;
        assert!(s.is_success());
        assert_eq!(s.nodes["eval"].status, Status::Skipped);
        assert_eq!(
            s.nodes["eval"].reason.as_deref(),
            Some("gate failed: f1 = 0.8 < min 0.9")
        );
        assert_eq!(s.nodes["down"].status, Status::Skipped);
        let id = s.nodes["eval"].outputs["metrics"];
        let a = store.get(&id).await.unwrap().unwrap();
        assert_eq!(a.manifest().metadata["gate"]["passed"], false);
        assert_eq!(lineage_status(&s, &store, "eval").await, "skipped");
        assert_eq!(lineage_status(&s, &store, "down").await, "skipped");

        // fail: node fails, descendants cancelled, nothing committed for eval
        let (s, store) = gate_run(json!({ "metric": "f1", "max": 0.1, "on_fail": "fail" })).await;
        assert!(!s.is_success());
        assert_eq!(s.nodes["eval"].status, Status::Failed);
        assert!(s.nodes["eval"].outputs.is_empty());
        assert_eq!(s.nodes["down"].status, Status::Cancelled);
        assert_eq!(lineage_status(&s, &store, "eval").await, "failed");
    }

    #[tokio::test]
    async fn evaluate_cancelled_before_run() {
        let store = Arc::new(MemoryStore::new());
        let inputs = stage(&store, &cls_batch(), Some("y")).await;
        let wd = tempfile::tempdir().unwrap();
        let (ctx, cancel) = ctx(&store, wd.path().to_path_buf(), json!({}));
        cancel.cancel();
        let e = Evaluate::classification(registry(false))
            .run(&ctx, inputs)
            .await;
        assert!(matches!(e, Err(Error::Cancelled)));
    }

    #[tokio::test]
    async fn evaluate_unknown_flavor_is_config_error() {
        let b = Evaluate::classification(Arc::new(PredictorRegistry::new()));
        let e = run_with(&b, &cls_batch(), Some("y"), json!({})).await;
        assert!(matches!(e, Err(Error::Config(m)) if m.contains("no predictor")));
    }

    // ---- AC4: a real trained model, scored through the real predictor ----

    #[cfg(all(feature = "train-smartcore", feature = "predict-smartcore"))]
    #[tokio::test]
    async fn evaluate_with_smartcore_predictor() {
        use crate::train::smartcore::tests::{
            run_block, separable_logistic_batch_int64, store_features,
        };

        let store = Arc::new(MemoryStore::new());
        let batch = separable_logistic_batch_int64();
        let train_in = store_features(&store, &batch, Some("y")).await;
        let (_w, trained) = run_block(
            &store,
            &train_in,
            json!({ "algorithm": "logistic_regression" }),
        )
        .await
        .unwrap();
        let d = trained.get("model").unwrap();
        let mid = store
            .put(d.manifest.clone(), d.payload.as_source())
            .await
            .unwrap();
        let model = Artifact::new(mid, d.manifest.clone());
        // same data as the held-out set: the problem is separable, so the fit is near perfect
        let test = train_in.get("train").unwrap().clone();
        let inputs = Inputs::new().with(MODEL, model).with(TEST, test);

        let mut preds = PredictorRegistry::new();
        crate::predictor::register_std_predictors(&mut preds).unwrap();
        let b = Evaluate::classification(Arc::new(preds));
        let wd = tempfile::tempdir().unwrap();
        let (ctx, _c) = ctx(
            &store,
            wd.path().to_path_buf(),
            json!({ "gate": { "metric": "accuracy", "min": 0.9 } }),
        );
        let out = drafts(b.run(&ctx, inputs).await.unwrap());
        let m = metrics_of(&out);
        assert!(m["accuracy"] >= 0.9, "{m:?}");
        assert!(m["roc_auc"] >= 0.95, "{m:?}");
        assert_eq!(out.get(METRICS).unwrap().manifest.metadata["positive"], 1);
    }
}
