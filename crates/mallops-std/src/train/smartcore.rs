//! `std/train.smartcore`: native Rust classic ML training block powered by `smartcore` 0.7
//! (pure Rust, no BLAS). Supports binary logistic regression and ordinary least-squares linear
//! regression on Parquet `features` artifacts, emitting a deterministic JSON model payload.

use std::collections::BTreeSet;
use std::path::PathBuf;

use ::smartcore::error::{Failed, FailedError};
use ::smartcore::linalg::basic::arrays::Array;
use ::smartcore::linalg::basic::matrix::DenseMatrix;
use ::smartcore::linear::linear_regression::{
    LinearRegression, LinearRegressionParameters, LinearRegressionSolverName,
};
use ::smartcore::linear::logistic_regression::{LogisticRegression, LogisticRegressionParameters};
use arrow_array::{Array as ArrowArray, BooleanArray, Float64Array, Int64Array, StringArray};
use arrow_schema::DataType;
use mallops_core::{
    ArtifactKind, Block, BlockOutcome, BlockSpec, BoxFuture, CancelToken, Context, DraftPayload,
    Error, Inputs, Manifest, OutputDraft, OutputDrafts, Port, ResolvedPorts, Stage,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::has_secret_ref;
use crate::ingest::{Column, ColumnType, DATA_FILE, blocking, parse_config, read_parquet_batch};

/// Required input port: training features Parquet.
pub(crate) const TRAIN: &str = "train";
/// Output port: trained JSON model.
pub(crate) const MODEL: &str = "model";
/// File name of the JSON model payload inside the output artifact.
pub(crate) const MODEL_FILE: &str = "model.json";

const FLAVOR_LOGISTIC: &str = "smartcore-logistic";
const FLAVOR_LINEAR: &str = "smartcore-linear";
const FRAMEWORK: &str = "smartcore";
const FRAMEWORK_VERSION: &str = "0.7";
const MODEL_SCHEMA_VERSION: u32 = 1;
const DEFAULT_ALPHA: f64 = 1.0;
const DEFAULT_THRESHOLD: f64 = 0.5;

/// Algorithm supported by `std/train.smartcore`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Algorithm {
    LogisticRegression,
    LinearRegression,
}

/// Solver for linear regression.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LinearSolver {
    #[default]
    Svd,
    Qr,
}

fn default_alpha() -> f64 {
    DEFAULT_ALPHA
}

/// Hyperparameters for logistic regression.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LogisticHyperparameters {
    #[serde(default = "default_alpha")]
    pub(crate) alpha: f64,
}

impl Default for LogisticHyperparameters {
    fn default() -> Self {
        Self {
            alpha: DEFAULT_ALPHA,
        }
    }
}

/// Hyperparameters for linear regression.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LinearHyperparameters {
    #[serde(default)]
    pub(crate) solver: LinearSolver,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawConfig {
    algorithm: Algorithm,
    #[serde(default)]
    target: Option<String>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    hyperparameters: Option<serde_json::Value>,
}

/// Validated algorithm and its typed hyperparameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum AlgorithmConfig {
    Logistic(LogisticHyperparameters),
    Linear(LinearHyperparameters),
}

/// Validated `with:` configuration of `std/train.smartcore`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TrainSmartcoreConfig {
    pub(crate) algorithm: AlgorithmConfig,
    pub(crate) target: Option<String>,
    pub(crate) seed: Option<u64>,
}

impl TrainSmartcoreConfig {
    /// Deserializes and validates `with:`; every error is an [`Error::Config`].
    pub(crate) fn parse(config: &serde_json::Value) -> Result<Self, Error> {
        if has_secret_ref(config) {
            return Err(Error::Config(
                "`${secret:...}` is not supported in `std/train.smartcore` config".into(),
            ));
        }
        let raw: RawConfig = parse_config(config)?;
        if raw.target.as_deref().is_some_and(str::is_empty) {
            return Err(Error::Config("target must not be empty".into()));
        }
        let algorithm = match raw.algorithm {
            Algorithm::LogisticRegression => {
                let hp: LogisticHyperparameters = match raw.hyperparameters {
                    None => LogisticHyperparameters::default(),
                    Some(v) => parse_config(&v)?,
                };
                if !hp.alpha.is_finite() || hp.alpha < 0.0 {
                    return Err(Error::Config(format!(
                        "hyperparameters.alpha must be a finite number >= 0, got {}",
                        hp.alpha
                    )));
                }
                AlgorithmConfig::Logistic(hp)
            }
            Algorithm::LinearRegression => {
                let hp: LinearHyperparameters = match raw.hyperparameters {
                    None => LinearHyperparameters::default(),
                    Some(v) => parse_config(&v)?,
                };
                AlgorithmConfig::Linear(hp)
            }
        };
        Ok(Self {
            algorithm,
            target: raw.target,
            seed: raw.seed,
        })
    }

    fn algorithm_kind(&self) -> Algorithm {
        match self.algorithm {
            AlgorithmConfig::Logistic(_) => Algorithm::LogisticRegression,
            AlgorithmConfig::Linear(_) => Algorithm::LinearRegression,
        }
    }

    fn flavor(&self) -> &'static str {
        match self.algorithm {
            AlgorithmConfig::Logistic(_) => FLAVOR_LOGISTIC,
            AlgorithmConfig::Linear(_) => FLAVOR_LINEAR,
        }
    }

    fn effective_hyperparameters(&self) -> Result<serde_json::Value, Error> {
        match self.algorithm {
            AlgorithmConfig::Logistic(hp) => serde_json::to_value(hp),
            AlgorithmConfig::Linear(hp) => serde_json::to_value(hp),
        }
        .map_err(|e| Error::Runtime(format!("failed to serialize hyperparameters: {e}")))
    }
}

/// Typed binary classes serialized into `model.json` for logistic regression.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
enum ModelClasses {
    Int64(Vec<i64>),
    Bool(Vec<bool>),
    String(Vec<String>),
}

/// Serializable payload written to `model.json` (fixed field order).
#[derive(Debug, Clone, PartialEq, Serialize)]
struct ModelPayload {
    v: u32,
    algorithm: Algorithm,
    features: Vec<String>,
    target: String,
    intercept: f64,
    coefficients: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    classes: Option<ModelClasses>,
    #[serde(skip_serializing_if = "Option::is_none")]
    threshold: Option<f64>,
}

/// The `std/train.smartcore` block.
pub(crate) struct TrainSmartcore(BlockSpec);

impl TrainSmartcore {
    /// Creates the `std/train.smartcore` block.
    ///
    /// Note: bump `BlockSpec.version` when upgrading smartcore or changing the model format.
    pub(crate) fn new() -> Self {
        Self(
            BlockSpec::new("std/train.smartcore", "0.1.0", Stage::Train)
                .with_input(Port::new(TRAIN, ArtifactKind::Features).with_format("parquet"))
                .with_output(
                    Port::new(MODEL, ArtifactKind::Model)
                        .with_format("json")
                        .with_flavor(FLAVOR_LOGISTIC)
                        .with_flavor(FLAVOR_LINEAR),
                ),
        )
    }
}

impl Block for TrainSmartcore {
    fn spec(&self) -> &BlockSpec {
        &self.0
    }

    fn validate_config(&self, config: &serde_json::Value) -> Result<(), Error> {
        TrainSmartcoreConfig::parse(config).map(|_| ())
    }

    fn resolve_ports(&self, config: &serde_json::Value) -> Result<ResolvedPorts, Error> {
        let cfg = TrainSmartcoreConfig::parse(config)?;
        let output = Port::new(MODEL, ArtifactKind::Model)
            .with_format("json")
            .with_flavor(cfg.flavor());
        Ok(ResolvedPorts::new(self.0.inputs.clone(), vec![output]))
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
            let cfg = TrainSmartcoreConfig::parse(ctx.config())?;
            let train = inputs.get(TRAIN)?;
            let train_manifest = train.manifest().clone();
            let in_dir = ctx.workdir().join("in");
            ctx.store().fetch(&train.id(), &in_dir).await?;
            if ctx.cancel().is_cancelled() {
                return Err(Error::Cancelled);
            }
            let in_file = in_dir.join(DATA_FILE);
            let workdir = ctx.workdir().to_path_buf();
            let cancel = ctx.cancel().clone();
            let drafts =
                blocking(move || execute_blocking(cfg, &train_manifest, in_file, workdir, cancel))
                    .await?;
            Ok(BlockOutcome::Produced(drafts))
        })
    }
}

fn resolve_target(cfg: &TrainSmartcoreConfig, manifest: &Manifest) -> Result<String, Error> {
    let meta_target = manifest
        .metadata
        .get("target")
        .and_then(serde_json::Value::as_str)
        .filter(|s| !s.is_empty());
    match (cfg.target.as_deref(), meta_target) {
        (Some(c), Some(m)) if c != m => Err(Error::Runtime(format!(
            "with.target `{c}` differs from the input's metadata target `{m}`"
        ))),
        (Some(c), _) => Ok(c.to_owned()),
        (None, Some(m)) => Ok(m.to_owned()),
        (None, None) => Err(Error::Runtime(
            "target column not specified in `with.target` or input manifest metadata".into(),
        )),
    }
}

fn map_smartcore_err(err: Failed) -> Error {
    if err.error() == FailedError::SolutionFailed
        || err
            .to_string()
            .to_ascii_lowercase()
            .contains("rank deficient")
    {
        return Error::Runtime(
            "features are linearly dependent (e.g. one_hot with all categories); drop a column"
                .into(),
        );
    }
    Error::Runtime(format!("smartcore fit failed: {err}"))
}

fn cast_finite_f64(col: &arrow_array::ArrayRef, role: &str, name: &str) -> Result<Vec<f64>, Error> {
    if col.null_count() > 0 {
        return Err(Error::Runtime(format!(
            "{role} column `{name}` contains {} null value(s)",
            col.null_count()
        )));
    }
    let cast_arr = arrow_cast::cast(col, &DataType::Float64).map_err(|e| {
        Error::Runtime(format!(
            "failed to cast {role} column `{name}` to float64: {e}"
        ))
    })?;
    let f64_arr = cast_arr
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| {
            Error::Runtime(format!(
                "internal error: {role} column `{name}` did not cast to Float64Array"
            ))
        })?;
    let mut out = Vec::with_capacity(f64_arr.len());
    for (row, &v) in f64_arr.values().iter().enumerate() {
        if !v.is_finite() {
            return Err(Error::Runtime(format!(
                "{role} column `{name}` contains non-finite value `{v}` at row {row}"
            )));
        }
        out.push(v);
    }
    Ok(out)
}

fn encode_binary<T: Ord + Clone>(
    values: impl Iterator<Item = T>,
) -> Result<(Vec<i32>, Vec<T>), Error> {
    let collected: Vec<T> = values.collect();
    let classes: Vec<T> = collected
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if classes.len() != 2 {
        return Err(Error::Runtime(format!(
            "logistic_regression requires binary target (2 classes), found {}",
            classes.len()
        )));
    }
    let y = collected
        .iter()
        .map(|v| i32::from(*v == classes[1]))
        .collect();
    Ok((y, classes))
}

fn extract_features(
    batch: &arrow_array::RecordBatch,
    target_idx: usize,
) -> Result<(Vec<Column>, DenseMatrix<f64>), Error> {
    let nrows = batch.num_rows();
    let schema = batch.schema();
    let mut feature_cols = Vec::with_capacity(schema.fields().len().saturating_sub(1));
    let mut values = Vec::with_capacity(nrows * schema.fields().len().saturating_sub(1));

    for (idx, field) in schema.fields().iter().enumerate() {
        if idx == target_idx {
            continue;
        }
        let ty = match field.data_type() {
            DataType::Boolean => ColumnType::Bool,
            DataType::Int64 => ColumnType::Int64,
            DataType::Float64 => ColumnType::Float64,
            dt => {
                return Err(Error::Runtime(format!(
                    "feature column `{}` has unsupported type `{dt}`; use `encode` or `select` in `std/preprocess.tabular` first",
                    field.name()
                )));
            }
        };
        values.extend(cast_finite_f64(batch.column(idx), "feature", field.name())?);
        feature_cols.push(Column {
            name: field.name().clone(),
            ty,
        });
    }

    if feature_cols.is_empty() {
        return Err(Error::Runtime(
            "input dataset has no feature columns".into(),
        ));
    }

    let x = DenseMatrix::new(nrows, feature_cols.len(), values, true)
        .map_err(|e| Error::Runtime(format!("failed to construct feature matrix: {e}")))?;
    Ok((feature_cols, x))
}

fn extract_logistic_target(
    col: &arrow_array::ArrayRef,
    name: &str,
    dt: &DataType,
) -> Result<(Vec<i32>, ModelClasses, ColumnType), Error> {
    if col.null_count() > 0 {
        return Err(Error::Runtime(format!(
            "target column `{name}` contains {} null value(s)",
            col.null_count()
        )));
    }
    match dt {
        DataType::Int64 => {
            let arr = col
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| Error::Runtime(format!("target column `{name}` is not Int64")))?;
            let (y, classes) = encode_binary(arr.values().iter().copied())?;
            Ok((y, ModelClasses::Int64(classes), ColumnType::Int64))
        }
        DataType::Boolean => {
            let arr = col
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| Error::Runtime(format!("target column `{name}` is not Boolean")))?;
            let (y, classes) = encode_binary((0..arr.len()).map(|i| arr.value(i)))?;
            Ok((y, ModelClasses::Bool(classes), ColumnType::Bool))
        }
        DataType::Utf8 => {
            let arr = col
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| Error::Runtime(format!("target column `{name}` is not Utf8")))?;
            let (y, classes) = encode_binary((0..arr.len()).map(|i| arr.value(i).to_owned()))?;
            Ok((y, ModelClasses::String(classes), ColumnType::String))
        }
        _ => Err(Error::Runtime(format!(
            "logistic_regression target column `{name}` has unsupported type `{dt}`; expected int64, bool, or string"
        ))),
    }
}

fn extract_linear_target(
    col: &arrow_array::ArrayRef,
    name: &str,
    dt: &DataType,
) -> Result<Vec<f64>, Error> {
    if !matches!(dt, DataType::Int64 | DataType::Float64) {
        return Err(Error::Runtime(format!(
            "linear_regression target column `{name}` has unsupported type `{dt}`; expected int64 or float64"
        )));
    }
    cast_finite_f64(col, "target", name)
}

fn check_finite_params(intercept: f64, coefficients: &[f64]) -> Result<(), Error> {
    if !intercept.is_finite() || coefficients.iter().any(|c| !c.is_finite()) {
        return Err(Error::Runtime(
            "fitted model produced non-finite intercept or coefficients".into(),
        ));
    }
    Ok(())
}

fn execute_blocking(
    cfg: TrainSmartcoreConfig,
    train_manifest: &Manifest,
    in_file: PathBuf,
    workdir: PathBuf,
    cancel: CancelToken,
) -> Result<OutputDrafts, Error> {
    let target_name = resolve_target(&cfg, train_manifest)?;
    let batch = read_parquet_batch(&in_file, &cancel)?;
    let schema = batch.schema();
    let target_idx = schema.index_of(&target_name).map_err(|_| {
        Error::Runtime(format!(
            "target column `{target_name}` not found in input schema"
        ))
    })?;
    if batch.num_rows() == 0 {
        return Err(Error::Runtime("input dataset has 0 rows".into()));
    }

    let (feature_cols, x) = extract_features(&batch, target_idx)?;
    let nfeatures = feature_cols.len();
    let target_field = schema.field(target_idx);
    let target_col = batch.column(target_idx);

    let (intercept, coefficients, classes, threshold, output_ty) = match cfg.algorithm {
        AlgorithmConfig::Logistic(hp) => {
            let (y, classes, out_ty) =
                extract_logistic_target(target_col, &target_name, target_field.data_type())?;
            if cancel.is_cancelled() {
                return Err(Error::Cancelled);
            }
            let params = LogisticRegressionParameters::default().with_alpha(hp.alpha);
            let fitted = LogisticRegression::fit(&x, &y, params).map_err(map_smartcore_err)?;
            if cancel.is_cancelled() {
                return Err(Error::Cancelled);
            }
            let intercept = *fitted.intercept().get((0, 0));
            let coef_mat = fitted.coefficients();
            let coefficients: Vec<f64> = (0..nfeatures).map(|j| *coef_mat.get((0, j))).collect();
            (
                intercept,
                coefficients,
                Some(classes),
                Some(DEFAULT_THRESHOLD),
                out_ty,
            )
        }
        AlgorithmConfig::Linear(hp) => {
            if batch.num_rows() < nfeatures + 1 {
                return Err(Error::Runtime(format!(
                    "linear_regression requires at least n_features + 1 rows ({}), got {}",
                    nfeatures + 1,
                    batch.num_rows()
                )));
            }
            let y = extract_linear_target(target_col, &target_name, target_field.data_type())?;
            if cancel.is_cancelled() {
                return Err(Error::Cancelled);
            }
            let solver = match hp.solver {
                LinearSolver::Svd => LinearRegressionSolverName::SVD,
                LinearSolver::Qr => LinearRegressionSolverName::QR,
            };
            let params = LinearRegressionParameters::default().with_solver(solver);
            let fitted = LinearRegression::fit(&x, &y, params).map_err(map_smartcore_err)?;
            if cancel.is_cancelled() {
                return Err(Error::Cancelled);
            }
            let intercept = *fitted.intercept();
            let coef_mat = fitted.coefficients();
            let coefficients: Vec<f64> = (0..nfeatures).map(|j| *coef_mat.get((j, 0))).collect();
            (intercept, coefficients, None, None, ColumnType::Float64)
        }
    };

    check_finite_params(intercept, &coefficients)?;

    if cancel.is_cancelled() {
        return Err(Error::Cancelled);
    }

    let payload = ModelPayload {
        v: MODEL_SCHEMA_VERSION,
        algorithm: cfg.algorithm_kind(),
        features: feature_cols.iter().map(|c| c.name.clone()).collect(),
        target: target_name.clone(),
        intercept,
        coefficients,
        classes,
        threshold,
    };
    let model_dir = workdir.join(MODEL);
    std::fs::create_dir_all(&model_dir)
        .map_err(|e| Error::Runtime(format!("{}: {e}", model_dir.display())))?;
    let model_path = model_dir.join(MODEL_FILE);
    let model_bytes = serde_json::to_vec(&payload)
        .map_err(|e| Error::Runtime(format!("failed to serialize model payload: {e}")))?;
    std::fs::write(&model_path, model_bytes)
        .map_err(|e| Error::Runtime(format!("{}: {e}", model_path.display())))?;

    let output_col = Column {
        name: target_name.clone(),
        ty: output_ty,
    };
    let mut manifest = Manifest::new(ArtifactKind::Model, "json")
        .with_flavor(cfg.flavor())
        .with_io_schema(json!({
            "input": feature_cols,
            "output": [output_col],
        }))
        .with_metadata_entry("algorithm", json!(cfg.algorithm_kind()))
        .with_metadata_entry("hyperparameters", cfg.effective_hyperparameters()?)
        .with_metadata_entry("target", json!(target_name))
        .with_metadata_entry("framework", json!(FRAMEWORK))
        .with_metadata_entry("framework_version", json!(FRAMEWORK_VERSION));
    if let Some(seed) = cfg.seed {
        manifest = manifest.with_metadata_entry("seed", json!(seed));
    }

    Ok(OutputDrafts::new().with(
        MODEL,
        OutputDraft::new(manifest, DraftPayload::File(model_path)),
    ))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, RecordBatchIterator,
        StringArray,
    };
    use mallops_core::{
        Artifact, ArtifactId, ArtifactStore, CachePolicy, CancelSource, MemorySecretProvider,
        MemoryStore, PayloadFile, PayloadSource, RelPath, RunId,
    };
    use serde_json::json;

    use crate::ingest::write_parquet;

    use super::*;

    async fn store_features(
        store: &Arc<MemoryStore>,
        batch: &RecordBatch,
        meta_target: Option<&str>,
    ) -> Inputs {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(DATA_FILE);
        let cancel = CancelSource::new().token();
        let iter = RecordBatchIterator::new([Ok(batch.clone())], batch.schema());
        write_parquet(iter, &path, &path, &cancel).unwrap();

        let mut manifest = Manifest::new(ArtifactKind::Features, "parquet");
        if let Some(t) = meta_target {
            manifest = manifest.with_metadata_entry("target", json!(t));
        }
        let id = store
            .put(manifest.clone(), PayloadSource::File(&path))
            .await
            .unwrap();
        Inputs::new().with(TRAIN, Artifact::new(id, manifest))
    }

    fn make_ctx(
        workdir: PathBuf,
        store: Arc<MemoryStore>,
        config: serde_json::Value,
    ) -> (Context, CancelSource) {
        let cancel = CancelSource::new();
        let ctx = Context::new(
            RunId::new(),
            workdir,
            store,
            config,
            cancel.token(),
            Arc::new(MemorySecretProvider::default()),
        );
        (ctx, cancel)
    }

    fn draft_artifact_id(draft: &OutputDraft) -> ArtifactId {
        let DraftPayload::File(path) = &draft.payload else {
            panic!("expected DraftPayload::File");
        };
        let name = path.file_name().unwrap().to_str().unwrap();
        let rel = RelPath::parse(name).unwrap();
        let bytes = std::fs::read(path).unwrap();
        ArtifactId::compute(&draft.manifest, &[PayloadFile::from_bytes(rel, &bytes)]).unwrap()
    }

    fn read_model_json(draft: &OutputDraft) -> serde_json::Value {
        let DraftPayload::File(path) = &draft.payload else {
            panic!("expected DraftPayload::File");
        };
        let bytes = std::fs::read(path).unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    async fn run_block(
        store: &Arc<MemoryStore>,
        inputs: &Inputs,
        config: serde_json::Value,
    ) -> Result<(tempfile::TempDir, OutputDrafts), Error> {
        let workdir = tempfile::tempdir().unwrap();
        let (ctx, _cancel) = make_ctx(workdir.path().to_path_buf(), Arc::clone(store), config);
        let BlockOutcome::Produced(drafts) =
            TrainSmartcore::new().run(&ctx, inputs.clone()).await?
        else {
            panic!("expected Produced");
        };
        Ok((workdir, drafts))
    }

    fn separable_logistic_batch_int64() -> RecordBatch {
        let x1: Vec<f64> = vec![
            -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5,
            3.0, 3.5, 4.0, 4.5, 5.0,
        ];
        let x2: Vec<i64> = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
        let y: Vec<i64> = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        RecordBatch::try_from_iter([
            ("x1", Arc::new(Float64Array::from(x1)) as ArrayRef),
            ("x2", Arc::new(Int64Array::from(x2)) as _),
            ("y", Arc::new(Int64Array::from(y)) as _),
        ])
        .unwrap()
    }

    fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    #[tokio::test]
    async fn train_smartcore_logistic() {
        let store = Arc::new(MemoryStore::new());
        let batch = separable_logistic_batch_int64();
        let inputs = store_features(&store, &batch, Some("y")).await;

        let (_w, drafts) = run_block(
            &store,
            &inputs,
            json!({ "algorithm": "logistic_regression" }),
        )
        .await
        .unwrap();
        let draft = drafts.get(MODEL).unwrap();
        let manifest = &draft.manifest;

        assert_eq!(manifest.kind, ArtifactKind::Model);
        assert_eq!(manifest.format, "json");
        assert_eq!(manifest.flavor.as_deref(), Some(FLAVOR_LOGISTIC));
        assert_eq!(
            manifest.metadata.get("algorithm"),
            Some(&json!("logistic_regression"))
        );
        assert_eq!(
            manifest.metadata.get("hyperparameters"),
            Some(&json!({ "alpha": 1.0 }))
        );
        assert_eq!(manifest.metadata.get("target"), Some(&json!("y")));
        assert_eq!(
            manifest.metadata.get("framework"),
            Some(&json!("smartcore"))
        );
        assert_eq!(
            manifest.metadata.get("framework_version"),
            Some(&json!("0.7"))
        );
        assert!(!manifest.metadata.contains_key("seed"));
        assert_eq!(
            manifest.io_schema,
            Some(json!({
                "input": [
                    { "name": "x1", "type": "float64" },
                    { "name": "x2", "type": "int64" }
                ],
                "output": [
                    { "name": "y", "type": "int64" }
                ]
            }))
        );

        let model = read_model_json(draft);
        assert_eq!(model["v"], 1);
        assert_eq!(model["algorithm"], "logistic_regression");
        assert_eq!(model["features"], json!(["x1", "x2"]));
        assert_eq!(model["target"], "y");
        assert_eq!(model["classes"], json!([0, 1]));
        assert_eq!(model["threshold"], 0.5);

        let intercept = model["intercept"].as_f64().unwrap();
        let coefs: Vec<f64> = model["coefficients"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        assert_eq!(coefs.len(), 2);
        // x1 grows with class 1 -> positive coefficient.
        assert!(
            coefs[0] > 0.0,
            "expected positive coef for x1, got {}",
            coefs[0]
        );

        // Check training accuracy >= 0.9 using the documented prediction rule.
        let x1 = batch
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let x2 = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let y = batch
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let mut correct = 0;
        for i in 0..batch.num_rows() {
            let z = intercept + coefs[0] * x1.value(i) + coefs[1] * (x2.value(i) as f64);
            let pred = i64::from(sigmoid(z) >= 0.5);
            if pred == y.value(i) {
                correct += 1;
            }
        }
        let acc = correct as f64 / batch.num_rows() as f64;
        assert!(acc >= 0.9, "expected accuracy >= 0.9, got {acc}");
    }

    #[tokio::test]
    async fn train_smartcore_logistic_bool_and_string_targets_and_permuted_rows() {
        let store = Arc::new(MemoryStore::new());

        // Boolean target + boolean feature
        let x_feat = vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0];
        let b_feat = vec![false, false, false, true, true, true];
        let y_bool = vec![false, false, false, true, true, true];
        let batch_bool = RecordBatch::try_from_iter([
            (
                "x",
                Arc::new(Float64Array::from(x_feat.clone())) as ArrayRef,
            ),
            ("flag", Arc::new(BooleanArray::from(b_feat.clone())) as _),
            ("churned", Arc::new(BooleanArray::from(y_bool)) as _),
        ])
        .unwrap();
        let inputs_bool = store_features(&store, &batch_bool, None).await;
        let (_w, d_bool) = run_block(
            &store,
            &inputs_bool,
            json!({ "algorithm": "logistic_regression", "target": "churned" }),
        )
        .await
        .unwrap();
        let draft_bool = d_bool.get(MODEL).unwrap();
        let model_bool = read_model_json(draft_bool);
        assert_eq!(model_bool["classes"], json!([false, true]));
        assert!(model_bool["coefficients"][0].as_f64().unwrap() > 0.0);
        assert_eq!(
            draft_bool.manifest.io_schema,
            Some(json!({
                "input": [
                    { "name": "x", "type": "float64" },
                    { "name": "flag", "type": "bool" }
                ],
                "output": [
                    { "name": "churned", "type": "bool" }
                ]
            }))
        );

        // String target ("no" < "yes")
        let y_str = vec!["no", "no", "no", "yes", "yes", "yes"];
        let batch_str = RecordBatch::try_from_iter([
            (
                "x",
                Arc::new(Float64Array::from(x_feat.clone())) as ArrayRef,
            ),
            ("label", Arc::new(StringArray::from(y_str)) as _),
        ])
        .unwrap();
        let inputs_str = store_features(&store, &batch_str, Some("label")).await;
        let (_w1, d_str) = run_block(
            &store,
            &inputs_str,
            json!({ "algorithm": "logistic_regression" }),
        )
        .await
        .unwrap();
        let draft_str = d_str.get(MODEL).unwrap();
        let model_str = read_model_json(draft_str);
        assert_eq!(model_str["classes"], json!(["no", "yes"]));
        assert!(model_str["coefficients"][0].as_f64().unwrap() > 0.0);
        assert_eq!(
            draft_str.manifest.io_schema,
            Some(json!({
                "input": [{ "name": "x", "type": "float64" }],
                "output": [{ "name": "label", "type": "string" }]
            }))
        );

        // Permuted rows (class "yes" appears first) -> same model.json
        let x_perm = vec![3.0, 2.0, 1.0, -1.0, -2.0, -3.0];
        let y_perm = vec!["yes", "yes", "yes", "no", "no", "no"];
        let batch_perm = RecordBatch::try_from_iter([
            ("x", Arc::new(Float64Array::from(x_perm)) as ArrayRef),
            ("label", Arc::new(StringArray::from(y_perm)) as _),
        ])
        .unwrap();
        let inputs_perm = store_features(&store, &batch_perm, Some("label")).await;
        let (_w2, d_perm) = run_block(
            &store,
            &inputs_perm,
            json!({ "algorithm": "logistic_regression" }),
        )
        .await
        .unwrap();
        let model_perm = read_model_json(d_perm.get(MODEL).unwrap());
        assert_eq!(model_str["v"], model_perm["v"]);
        assert_eq!(model_str["algorithm"], model_perm["algorithm"]);
        assert_eq!(model_str["features"], model_perm["features"]);
        assert_eq!(model_str["target"], model_perm["target"]);
        assert_eq!(model_str["classes"], model_perm["classes"]);
        assert_eq!(model_str["threshold"], model_perm["threshold"]);
        assert!(
            (model_str["intercept"].as_f64().unwrap() - model_perm["intercept"].as_f64().unwrap())
                .abs()
                < 1e-12
        );
        assert!(
            (model_str["coefficients"][0].as_f64().unwrap()
                - model_perm["coefficients"][0].as_f64().unwrap())
            .abs()
                < 1e-12
        );
    }

    #[tokio::test]
    async fn train_smartcore_linear() {
        let store = Arc::new(MemoryStore::new());
        // y = 2*x1 - 3*x2 + 1 (integer target to verify output type is still float64)
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 2.0];
        let x2 = vec![0.0, 1.0, 0.0, 2.0, 1.0, 3.0];
        let y: Vec<i64> = x1
            .iter()
            .zip(&x2)
            .map(|(&a, &b)| (2.0 * a - 3.0 * b + 1.0) as i64)
            .collect();
        let batch = RecordBatch::try_from_iter([
            ("x1", Arc::new(Float64Array::from(x1)) as ArrayRef),
            ("x2", Arc::new(Float64Array::from(x2)) as _),
            ("y", Arc::new(Int64Array::from(y)) as _),
        ])
        .unwrap();
        let inputs = store_features(&store, &batch, Some("y")).await;

        for (cfg, expected_solver) in [
            (json!({ "algorithm": "linear_regression" }), "svd"),
            (
                json!({ "algorithm": "linear_regression", "hyperparameters": { "solver": "qr" } }),
                "qr",
            ),
        ] {
            let (_w, drafts) = run_block(&store, &inputs, cfg).await.unwrap();
            let draft = drafts.get(MODEL).unwrap();
            let manifest = &draft.manifest;
            assert_eq!(manifest.kind, ArtifactKind::Model);
            assert_eq!(manifest.format, "json");
            assert_eq!(manifest.flavor.as_deref(), Some(FLAVOR_LINEAR));
            assert_eq!(
                manifest.metadata.get("hyperparameters"),
                Some(&json!({ "solver": expected_solver }))
            );
            assert_eq!(
                manifest.io_schema,
                Some(json!({
                    "input": [
                        { "name": "x1", "type": "float64" },
                        { "name": "x2", "type": "float64" }
                    ],
                    "output": [
                        { "name": "y", "type": "float64" }
                    ]
                }))
            );

            let model = read_model_json(draft);
            assert_eq!(model["v"], 1);
            assert_eq!(model["algorithm"], "linear_regression");
            assert_eq!(model["features"], json!(["x1", "x2"]));
            assert_eq!(model["target"], "y");
            assert!(model.get("classes").is_none());
            assert!(model.get("threshold").is_none());
            let intercept = model["intercept"].as_f64().unwrap();
            let c0 = model["coefficients"][0].as_f64().unwrap();
            let c1 = model["coefficients"][1].as_f64().unwrap();
            assert!((intercept - 1.0).abs() < 1e-6, "intercept={intercept}");
            assert!((c0 - 2.0).abs() < 1e-6, "c0={c0}");
            assert!((c1 - (-3.0)).abs() < 1e-6, "c1={c1}");
        }
    }

    #[tokio::test]
    async fn train_smartcore_deterministic() {
        let store = Arc::new(MemoryStore::new());
        let batch = separable_logistic_batch_int64();
        let inputs = store_features(&store, &batch, Some("y")).await;

        let cfg1 = json!({ "algorithm": "logistic_regression", "seed": 42 });
        let cfg2 = json!({ "algorithm": "logistic_regression", "seed": 99 });

        let (_w1, d1) = run_block(&store, &inputs, cfg1.clone()).await.unwrap();
        let (_w2, d2) = run_block(&store, &inputs, cfg1).await.unwrap();
        let (_w3, d3) = run_block(&store, &inputs, cfg2).await.unwrap();

        let id1 = draft_artifact_id(d1.get(MODEL).unwrap());
        let id2 = draft_artifact_id(d2.get(MODEL).unwrap());
        let id3 = draft_artifact_id(d3.get(MODEL).unwrap());
        assert_eq!(id1, id2, "two runs with same config must share ArtifactId");
        assert_ne!(id1, id3, "different seed must produce different ArtifactId");
        assert_eq!(
            d1.get(MODEL).unwrap().manifest.metadata.get("seed"),
            Some(&json!(42))
        );

        // CachePolicy must be Deterministic { extra_hash: None }.
        let dir = tempfile::tempdir().unwrap();
        let (ctx, _) = make_ctx(
            dir.path().to_path_buf(),
            store,
            json!({ "algorithm": "logistic_regression" }),
        );
        assert_eq!(
            TrainSmartcore::new().cache_policy(&ctx).await.unwrap(),
            CachePolicy::Deterministic { extra_hash: None }
        );
    }

    #[test]
    fn train_smartcore_resolve_ports() {
        let block = TrainSmartcore::new();
        // Static spec lists both flavors.
        assert_eq!(
            block.spec().outputs[0].flavors,
            [FLAVOR_LOGISTIC, FLAVOR_LINEAR]
        );

        let p_log = block
            .resolve_ports(&json!({ "algorithm": "logistic_regression" }))
            .unwrap();
        assert_eq!(p_log.outputs[0].flavors, [FLAVOR_LOGISTIC]);
        assert_eq!(p_log.inputs[0].name, TRAIN);
        assert!(p_log.inputs[0].required);

        let p_lin = block
            .resolve_ports(&json!({ "algorithm": "linear_regression" }))
            .unwrap();
        assert_eq!(p_lin.outputs[0].flavors, [FLAVOR_LINEAR]);

        // validate_config rejects invalid configs.
        let invalid = [
            json!({ "algorithm": "random_forest" }),
            json!({ "algorithm": "logistic_regression", "hyperparameters": { "alpha": -1.0 } }),
            json!({ "algorithm": "logistic_regression", "hyperparameters": { "alpha": "NaN" } }),
            json!({ "algorithm": "logistic_regression", "hyperparameters": { "alpha": true } }),
            json!({ "algorithm": "logistic_regression", "unknown_key": 1 }),
            json!({ "algorithm": "linear_regression", "hyperparameters": { "alpha": 1.0 } }),
            json!({ "algorithm": "logistic_regression", "hyperparameters": { "solver": "svd" } }),
            json!({ "algorithm": "linear_regression", "hyperparameters": { "solver": "lu" } }),
            json!({ "algorithm": "logistic_regression", "target": "${secret:TOK}" }),
            json!({ "algorithm": "logistic_regression", "target": "" }),
        ];
        for cfg in invalid {
            let err = block.validate_config(&cfg);
            assert!(
                matches!(err, Err(Error::Config(_))),
                "expected Error::Config for {cfg}, got {err:?}"
            );
        }
    }

    #[tokio::test]
    async fn train_smartcore_errors_and_cancel() {
        let store = Arc::new(MemoryStore::new());

        // Missing target (neither with.target nor metadata.target)
        let batch = separable_logistic_batch_int64();
        let inputs_no_meta = store_features(&store, &batch, None).await;
        let err = run_block(
            &store,
            &inputs_no_meta,
            json!({ "algorithm": "logistic_regression" }),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, Error::Runtime(m) if m.contains("target")));

        // Conflicting target
        let inputs_meta_y = store_features(&store, &batch, Some("y")).await;
        let err = run_block(
            &store,
            &inputs_meta_y,
            json!({ "algorithm": "logistic_regression", "target": "x2" }),
        )
        .await
        .unwrap_err();
        assert!(
            matches!(&err, Error::Runtime(m) if m.contains("with.target `x2` differs from the input's metadata target `y`")),
            "{err:?}"
        );

        // Null in feature
        let batch_null = RecordBatch::try_from_iter([
            (
                "x",
                Arc::new(Float64Array::from(vec![
                    Some(1.0),
                    None,
                    Some(3.0),
                    Some(4.0),
                ])) as ArrayRef,
            ),
            ("y", Arc::new(Int64Array::from(vec![0, 0, 1, 1])) as _),
        ])
        .unwrap();
        let err = run_block(
            &store,
            &store_features(&store, &batch_null, Some("y")).await,
            json!({ "algorithm": "logistic_regression" }),
        )
        .await
        .unwrap_err();
        assert!(
            matches!(&err, Error::Runtime(m) if m.contains("`x`") && m.contains("null")),
            "{err:?}"
        );

        // NaN in feature
        let batch_nan = RecordBatch::try_from_iter([
            (
                "x",
                Arc::new(Float64Array::from(vec![1.0, f64::NAN, 3.0, 4.0])) as ArrayRef,
            ),
            ("y", Arc::new(Int64Array::from(vec![0, 0, 1, 1])) as _),
        ])
        .unwrap();
        let err = run_block(
            &store,
            &store_features(&store, &batch_nan, Some("y")).await,
            json!({ "algorithm": "logistic_regression" }),
        )
        .await
        .unwrap_err();
        assert!(
            matches!(&err, Error::Runtime(m) if m.contains("`x`") && m.contains("row 1")),
            "{err:?}"
        );

        // String feature -> suggests encode/select
        let batch_str_feat = RecordBatch::try_from_iter([
            (
                "cat",
                Arc::new(StringArray::from(vec!["a", "b", "a", "b"])) as ArrayRef,
            ),
            ("y", Arc::new(Int64Array::from(vec![0, 0, 1, 1])) as _),
        ])
        .unwrap();
        let err = run_block(
            &store,
            &store_features(&store, &batch_str_feat, Some("y")).await,
            json!({ "algorithm": "logistic_regression" }),
        )
        .await
        .unwrap_err();
        assert!(
            matches!(&err, Error::Runtime(m) if m.contains("encode") && m.contains("select")),
            "{err:?}"
        );

        // 1 class and 3 classes -> error
        for y_vals in [vec![0, 0, 0, 0], vec![0, 1, 2, 0]] {
            let b = RecordBatch::try_from_iter([
                (
                    "x",
                    Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])) as ArrayRef,
                ),
                ("y", Arc::new(Int64Array::from(y_vals)) as _),
            ])
            .unwrap();
            let err = run_block(
                &store,
                &store_features(&store, &b, Some("y")).await,
                json!({ "algorithm": "logistic_regression" }),
            )
            .await
            .unwrap_err();
            assert!(
                matches!(&err, Error::Runtime(m) if m.contains("2 classes")),
                "{err:?}"
            );
        }

        // Float target for logistic -> error
        let b_float_target = RecordBatch::try_from_iter([
            (
                "x",
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])) as ArrayRef,
            ),
            (
                "y",
                Arc::new(Float64Array::from(vec![0.0, 0.0, 1.0, 1.0])) as _,
            ),
        ])
        .unwrap();
        let err = run_block(
            &store,
            &store_features(&store, &b_float_target, Some("y")).await,
            json!({ "algorithm": "logistic_regression" }),
        )
        .await
        .unwrap_err();
        assert!(
            matches!(&err, Error::Runtime(m) if m.contains("unsupported type")),
            "{err:?}"
        );

        // 0 rows -> error
        let b_empty = RecordBatch::try_from_iter([
            (
                "x",
                Arc::new(Float64Array::from(Vec::<f64>::new())) as ArrayRef,
            ),
            ("y", Arc::new(Int64Array::from(Vec::<i64>::new())) as _),
        ])
        .unwrap();
        let err = run_block(
            &store,
            &store_features(&store, &b_empty, Some("y")).await,
            json!({ "algorithm": "logistic_regression" }),
        )
        .await
        .unwrap_err();
        assert!(
            matches!(&err, Error::Runtime(m) if m.contains("0 rows")),
            "{err:?}"
        );

        // Target not in schema -> error with target name
        let err = run_block(
            &store,
            &inputs_no_meta,
            json!({ "algorithm": "logistic_regression", "target": "missing_col" }),
        )
        .await
        .unwrap_err();
        assert!(
            matches!(&err, Error::Runtime(m) if m.contains("`missing_col`")),
            "{err:?}"
        );

        // No feature columns -> error
        let b_no_feat = RecordBatch::try_from_iter([(
            "y",
            Arc::new(Int64Array::from(vec![0, 0, 1, 1])) as ArrayRef,
        )])
        .unwrap();
        let err = run_block(
            &store,
            &store_features(&store, &b_no_feat, Some("y")).await,
            json!({ "algorithm": "logistic_regression" }),
        )
        .await
        .unwrap_err();
        assert!(
            matches!(&err, Error::Runtime(m) if m.contains("no feature")),
            "{err:?}"
        );

        // Linear with nrows < nfeatures + 1 -> error
        let b_under = RecordBatch::try_from_iter([
            (
                "x1",
                Arc::new(Float64Array::from(vec![1.0, 2.0])) as ArrayRef,
            ),
            ("x2", Arc::new(Float64Array::from(vec![3.0, 4.0])) as _),
            ("y", Arc::new(Float64Array::from(vec![5.0, 6.0])) as _),
        ])
        .unwrap();
        let err = run_block(
            &store,
            &store_features(&store, &b_under, Some("y")).await,
            json!({ "algorithm": "linear_regression" }),
        )
        .await
        .unwrap_err();
        assert!(
            matches!(&err, Error::Runtime(m) if m.contains("n_features + 1")),
            "{err:?}"
        );

        // Collinear columns (constant column collinear with intercept): QR returns helpful linear
        // dependence error, SVD does not panic.
        let b_collinear = RecordBatch::try_from_iter([
            (
                "a",
                Arc::new(Float64Array::from(vec![1.0, 1.0, 1.0, 1.0])) as ArrayRef,
            ),
            (
                "b",
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])) as _,
            ),
            (
                "y",
                Arc::new(Float64Array::from(vec![2.0, 4.0, 6.0, 8.0])) as _,
            ),
        ])
        .unwrap();
        let inputs_col = store_features(&store, &b_collinear, Some("y")).await;
        let err = run_block(
            &store,
            &inputs_col,
            json!({ "algorithm": "linear_regression", "hyperparameters": { "solver": "qr" } }),
        )
        .await
        .unwrap_err();
        assert!(
            matches!(&err, Error::Runtime(m) if m.contains("features are linearly dependent")),
            "{err:?}"
        );
        // SVD handles collinear columns without panicking.
        assert!(
            run_block(
                &store,
                &inputs_col,
                json!({ "algorithm": "linear_regression", "hyperparameters": { "solver": "svd" } }),
            )
            .await
            .is_ok()
        );

        // Cancel before run -> Error::Cancelled
        let workdir = tempfile::tempdir().unwrap();
        let (ctx, cancel) = make_ctx(
            workdir.path().to_path_buf(),
            Arc::clone(&store),
            json!({ "algorithm": "logistic_regression" }),
        );
        cancel.cancel();
        let err = TrainSmartcore::new()
            .run(&ctx, inputs_meta_y)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Cancelled), "{err:?}");
    }
}
