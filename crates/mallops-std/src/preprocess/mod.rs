//! `std/preprocess.tabular`: tabular transforms on a Parquet `dataset` into feature splits plus a
//! fitted, serializable `transform` artifact (FR-PRE-1, FR-PRE-3).
//!
//! Steps run in declared order. `split` appears exactly once; only stateless steps (`select`,
//! `filter`) may precede it, so stateful steps (`impute`, `scale`, `encode`) are always fitted on
//! the training partition only (no leakage) and applied to every partition.

mod split;
pub(crate) mod steps;
pub mod transform;

use std::fs::File;
use std::path::{Path, PathBuf};

use ::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use arrow_array::{RecordBatch, RecordBatchIterator};
use arrow_schema::Schema;
use arrow_select::concat::concat_batches;
use mallops_core::{
    ArtifactKind, Block, BlockOutcome, BlockSpec, BoxFuture, CancelToken, Context, DraftPayload,
    Error, Inputs, Manifest, OutputDraft, OutputDrafts, Port, ResolvedPorts, Stage,
};
use serde::Deserialize;
use serde_json::json;

use crate::ingest::{Column, ColumnType, DATA_FILE, blocking, parse_config, write_parquet};
use transform::{FittedTransform, TRANSFORM_FILE};

/// Input port: the dataset to transform.
pub(crate) const INPUT: &str = "data";
/// Output port: training features.
pub(crate) const TRAIN: &str = "train";
/// Output port: validation features (only when `split.validation` is set).
pub(crate) const VALIDATION: &str = "validation";
/// Output port: test features.
pub(crate) const TEST: &str = "test";
/// Output port: fitted transform.
pub(crate) const TRANSFORM: &str = "transform";

/// `with:` config of `std/preprocess.tabular`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PreprocessConfig {
    /// Target column: kept by `select`, never transformed, recorded in `metadata["target"]`.
    pub(crate) target: String,
    /// Steps in execution order.
    pub(crate) steps: Vec<Step>,
}

/// One preprocessing step (`- <name>: { … }` in YAML).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Step {
    /// Keep only these feature columns (the target is always kept).
    Select(SelectStep),
    /// Keep rows matching a predicate (not replayed at serving time).
    Filter(FilterStep),
    /// Replace nulls with a value fitted on the training partition.
    Impute(ImputeStep),
    /// Scale numeric columns with parameters fitted on the training partition.
    Scale(ScaleStep),
    /// Encode string columns with categories fitted on the training partition.
    Encode(EncodeStep),
    /// Seeded row split into train / validation / test.
    Split(SplitStep),
}

impl Step {
    fn name(&self) -> &'static str {
        match self {
            Self::Select(_) => "select",
            Self::Filter(_) => "filter",
            Self::Impute(_) => "impute",
            Self::Scale(_) => "scale",
            Self::Encode(_) => "encode",
            Self::Split(_) => "split",
        }
    }

    fn is_fitted(&self) -> bool {
        matches!(self, Self::Impute(_) | Self::Scale(_) | Self::Encode(_))
    }
}

/// `select: { columns: [..] }`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SelectStep {
    pub(crate) columns: Vec<String>,
}

/// `filter: { column, op, value? }`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FilterStep {
    pub(crate) column: String,
    pub(crate) op: FilterOp,
    /// Required for comparisons, forbidden for `is_null` / `not_null`.
    pub(crate) value: Option<serde_json::Value>,
}

/// Comparison of a [`FilterStep`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FilterOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    IsNull,
    NotNull,
}

/// `impute: { strategy, value?, columns? }`; default columns: numeric non-target.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ImputeStep {
    pub(crate) strategy: ImputeStrategy,
    /// Required for `constant`, forbidden otherwise.
    pub(crate) value: Option<serde_json::Value>,
    pub(crate) columns: Option<Vec<String>>,
}

/// Strategy of an [`ImputeStep`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ImputeStrategy {
    Mean,
    Median,
    Constant,
}

/// `scale: { method, columns? }`; default columns: numeric non-target.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ScaleStep {
    pub(crate) method: ScaleMethod,
    pub(crate) columns: Option<Vec<String>>,
}

/// Method of a [`ScaleStep`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ScaleMethod {
    Standard,
    Minmax,
}

/// `encode: { method, columns? }`; default columns: string non-target.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EncodeStep {
    pub(crate) method: EncodeMethod,
    pub(crate) columns: Option<Vec<String>>,
}

/// Method of an [`EncodeStep`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum EncodeMethod {
    Ordinal,
    OneHot,
}

/// `split: { test, validation?, seed }`.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SplitStep {
    /// Fraction of rows in `test`, in `(0, 1)`.
    pub(crate) test: f64,
    /// Fraction of rows in `validation`, in `(0, 1)`; absent = no `validation` output.
    pub(crate) validation: Option<f64>,
    pub(crate) seed: u64,
}

impl PreprocessConfig {
    /// Parses and checks the config; every error is an [`Error::Config`].
    pub(crate) fn parse(config: &serde_json::Value) -> Result<Self, Error> {
        let cfg: Self = parse_config(config)?;
        cfg.check()?;
        Ok(cfg)
    }

    /// The single `split` step (present after [`Self::parse`]).
    pub(crate) fn split(&self) -> Option<SplitStep> {
        self.steps.iter().find_map(|s| match s {
            Step::Split(s) => Some(*s),
            _ => None,
        })
    }

    fn check(&self) -> Result<(), Error> {
        let err = |i: usize, msg: String| Err(Error::Config(format!("steps[{i}]: {msg}")));
        if self.target.is_empty() {
            return Err(Error::Config("target must not be empty".into()));
        }
        let mut split_at = None;
        for (i, step) in self.steps.iter().enumerate() {
            match step {
                Step::Split(s) => {
                    if split_at.is_some() {
                        return err(i, "`split` must appear exactly once".into());
                    }
                    split_at = Some(i);
                    let ratio_ok = |r: f64| r > 0.0 && r < 1.0;
                    if !ratio_ok(s.test) {
                        return err(i, format!("split.test {} must be in (0, 1)", s.test));
                    }
                    if let Some(v) = s.validation {
                        if !ratio_ok(v) || s.test + v >= 1.0 {
                            return err(
                                i,
                                format!(
                                    "split.validation {v} must be in (0, 1) with test + validation < 1"
                                ),
                            );
                        }
                    }
                }
                s if s.is_fitted() && split_at.is_none() => {
                    return err(
                        i,
                        format!(
                            "`{}` must come after `split`: fitted steps are fitted on the training split only",
                            s.name()
                        ),
                    );
                }
                Step::Select(s) if s.columns.is_empty() => {
                    return err(i, "select.columns must not be empty".into());
                }
                Step::Filter(f) => {
                    let unary = matches!(f.op, FilterOp::IsNull | FilterOp::NotNull);
                    match (&f.value, unary) {
                        (Some(_), true) => {
                            return err(
                                i,
                                "filter.value is not allowed with is_null/not_null".into(),
                            );
                        }
                        (None, false) => return err(i, "filter.value is required".into()),
                        (Some(v), false) if !(v.is_number() || v.is_string() || v.is_boolean()) => {
                            return err(i, "filter.value must be a number, string or bool".into());
                        }
                        _ => {}
                    }
                }
                Step::Impute(s) => match (s.strategy, &s.value) {
                    (ImputeStrategy::Constant, None) => {
                        return err(i, "impute.value is required with strategy constant".into());
                    }
                    (ImputeStrategy::Constant, Some(v)) if !(v.is_number() || v.is_string()) => {
                        return err(i, "impute.value must be a number or string".into());
                    }
                    (ImputeStrategy::Mean | ImputeStrategy::Median, Some(_)) => {
                        return err(
                            i,
                            "impute.value is only allowed with strategy constant".into(),
                        );
                    }
                    _ => {}
                },
                _ => {}
            }
        }
        if split_at.is_none() {
            return Err(Error::Config("steps: a `split` step is required".into()));
        }
        Ok(())
    }
}

/// Features output port (`kind: features`, `format: parquet`).
fn features_port(name: &'static str) -> Port {
    Port::new(name, ArtifactKind::Features).with_format("parquet")
}

/// `std/preprocess.tabular` block.
pub(crate) struct PreprocessTabular(BlockSpec);

impl PreprocessTabular {
    pub(crate) fn new() -> Self {
        Self(
            BlockSpec::new("std/preprocess.tabular", "0.1.0", Stage::Preprocess)
                .with_input(Port::new(INPUT, ArtifactKind::Dataset).with_format("parquet"))
                .with_output(features_port(TRAIN))
                .with_output(features_port(TEST))
                .with_output(Port::new(TRANSFORM, ArtifactKind::Transform).with_format("json")),
        )
    }
}

/// Maps every Arrow field of `schema` into the v0.1 [`Column`] subset.
fn columns_from_schema(schema: &Schema) -> Result<Vec<Column>, Error> {
    schema
        .fields()
        .iter()
        .map(|f| {
            let ty = ColumnType::from_data_type(f.data_type()).ok_or_else(|| {
                Error::Runtime(format!(
                    "column `{}` has unsupported type `{}`; supported types are bool, int64, float64, string, date32 — use a `select` step before `split` to exclude it",
                    f.name(),
                    f.data_type()
                ))
            })?;
            Ok(Column {
                name: f.name().clone(),
                ty,
            })
        })
        .collect()
}

/// Reads a Parquet file into a single concatenated [`RecordBatch`], checking `cancel` between
/// batches.
fn read_parquet_batch(path: &Path, cancel: &CancelToken) -> Result<RecordBatch, Error> {
    let err = |e: &dyn std::fmt::Display| Error::Runtime(format!("{}: {e}", path.display()));
    let file = File::open(path)
        .map_err(|e| Error::Runtime(format!("cannot read {}: {e}", path.display())))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| err(&e))?;
    let schema = builder.schema().clone();
    let reader = builder.build().map_err(|e| err(&e))?;
    let mut batches = Vec::new();
    for batch in reader {
        if cancel.is_cancelled() {
            return Err(Error::Cancelled);
        }
        batches.push(batch.map_err(|e| err(&e))?);
    }
    concat_batches(&schema, &batches).map_err(|e| err(&e))
}

/// Writes a single partition batch to `<workdir>/<port>/data.parquet`.
fn write_partition(
    workdir: &Path,
    port: &str,
    batch: RecordBatch,
    source: &Path,
    cancel: &CancelToken,
) -> Result<PathBuf, Error> {
    let dir = workdir.join(port);
    std::fs::create_dir_all(&dir).map_err(|e| Error::Runtime(format!("{}: {e}", dir.display())))?;
    let path = dir.join(DATA_FILE);
    let schema = batch.schema();
    let iter = RecordBatchIterator::new([Ok(batch)], schema);
    write_parquet(iter, &path, source, cancel)?;
    Ok(path)
}

fn features_manifest(target: &str, io_schema: &serde_json::Value) -> Manifest {
    Manifest::new(ArtifactKind::Features, "parquet")
        .with_metadata_entry("target", json!(target))
        .with_io_schema(io_schema.clone())
}

fn execute_blocking(
    cfg: PreprocessConfig,
    in_file: PathBuf,
    workdir: PathBuf,
    cancel: CancelToken,
) -> Result<OutputDrafts, Error> {
    let mut batch = read_parquet_batch(&in_file, &cancel)?;
    if batch.schema().index_of(&cfg.target).is_err() {
        return Err(Error::Runtime(format!(
            "target column `{}` not found in input dataset",
            cfg.target
        )));
    }

    let mut fitted_steps = Vec::new();
    let mut splits: Option<split::SplitBatches> = None;
    let mut transform_input: Option<Vec<Column>> = None;

    for step in &cfg.steps {
        if cancel.is_cancelled() {
            return Err(Error::Cancelled);
        }
        match (&mut splits, step) {
            (None, Step::Select(_)) => {
                let fitted = steps::fit(step, &cfg.target, &batch)?;
                batch = fitted.apply(&batch)?;
                fitted_steps.push(fitted);
            }
            (None, Step::Filter(f)) => {
                batch = steps::filter(f, &batch)?;
            }
            (None, Step::Split(s)) => {
                transform_input = Some(columns_from_schema(batch.schema().as_ref())?);
                splits = Some(split::split_batch(&batch, *s)?);
            }
            (Some(part), Step::Filter(f)) => {
                part.train = steps::filter(f, &part.train)?;
                part.test = steps::filter(f, &part.test)?;
                if let Some(val) = part.validation.take() {
                    part.validation = Some(steps::filter(f, &val)?);
                }
            }
            (Some(part), Step::Select(_) | Step::Impute(_) | Step::Scale(_) | Step::Encode(_)) => {
                let fitted = steps::fit(step, &cfg.target, &part.train)?;
                part.train = fitted.apply(&part.train)?;
                part.test = fitted.apply(&part.test)?;
                if let Some(val) = part.validation.take() {
                    part.validation = Some(fitted.apply(&val)?);
                }
                fitted_steps.push(fitted);
            }
            (None, s) => {
                return Err(Error::Config(format!(
                    "`{}` must come after `split`",
                    s.name()
                )));
            }
            (Some(_), Step::Split(_)) => {
                return Err(Error::Config("`split` must appear exactly once".into()));
            }
        }
    }

    let splits = splits.ok_or_else(|| Error::Config("steps: a `split` step is required".into()))?;
    let transform_input =
        transform_input.ok_or_else(|| Error::Config("steps: a `split` step is required".into()))?;

    let output_cols = columns_from_schema(splits.train.schema().as_ref())?;
    let mut features = Vec::with_capacity(output_cols.len().saturating_sub(1));
    let mut target_col = None;
    for col in &output_cols {
        if col.name == cfg.target {
            target_col = Some(col.clone());
        } else {
            features.push(col.clone());
        }
    }
    let target_col = target_col.ok_or_else(|| {
        Error::Runtime(format!(
            "target column `{}` missing after preprocessing",
            cfg.target
        ))
    })?;

    let features_io = json!({
        "features": features,
        "target": target_col,
    });

    if cancel.is_cancelled() {
        return Err(Error::Cancelled);
    }
    let train_path = write_partition(&workdir, TRAIN, splits.train, &in_file, &cancel)?;
    let test_path = write_partition(&workdir, TEST, splits.test, &in_file, &cancel)?;
    let val_path = match splits.validation {
        Some(val) => Some(write_partition(
            &workdir, VALIDATION, val, &in_file, &cancel,
        )?),
        None => None,
    };

    if cancel.is_cancelled() {
        return Err(Error::Cancelled);
    }
    let transform_dir = workdir.join(TRANSFORM);
    std::fs::create_dir_all(&transform_dir)
        .map_err(|e| Error::Runtime(format!("{}: {e}", transform_dir.display())))?;
    let transform_path = transform_dir.join(TRANSFORM_FILE);
    let transform = FittedTransform::new(cfg.target.clone(), transform_input.clone(), fitted_steps);
    let transform_bytes = transform.to_json()?;
    std::fs::write(&transform_path, transform_bytes)
        .map_err(|e| Error::Runtime(format!("{}: {e}", transform_path.display())))?;

    let transform_manifest = Manifest::new(ArtifactKind::Transform, "json").with_io_schema(json!({
        "input": transform_input,
        "output": output_cols,
    }));

    let mut drafts = OutputDrafts::new()
        .with(
            TRAIN,
            OutputDraft::new(
                features_manifest(&cfg.target, &features_io),
                DraftPayload::File(train_path),
            ),
        )
        .with(
            TEST,
            OutputDraft::new(
                features_manifest(&cfg.target, &features_io),
                DraftPayload::File(test_path),
            ),
        )
        .with(
            TRANSFORM,
            OutputDraft::new(transform_manifest, DraftPayload::File(transform_path)),
        );

    if let Some(path) = val_path {
        drafts = drafts.with(
            VALIDATION,
            OutputDraft::new(
                features_manifest(&cfg.target, &features_io),
                DraftPayload::File(path),
            ),
        );
    }

    Ok(drafts)
}

impl Block for PreprocessTabular {
    fn spec(&self) -> &BlockSpec {
        &self.0
    }

    fn validate_config(&self, config: &serde_json::Value) -> Result<(), Error> {
        PreprocessConfig::parse(config).map(|_| ())
    }

    fn resolve_ports(&self, config: &serde_json::Value) -> Result<ResolvedPorts, Error> {
        let cfg = PreprocessConfig::parse(config)?;
        let mut ports = ResolvedPorts::from(self.spec());
        if cfg.split().and_then(|s| s.validation).is_some() {
            ports.outputs.push(features_port(VALIDATION));
        }
        Ok(ports)
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
            let cfg = PreprocessConfig::parse(ctx.config())?;
            let data = inputs.get(INPUT)?;
            let in_dir = ctx.workdir().join("in");
            ctx.store().fetch(&data.id(), &in_dir).await?;
            if ctx.cancel().is_cancelled() {
                return Err(Error::Cancelled);
            }
            let in_file = in_dir.join(DATA_FILE);
            let workdir = ctx.workdir().to_path_buf();
            let cancel = ctx.cancel().clone();
            let drafts = blocking(move || execute_blocking(cfg, in_file, workdir, cancel)).await?;
            Ok(BlockOutcome::Produced(drafts))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Float64Array, Int32Array, Int64Array, StringArray};
    use mallops_core::{
        Artifact, ArtifactId, ArtifactStore, CachePolicy, CancelSource, MemorySecretProvider,
        MemoryStore, PayloadFile, PayloadSource, RelPath, RunId,
    };
    use serde_json::json;

    use super::*;

    fn cfg(steps: serde_json::Value) -> Result<PreprocessConfig, Error> {
        PreprocessConfig::parse(&json!({ "target": "y", "steps": steps }))
    }

    fn config_err(steps: serde_json::Value) -> String {
        match cfg(steps) {
            Err(Error::Config(m)) => m,
            other => panic!("expected Error::Config, got {other:?}"),
        }
    }

    fn sample_batch(n: usize) -> RecordBatch {
        let ids: Vec<i64> = (0..n as i64).collect();
        let scores: Vec<f64> = (0..n).map(|i| (i as f64) * 1.5).collect();
        let cats: Vec<String> = (0..n).map(|i| format!("c{}", i % 3)).collect();
        let targets: Vec<i64> = (0..n as i64).map(|i| i % 2).collect();
        RecordBatch::try_from_iter([
            ("id", Arc::new(Int64Array::from(ids)) as ArrayRef),
            ("score", Arc::new(Float64Array::from(scores)) as _),
            ("cat", Arc::new(StringArray::from(cats)) as _),
            ("y", Arc::new(Int64Array::from(targets)) as _),
        ])
        .unwrap()
    }

    async fn store_dataset(store: &Arc<MemoryStore>, batch: &RecordBatch) -> Inputs {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(DATA_FILE);
        let cancel = CancelSource::new().token();
        let iter = RecordBatchIterator::new([Ok(batch.clone())], batch.schema());
        write_parquet(iter, &path, &path, &cancel).unwrap();

        let manifest = Manifest::new(ArtifactKind::Dataset, "parquet");
        let id = store
            .put(manifest.clone(), PayloadSource::File(&path))
            .await
            .unwrap();
        Inputs::new().with(INPUT, Artifact::new(id, manifest))
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

    fn read_ids(draft: &OutputDraft) -> Vec<i64> {
        let DraftPayload::File(path) = &draft.payload else {
            panic!("expected DraftPayload::File");
        };
        let cancel = CancelSource::new().token();
        let batch = read_parquet_batch(path, &cancel).unwrap();
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        col.values().to_vec()
    }

    async fn run_block(
        store: &Arc<MemoryStore>,
        inputs: &Inputs,
        config: serde_json::Value,
    ) -> (tempfile::TempDir, OutputDrafts) {
        let workdir = tempfile::tempdir().unwrap();
        let (ctx, _cancel) = make_ctx(workdir.path().to_path_buf(), Arc::clone(store), config);
        let BlockOutcome::Produced(drafts) = PreprocessTabular::new()
            .run(&ctx, inputs.clone())
            .await
            .unwrap()
        else {
            panic!("expected Produced");
        };
        (workdir, drafts)
    }

    #[test]
    fn preprocess_steps_config_parses_every_step() {
        let c = cfg(json!([
            { "select": { "columns": ["a", "b"] } },
            { "filter": { "column": "a", "op": "gt", "value": 0 } },
            { "filter": { "column": "b", "op": "not_null" } },
            { "split": { "test": 0.2, "validation": 0.1, "seed": 42 } },
            { "impute": { "strategy": "median" } },
            { "impute": { "strategy": "constant", "value": "n/a", "columns": ["b"] } },
            { "scale": { "method": "minmax" } },
            { "encode": { "method": "one_hot", "columns": ["b"] } },
        ]))
        .unwrap();
        assert_eq!(c.steps.len(), 8);
        assert_eq!(
            c.split(),
            Some(SplitStep {
                test: 0.2,
                validation: Some(0.1),
                seed: 42
            })
        );
    }

    #[test]
    fn preprocess_steps_config_rejects_invalid() {
        let split = json!({ "split": { "test": 0.2, "seed": 1 } });
        for (steps, needle) in [
            (json!([]), "`split` step is required"),
            (json!([split, split]), "exactly once"),
            (
                json!([{ "scale": { "method": "standard" } }, split]),
                "after `split`",
            ),
            (
                json!([{ "split": { "test": 1.0, "seed": 1 } }]),
                "split.test",
            ),
            (
                json!([{ "split": { "test": 0.5, "validation": 0.5, "seed": 1 } }]),
                "split.validation",
            ),
            (
                json!([split, { "impute": { "strategy": "constant" } }]),
                "required with strategy constant",
            ),
            (
                json!([split, { "impute": { "strategy": "mean", "value": 1 } }]),
                "only allowed",
            ),
            (
                json!([{ "filter": { "column": "a", "op": "eq" } }, split]),
                "filter.value is required",
            ),
            (
                json!([{ "filter": { "column": "a", "op": "is_null", "value": 1 } }, split]),
                "not allowed",
            ),
            (
                json!([{ "select": { "columns": [] } }, split]),
                "must not be empty",
            ),
            (
                json!([split, { "scale": { "method": "robust" } }]),
                "unknown variant",
            ),
            (
                json!([split, { "encode": { "method": "ordinal", "extra": 1 } }]),
                "unknown field",
            ),
        ] {
            let msg = config_err(steps.clone());
            assert!(msg.contains(needle), "{steps}: {msg}");
        }
        assert!(matches!(
            PreprocessConfig::parse(&json!({ "steps": [split] })),
            Err(Error::Config(_))
        ));
    }

    #[test]
    fn preprocess_validation_split_port_only_when_configured() {
        let block = PreprocessTabular::new();
        let outputs = |split: serde_json::Value| -> Vec<String> {
            let c = json!({ "target": "y", "steps": [{ "split": split }] });
            block
                .resolve_ports(&c)
                .unwrap()
                .outputs
                .iter()
                .map(|p| p.name.to_string())
                .collect()
        };
        assert_eq!(
            outputs(json!({ "test": 0.2, "seed": 1 })),
            [TRAIN, TEST, TRANSFORM]
        );
        assert_eq!(
            outputs(json!({ "test": 0.2, "validation": 0.1, "seed": 1 })),
            [TRAIN, TEST, TRANSFORM, VALIDATION]
        );
    }

    #[tokio::test]
    async fn preprocess_deterministic_seed() {
        let store = Arc::new(MemoryStore::new());
        let inputs = store_dataset(&store, &sample_batch(20)).await;

        let cfg_seed_42 = json!({
            "target": "y",
            "steps": [{ "split": { "test": 0.2, "seed": 42 } }]
        });
        let cfg_seed_99 = json!({
            "target": "y",
            "steps": [{ "split": { "test": 0.2, "seed": 99 } }]
        });

        let (_w1, d1) = run_block(&store, &inputs, cfg_seed_42.clone()).await;
        let (_w2, d2) = run_block(&store, &inputs, cfg_seed_42).await;
        let (_w3, d3) = run_block(&store, &inputs, cfg_seed_99).await;

        for port in [TRAIN, TEST, TRANSFORM] {
            assert_eq!(
                draft_artifact_id(d1.get(port).unwrap()),
                draft_artifact_id(d2.get(port).unwrap()),
                "same seed must produce identical ArtifactId for {port}"
            );
        }
        assert_ne!(
            draft_artifact_id(d1.get(TRAIN).unwrap()),
            draft_artifact_id(d3.get(TRAIN).unwrap()),
            "different seed must produce different train split"
        );
        assert_ne!(
            draft_artifact_id(d1.get(TEST).unwrap()),
            draft_artifact_id(d3.get(TEST).unwrap()),
            "different seed must produce different test split"
        );

        // With validation configured as well.
        let cfg_val_42 = json!({
            "target": "y",
            "steps": [{ "split": { "test": 0.2, "validation": 0.2, "seed": 42 } }]
        });
        let cfg_val_99 = json!({
            "target": "y",
            "steps": [{ "split": { "test": 0.2, "validation": 0.2, "seed": 99 } }]
        });
        let (_wv1, dv1) = run_block(&store, &inputs, cfg_val_42.clone()).await;
        let (_wv2, dv2) = run_block(&store, &inputs, cfg_val_42).await;
        let (_wv3, dv3) = run_block(&store, &inputs, cfg_val_99).await;

        for port in [TRAIN, VALIDATION, TEST, TRANSFORM] {
            assert_eq!(
                draft_artifact_id(dv1.get(port).unwrap()),
                draft_artifact_id(dv2.get(port).unwrap()),
                "same seed with validation must produce identical ArtifactId for {port}"
            );
        }
        assert_ne!(
            draft_artifact_id(dv1.get(VALIDATION).unwrap()),
            draft_artifact_id(dv3.get(VALIDATION).unwrap()),
            "different seed must produce different validation split"
        );
    }

    #[tokio::test]
    async fn preprocess_manifest_metadata() {
        let store = Arc::new(MemoryStore::new());
        let inputs = store_dataset(&store, &sample_batch(10)).await;
        let config = json!({
            "target": "y",
            "steps": [{ "split": { "test": 0.2, "validation": 0.2, "seed": 7 } }]
        });

        let block = PreprocessTabular::new();
        let ports = block.resolve_ports(&config).unwrap();
        let (_w, drafts) = run_block(&store, &inputs, config).await;

        let expected_features_io = json!({
            "features": [
                { "name": "id", "type": "int64" },
                { "name": "score", "type": "float64" },
                { "name": "cat", "type": "string" }
            ],
            "target": { "name": "y", "type": "int64" }
        });
        for port in [TRAIN, VALIDATION, TEST] {
            let draft = drafts.get(port).expect("partition draft");
            assert_eq!(draft.manifest.kind, ArtifactKind::Features);
            assert_eq!(draft.manifest.format, "parquet");
            assert_eq!(draft.manifest.metadata.get("target"), Some(&json!("y")));
            assert_eq!(draft.manifest.io_schema, Some(expected_features_io.clone()));
            let p = ports.outputs.iter().find(|p| p.name == port).unwrap();
            assert!(p.accepts_manifest(&draft.manifest));
        }

        let transform = drafts.get(TRANSFORM).expect("transform draft");
        assert_eq!(transform.manifest.kind, ArtifactKind::Transform);
        assert_eq!(transform.manifest.format, "json");
        let expected_cols = json!([
            { "name": "id", "type": "int64" },
            { "name": "score", "type": "float64" },
            { "name": "cat", "type": "string" },
            { "name": "y", "type": "int64" }
        ]);
        assert_eq!(
            transform.manifest.io_schema,
            Some(json!({
                "input": expected_cols,
                "output": expected_cols
            }))
        );
        let t_port = ports.outputs.iter().find(|p| p.name == TRANSFORM).unwrap();
        assert!(t_port.accepts_manifest(&transform.manifest));

        let DraftPayload::File(t_path) = &transform.payload else {
            panic!("expected DraftPayload::File");
        };
        let t_json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(t_path).unwrap()).unwrap();
        assert_eq!(t_json["v"], json!(1));
        assert_eq!(t_json["target"], json!("y"));
        assert_eq!(t_json["input"], expected_cols);
        assert_eq!(t_json["steps"], json!([]));
    }

    #[tokio::test]
    async fn preprocess_validation_split() {
        let store = Arc::new(MemoryStore::new());
        let n = 20usize;
        let inputs = store_dataset(&store, &sample_batch(n)).await;
        let all_ids: BTreeSet<i64> = (0..n as i64).collect();

        // Without validation: validation port absent, train+test disjoint and complete.
        let cfg_no_val = json!({
            "target": "y",
            "steps": [{ "split": { "test": 0.2, "seed": 42 } }]
        });
        let (_w1, d_no_val) = run_block(&store, &inputs, cfg_no_val).await;
        assert!(d_no_val.get(VALIDATION).is_none());

        let train_no_val = read_ids(d_no_val.get(TRAIN).unwrap());
        let test_no_val = read_ids(d_no_val.get(TEST).unwrap());
        assert_eq!(test_no_val.len(), 4);
        assert_eq!(train_no_val.len(), 16);
        assert!(train_no_val.windows(2).all(|w| w[0] < w[1]));
        assert!(test_no_val.windows(2).all(|w| w[0] < w[1]));
        let union_no_val: BTreeSet<i64> = train_no_val
            .iter()
            .chain(test_no_val.iter())
            .copied()
            .collect();
        assert_eq!(union_no_val, all_ids);

        // With validation and the same seed: test is unchanged, all three partitions are disjoint
        // and sorted by original index, covering every row.
        let cfg_with_val = json!({
            "target": "y",
            "steps": [{ "split": { "test": 0.2, "validation": 0.25, "seed": 42 } }]
        });
        let (_w2, d_with_val) = run_block(&store, &inputs, cfg_with_val).await;
        let train_v = read_ids(d_with_val.get(TRAIN).unwrap());
        let val_v = read_ids(d_with_val.get(VALIDATION).unwrap());
        let test_v = read_ids(d_with_val.get(TEST).unwrap());

        assert_eq!(test_v, test_no_val);
        assert_eq!(
            draft_artifact_id(d_with_val.get(TEST).unwrap()),
            draft_artifact_id(d_no_val.get(TEST).unwrap())
        );
        assert_eq!(val_v.len(), 5);
        assert_eq!(train_v.len(), 11);
        assert!(train_v.windows(2).all(|w| w[0] < w[1]));
        assert!(val_v.windows(2).all(|w| w[0] < w[1]));
        assert!(test_v.windows(2).all(|w| w[0] < w[1]));

        let mut union_v = BTreeSet::new();
        for id in train_v.iter().chain(val_v.iter()).chain(test_v.iter()) {
            assert!(union_v.insert(*id), "duplicate row id {id}");
        }
        assert_eq!(union_v, all_ids);

        // Empty partition errors.
        let small_inputs = store_dataset(&store, &sample_batch(3)).await;
        let block = PreprocessTabular::new();
        for (data, split_cfg, needle) in [
            (
                small_inputs.clone(),
                json!({ "test": 0.2, "seed": 1 }),
                "empty `test`",
            ),
            (
                small_inputs,
                json!({ "test": 0.4, "validation": 0.2, "seed": 1 }),
                "empty `validation`",
            ),
            (
                store_dataset(&store, &sample_batch(0)).await,
                json!({ "test": 0.2, "seed": 1 }),
                "empty dataset",
            ),
        ] {
            let workdir = tempfile::tempdir().unwrap();
            let (ctx, _c) = make_ctx(
                workdir.path().to_path_buf(),
                Arc::clone(&store),
                json!({ "target": "y", "steps": [{ "split": split_cfg }] }),
            );
            let err = block.run(&ctx, data).await.unwrap_err();
            assert!(
                matches!(&err, Error::Runtime(m) if m.contains(needle)),
                "expected Runtime containing {needle:?}, got {err:?}"
            );
        }
    }

    #[tokio::test]
    async fn preprocess_run_errors_cancellation_and_cache_policy() {
        let store = Arc::new(MemoryStore::new());
        let block = PreprocessTabular::new();

        // Default cache policy: Deterministic without extra_hash.
        let workdir = tempfile::tempdir().unwrap();
        let cfg = json!({ "target": "y", "steps": [{ "split": { "test": 0.2, "seed": 1 } }] });
        let (ctx, cancel) = make_ctx(
            workdir.path().to_path_buf(),
            Arc::clone(&store),
            cfg.clone(),
        );
        assert_eq!(
            block.cache_policy(&ctx).await.unwrap(),
            CachePolicy::Deterministic { extra_hash: None }
        );

        // Cancellation before run returns Error::Cancelled.
        let inputs = store_dataset(&store, &sample_batch(10)).await;
        cancel.cancel();
        let err = block.run(&ctx, inputs.clone()).await.unwrap_err();
        assert!(matches!(err, Error::Cancelled), "got {err:?}");

        // Missing target column.
        let workdir2 = tempfile::tempdir().unwrap();
        let (ctx2, _c2) = make_ctx(
            workdir2.path().to_path_buf(),
            Arc::clone(&store),
            json!({ "target": "missing", "steps": [{ "split": { "test": 0.2, "seed": 1 } }] }),
        );
        let err = block.run(&ctx2, inputs).await.unwrap_err();
        assert!(
            matches!(&err, Error::Runtime(m) if m.contains("missing")),
            "got {err:?}"
        );

        // Unsupported Arrow type suggests `select`.
        let bad_batch = RecordBatch::try_from_iter([
            (
                "i32col",
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])) as ArrayRef,
            ),
            ("y", Arc::new(Int64Array::from(vec![0, 1, 0, 1, 0])) as _),
        ])
        .unwrap();
        let bad_inputs = store_dataset(&store, &bad_batch).await;
        let workdir3 = tempfile::tempdir().unwrap();
        let (ctx3, _c3) = make_ctx(workdir3.path().to_path_buf(), Arc::clone(&store), cfg);
        let err = block.run(&ctx3, bad_inputs).await.unwrap_err();
        assert!(
            matches!(&err, Error::Runtime(m) if m.contains("select") && m.contains("i32col")),
            "got {err:?}"
        );
    }

    #[tokio::test]
    async fn preprocess_scale_fits_on_train_only() {
        let store = Arc::new(MemoryStore::new());
        // Skewed values so train-only mean differs from full-dataset mean.
        let x_vals: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1000.0];
        let y_vals: Vec<i64> = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
        let batch = RecordBatch::try_from_iter([
            ("x", Arc::new(Float64Array::from(x_vals)) as ArrayRef),
            ("y", Arc::new(Int64Array::from(y_vals)) as _),
        ])
        .unwrap();
        let inputs = store_dataset(&store, &batch).await;

        // Raw split with the same seed gives the exact train values used for fitting.
        let (_w_raw, d_raw) = run_block(
            &store,
            &inputs,
            json!({
                "target": "y",
                "steps": [{ "split": { "test": 0.2, "seed": 42 } }]
            }),
        )
        .await;
        let DraftPayload::File(raw_train_path) = &d_raw.get(TRAIN).unwrap().payload else {
            panic!("expected file");
        };
        let raw_train = read_parquet_batch(raw_train_path, &CancelSource::new().token()).unwrap();
        let raw_x = raw_train
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let train_vals: Vec<f64> = raw_x.values().to_vec();
        let mean = train_vals.iter().sum::<f64>() / train_vals.len() as f64;
        let var =
            train_vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / train_vals.len() as f64;
        let std = var.sqrt();

        // Now run with standard scaling after split.
        let (_w_scaled, d_scaled) = run_block(
            &store,
            &inputs,
            json!({
                "target": "y",
                "steps": [
                    { "split": { "test": 0.2, "seed": 42 } },
                    { "scale": { "method": "standard" } }
                ]
            }),
        )
        .await;

        let DraftPayload::File(t_path) = &d_scaled.get(TRANSFORM).unwrap().payload else {
            panic!("expected file");
        };
        let t_json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(t_path).unwrap()).unwrap();
        let params = &t_json["steps"][0]["scale"]["params"]["x"];
        let offset = params["offset"].as_f64().unwrap();
        let scale = params["scale"].as_f64().unwrap();
        assert!((offset - mean).abs() < 1e-9, "offset={offset}, mean={mean}");
        assert!((scale - std).abs() < 1e-9, "scale={scale}, std={std}");
    }
}
