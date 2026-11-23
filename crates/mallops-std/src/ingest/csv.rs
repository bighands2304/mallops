//! `std/ingest.csv`: reads a local CSV file into a Parquet `dataset`.

use std::fs::File;
use std::io::Seek;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_csv::ReaderBuilder;
use arrow_csv::reader::Format;
use arrow_schema::Schema;
use mallops_core::{
    Block, BlockOutcome, BlockSpec, BoxFuture, CachePolicy, Context, Error, Inputs, Stage,
};
use serde::Deserialize;

use super::{
    Column, DATA_FILE, arrow_schema, blocking, dataset_outcome, dataset_port, parse_config,
    source_policy, write_parquet,
};

fn default_delimiter() -> char {
    ','
}

fn default_has_header() -> bool {
    true
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CsvConfig {
    path: PathBuf,
    #[serde(default = "default_delimiter")]
    delimiter: char,
    #[serde(default = "default_has_header")]
    has_header: bool,
    schema: Option<Vec<Column>>,
}

impl CsvConfig {
    fn parse(config: &serde_json::Value) -> Result<Self, Error> {
        let cfg: Self = parse_config(config)?;
        if !cfg.delimiter.is_ascii() {
            return Err(Error::Config(format!(
                "delimiter {:?} must be an ASCII character",
                cfg.delimiter
            )));
        }
        Ok(cfg)
    }

    fn delimiter_byte(&self) -> u8 {
        self.delimiter as u8
    }
}

fn check_header_names(expected: &[Column], header: &Schema) -> Result<(), Error> {
    let found: Vec<&str> = header.fields().iter().map(|f| f.name().as_str()).collect();
    let want: Vec<&str> = expected.iter().map(|c| c.name.as_str()).collect();
    if found == want {
        return Ok(());
    }
    Err(Error::Runtime(format!(
        "schema mismatch: expected [{}], found [{}]",
        want.join(", "),
        found.join(", "),
    )))
}

fn read_and_write_csv(cfg: &CsvConfig, out: &Path) -> Result<(), Error> {
    let read_err = |e: &dyn std::fmt::Display| {
        Error::Runtime(format!("cannot read {}: {e}", cfg.path.display()))
    };
    let mut file = File::open(&cfg.path).map_err(|e| read_err(&e))?;
    let format = Format::default()
        .with_header(cfg.has_header)
        .with_delimiter(cfg.delimiter_byte());

    let schema = match &cfg.schema {
        Some(expected) => {
            if cfg.has_header {
                let (header, _) = format
                    .infer_schema(&mut file, Some(0))
                    .map_err(|e| read_err(&e))?;
                check_header_names(expected, &header)?;
                file.rewind().map_err(|e| read_err(&e))?;
            }
            arrow_schema(expected)
        }
        None => {
            let (inferred, _) = format
                .infer_schema(&mut file, None)
                .map_err(|e| read_err(&e))?;
            file.rewind().map_err(|e| read_err(&e))?;
            inferred
        }
    };

    let reader = ReaderBuilder::new(Arc::new(schema))
        .with_format(format)
        .build(file)
        .map_err(|e| read_err(&e))?;
    write_parquet(reader, out)?;
    Ok(())
}

/// `std/ingest.csv` block.
pub(crate) struct IngestCsv(BlockSpec);

impl IngestCsv {
    pub(crate) fn new() -> Self {
        Self(BlockSpec::new("std/ingest.csv", "0.1.0", Stage::Ingest).with_output(dataset_port()))
    }
}

impl Block for IngestCsv {
    fn spec(&self) -> &BlockSpec {
        &self.0
    }

    fn validate_config(&self, config: &serde_json::Value) -> Result<(), Error> {
        CsvConfig::parse(config).map(|_| ())
    }

    fn cache_policy<'a>(&'a self, ctx: &'a Context) -> BoxFuture<'a, Result<CachePolicy, Error>> {
        Box::pin(async move {
            let cfg = CsvConfig::parse(ctx.config())?;
            source_policy(cfg.path).await
        })
    }

    fn run<'a>(
        &'a self,
        ctx: &'a Context,
        _inputs: Inputs,
    ) -> BoxFuture<'a, Result<BlockOutcome, Error>> {
        Box::pin(async move {
            if ctx.cancel().is_cancelled() {
                return Err(Error::Cancelled);
            }
            let cfg = CsvConfig::parse(ctx.config())?;
            let out = ctx.workdir().join(DATA_FILE);
            let out_path = out.clone();
            blocking(move || read_and_write_csv(&cfg, &out_path)).await?;
            Ok(dataset_outcome(out))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

    use ::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use arrow_array::{
        Array, BooleanArray, Date32Array, Float64Array, Int64Array, RecordBatch, RecordBatchReader,
        StringArray,
    };
    use arrow_schema::DataType;
    use mallops_core::{
        ArtifactKind, CancelSource, DraftPayload, MemorySecretProvider, MemoryStore, RunId,
    };
    use serde_json::json;

    use super::*;
    use crate::ingest::OUTPUT;

    fn make_ctx(workdir: &Path, config: serde_json::Value) -> (Context, CancelSource) {
        let cancel = CancelSource::new();
        let ctx = Context::new(
            RunId::new(),
            workdir.to_path_buf(),
            Arc::new(MemoryStore::new()),
            config,
            cancel.token(),
            Arc::new(MemorySecretProvider::default()),
        );
        (ctx, cancel)
    }

    fn read_parquet_batches(path: &Path) -> (Arc<Schema>, Vec<RecordBatch>) {
        let file = File::open(path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        let schema = reader.schema();
        let batches: Vec<RecordBatch> = reader.map(|b| b.unwrap()).collect();
        (schema, batches)
    }

    fn write_csv(dir: &Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn ingest_csv_rejects_unknown_config_and_non_ascii_delimiter() {
        let block = IngestCsv::new();
        assert!(block.validate_config(&json!({ "path": "a.csv" })).is_ok());
        assert!(
            block
                .validate_config(&json!({
                    "path": "a.csv",
                    "delimiter": ";",
                    "has_header": false,
                    "schema": [{ "name": "x", "type": "int64" }]
                }))
                .is_ok()
        );

        for bad in [
            json!({ "path": "a.csv", "unknown_field": true }),
            json!({}),
            json!({ "path": "a.csv", "delimiter": "€" }),
            json!({ "path": "a.csv", "delimiter": "é" }),
            json!({ "path": "a.csv", "delimiter": ";;" }),
            json!({ "path": "a.csv", "schema": [{ "name": "x", "type": "int64", "extra": 1 }] }),
        ] {
            let err = block.validate_config(&bad).unwrap_err();
            assert!(
                matches!(err, Error::Config(_)),
                "expected Config for {bad}: {err}"
            );
        }
    }

    #[tokio::test]
    async fn ingest_csv_custom_semicolon_delimiter() {
        let dir = tempfile::tempdir().unwrap();
        let csv = write_csv(
            dir.path(),
            "semi.csv",
            "id;name;score\n1;alice;9.5\n2;bob;8.0\n",
        );
        let workdir = tempfile::tempdir().unwrap();
        let (ctx, _cancel) = make_ctx(workdir.path(), json!({ "path": csv, "delimiter": ";" }));

        let outcome = IngestCsv::new().run(&ctx, Inputs::new()).await.unwrap();
        let BlockOutcome::Produced(drafts) = outcome else {
            panic!("expected Produced");
        };
        let DraftPayload::File(out_path) = &drafts.get(OUTPUT).unwrap().payload else {
            panic!("expected File payload");
        };

        let (schema, batches) = read_parquet_batches(out_path);
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec!["id", "name", "score"]);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 2);

        let ids = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let labels = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let scores = batches[0]
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(ids.values(), &[1, 2]);
        assert_eq!(labels.value(0), "alice");
        assert_eq!(labels.value(1), "bob");
        assert_eq!(scores.values(), &[9.5, 8.0]);
    }

    #[tokio::test]
    async fn ingest_csv_without_header() {
        let dir = tempfile::tempdir().unwrap();
        let csv = write_csv(dir.path(), "nohdr.csv", "1,alice,2.5\n2,bob,3.5\n");

        // Inferred schema without header.
        let workdir = tempfile::tempdir().unwrap();
        let (ctx, _cancel) = make_ctx(workdir.path(), json!({ "path": csv, "has_header": false }));
        IngestCsv::new().run(&ctx, Inputs::new()).await.unwrap();
        let (schema, batches) = read_parquet_batches(&workdir.path().join(DATA_FILE));
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec!["column_1", "column_2", "column_3"]);
        assert_eq!(schema.field(0).data_type(), &DataType::Int64);
        assert_eq!(schema.field(1).data_type(), &DataType::Utf8);
        assert_eq!(schema.field(2).data_type(), &DataType::Float64);
        let ids = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(ids.values(), &[1, 2]);

        // Explicit schema without header assigns column names and keeps row 1 as data.
        let workdir2 = tempfile::tempdir().unwrap();
        let (ctx2, _cancel2) = make_ctx(
            workdir2.path(),
            json!({
                "path": csv,
                "has_header": false,
                "schema": [
                    { "name": "id", "type": "int64" },
                    { "name": "name", "type": "string" },
                    { "name": "val", "type": "float64" }
                ]
            }),
        );
        IngestCsv::new().run(&ctx2, Inputs::new()).await.unwrap();
        let (schema2, batches2) = read_parquet_batches(&workdir2.path().join(DATA_FILE));
        let names2: Vec<&str> = schema2.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names2, vec!["id", "name", "val"]);
        let ids2 = batches2[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(ids2.values(), &[1, 2]);
    }

    #[tokio::test]
    async fn ingest_csv_infers_types_across_entire_file() {
        let dir = tempfile::tempdir().unwrap();
        // >100 rows so inference cannot rely on a small default prefix sample.
        let mut body = String::from("id,metric,label\n");
        for i in 1..=150 {
            let _ = writeln!(body, "{i},{i},{i}");
        }
        // Row 151 widens `metric` to Float64 and `label` to Utf8.
        body.push_str("151,151.5,tail\n");
        let csv = write_csv(dir.path(), "infer.csv", &body);

        let workdir = tempfile::tempdir().unwrap();
        let (ctx, _cancel) = make_ctx(workdir.path(), json!({ "path": csv }));
        IngestCsv::new().run(&ctx, Inputs::new()).await.unwrap();

        let (schema, batches) = read_parquet_batches(&workdir.path().join(DATA_FILE));
        assert_eq!(schema.field(0).data_type(), &DataType::Int64);
        assert_eq!(schema.field(1).data_type(), &DataType::Float64);
        assert_eq!(schema.field(2).data_type(), &DataType::Utf8);

        let total_rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(total_rows, 151);
    }

    #[tokio::test]
    async fn ingest_csv_schema_ok() {
        let dir = tempfile::tempdir().unwrap();
        // `"42"` parsed as string and `"2"` parsed as float64 confirm explicit schema rules.
        let csv = write_csv(
            dir.path(),
            "typed.csv",
            "flag,id,val,code,day\ntrue,1,2,42,2026-01-15\nfalse,2,3.5,99,2026-01-16\n",
        );
        let workdir = tempfile::tempdir().unwrap();
        let (ctx, _cancel) = make_ctx(
            workdir.path(),
            json!({
                "path": csv,
                "schema": [
                    { "name": "flag", "type": "bool" },
                    { "name": "id", "type": "int64" },
                    { "name": "val", "type": "float64" },
                    { "name": "code", "type": "string" },
                    { "name": "day", "type": "date32" }
                ]
            }),
        );

        IngestCsv::new().run(&ctx, Inputs::new()).await.unwrap();
        let (schema, batches) = read_parquet_batches(&workdir.path().join(DATA_FILE));
        let types: Vec<&DataType> = schema.fields().iter().map(|f| f.data_type()).collect();
        assert_eq!(
            types,
            vec![
                &DataType::Boolean,
                &DataType::Int64,
                &DataType::Float64,
                &DataType::Utf8,
                &DataType::Date32,
            ]
        );
        assert_eq!(batches.len(), 1);
        let flags = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        let vals = batches[0]
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let codes = batches[0]
            .column(3)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let days = batches[0]
            .column(4)
            .as_any()
            .downcast_ref::<Date32Array>()
            .unwrap();
        assert!(flags.value(0) && !flags.value(1));
        assert_eq!(vals.values(), &[2.0, 3.5]);
        assert_eq!(codes.value(0), "42");
        assert_eq!(days.len(), 2);
    }

    #[tokio::test]
    async fn ingest_csv_schema_ko_names_and_types() {
        let dir = tempfile::tempdir().unwrap();
        let csv = write_csv(dir.path(), "data.csv", "id,score\n1,3.5\n");
        let block = IngestCsv::new();

        // Wrong column name, wrong order, extra/missing column -> "schema mismatch".
        for bad_schema in [
            json!([{ "name": "id", "type": "int64" }, { "name": "other", "type": "float64" }]),
            json!([{ "name": "score", "type": "float64" }, { "name": "id", "type": "int64" }]),
            json!([{ "name": "id", "type": "int64" }]),
            json!([
                { "name": "id", "type": "int64" },
                { "name": "score", "type": "float64" },
                { "name": "extra", "type": "string" }
            ]),
        ] {
            let workdir = tempfile::tempdir().unwrap();
            let (ctx, _cancel) =
                make_ctx(workdir.path(), json!({ "path": csv, "schema": bad_schema }));
            let err = block
                .run(&ctx, Inputs::new())
                .await
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("schema mismatch"),
                "expected schema mismatch, got: {err}"
            );
        }

        // Incompatible column type (float value "3.5" declared as int64).
        let workdir = tempfile::tempdir().unwrap();
        let (ctx, _cancel) = make_ctx(
            workdir.path(),
            json!({
                "path": csv,
                "schema": [
                    { "name": "id", "type": "int64" },
                    { "name": "score", "type": "int64" }
                ]
            }),
        );
        let err = block.run(&ctx, Inputs::new()).await.unwrap_err();
        assert!(
            matches!(err, Error::Runtime(_)),
            "expected Runtime error, got: {err}"
        );
    }

    #[tokio::test]
    async fn ingest_csv_rejects_unconvertible_value() {
        let dir = tempfile::tempdir().unwrap();
        let csv = write_csv(
            dir.path(),
            "badval.csv",
            "id,day\n1,2026-01-01\nnope,2026-01-02\n",
        );
        let workdir = tempfile::tempdir().unwrap();
        let (ctx, _cancel) = make_ctx(
            workdir.path(),
            json!({
                "path": csv,
                "schema": [
                    { "name": "id", "type": "int64" },
                    { "name": "day", "type": "date32" }
                ]
            }),
        );

        let err = IngestCsv::new().run(&ctx, Inputs::new()).await.unwrap_err();
        assert!(
            matches!(err, Error::Runtime(_)),
            "expected Runtime error, got: {err}"
        );
    }

    #[tokio::test]
    async fn ingest_csv_manifest_dataset_parquet() {
        let dir = tempfile::tempdir().unwrap();
        let csv = write_csv(dir.path(), "m.csv", "x\n1\n2\n");
        let workdir = tempfile::tempdir().unwrap();
        let (ctx, _cancel) = make_ctx(workdir.path(), json!({ "path": csv }));

        let block = IngestCsv::new();
        let BlockOutcome::Produced(drafts) = block.run(&ctx, Inputs::new()).await.unwrap() else {
            panic!("expected Produced");
        };
        let draft = drafts.get(OUTPUT).expect("dataset draft");
        assert_eq!(draft.manifest.kind, ArtifactKind::Dataset);
        assert_eq!(draft.manifest.format, "parquet");
        assert!(block.spec().outputs[0].accepts_manifest(&draft.manifest));

        let expected_path = workdir.path().join(DATA_FILE);
        assert_eq!(draft.payload, DraftPayload::File(expected_path.clone()));
        assert!(expected_path.is_file());
    }

    #[tokio::test]
    async fn ingest_csv_cache_policy_extra_hash_changes_after_modification() {
        let dir = tempfile::tempdir().unwrap();
        let csv = write_csv(dir.path(), "src.csv", "x\n1\n");
        let workdir = tempfile::tempdir().unwrap();
        let (ctx, _cancel) = make_ctx(workdir.path(), json!({ "path": csv }));
        let block = IngestCsv::new();

        let p1 = block.cache_policy(&ctx).await.unwrap();
        let p2 = block.cache_policy(&ctx).await.unwrap();
        assert_eq!(p1, p2);
        let CachePolicy::Deterministic {
            extra_hash: Some(h1),
            ..
        } = p1
        else {
            panic!("expected Deterministic with extra_hash");
        };

        std::fs::write(&csv, "x\n1\n2\n").unwrap();
        let CachePolicy::Deterministic {
            extra_hash: Some(h2),
            ..
        } = block.cache_policy(&ctx).await.unwrap()
        else {
            panic!("expected Deterministic with extra_hash");
        };
        assert_ne!(h1, h2);

        // Missing file fails cache_policy.
        let (missing_ctx, _cancel2) = make_ctx(
            workdir.path(),
            json!({ "path": dir.path().join("missing.csv") }),
        );
        assert!(block.cache_policy(&missing_ctx).await.is_err());
    }

    #[tokio::test]
    async fn ingest_csv_same_input_produces_identical_parquet_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let csv = write_csv(dir.path(), "det.csv", "id,name,val\n1,a,1.5\n2,b,2.5\n");
        let block = IngestCsv::new();

        let run_once = || async {
            let workdir = tempfile::tempdir().unwrap();
            let (ctx, _cancel) = make_ctx(workdir.path(), json!({ "path": csv }));
            block.run(&ctx, Inputs::new()).await.unwrap();
            std::fs::read(workdir.path().join(DATA_FILE)).unwrap()
        };

        let bytes_a = run_once().await;
        let bytes_b = run_once().await;
        assert!(!bytes_a.is_empty());
        assert_eq!(bytes_a, bytes_b);
    }

    #[tokio::test]
    async fn ingest_csv_respects_cancellation_and_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let csv = write_csv(dir.path(), "ok.csv", "x\n1\n");
        let workdir = tempfile::tempdir().unwrap();
        let (ctx, cancel) = make_ctx(workdir.path(), json!({ "path": csv }));
        cancel.cancel();

        let err = IngestCsv::new().run(&ctx, Inputs::new()).await.unwrap_err();
        assert!(matches!(err, Error::Cancelled));

        let (missing_ctx, _cancel2) = make_ctx(
            workdir.path(),
            json!({ "path": dir.path().join("nope.csv") }),
        );
        let err = IngestCsv::new()
            .run(&missing_ctx, Inputs::new())
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Runtime(_)));
    }
}
